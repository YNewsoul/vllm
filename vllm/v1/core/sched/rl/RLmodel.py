import torch
import torch.nn as nn
import logging

import torch.nn.functional as F

try:
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

class MLPNetwork(nn.Module):
    """
    轻量 MLP 网络
    输入：6维状态；输出：16个离散动作的Q值（评估动作长期价值）
    """
    def __init__(self, state_dim: int, action_dim: int):
        super(MLPNetwork, self).__init__()
        # 网络结构：2层隐藏层（24+16节点），平衡拟合能力与延迟
        self.network = nn.Sequential(
            nn.Linear(state_dim, 24),  # 输入层→隐藏层1
            nn.ReLU(),                 # 激活函数（计算快，缓解梯度消失）
            nn.Linear(24, 16),         # 隐藏层1→隐藏层2
            nn.ReLU(),
            nn.Linear(16, action_dim)  # 隐藏层2→输出层（16个动作Q值）
        )
        # 权重初始化
        self._init_weights()

    def _init_weights(self) -> None:
        """初始化权重，确保训练稳定"""
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Xavier初始化（避免梯度爆炸）
                nn.init.constant_(m.bias, 0.1)     # 偏置初始化（避免初始输出过小）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return self.network(x)

class DualAttentionNetwork(nn.Module):
    """
    Dual Attention Network（支持 mask、初始化、可返回 attention 权重）
    输入:
        global_vec: [B, G]
        wait_arr: [B, K_wait, F_wait]
        run_arr:  [B, K_run, F_run]
        wait_mask: [B, K_wait] (optional) 1=valid,0=pad
        run_mask:  [B, K_run]  (optional)
    输出:
        mode='discrete' -> Q-values [B, action_dim]
        mode='continuous' -> (mu [B,2], logstd [B,2])
    forward 支持 return_attn=True 返回 (out, wait_weights, run_weights)
    """
    def __init__(self, G: int, K_wait: int, F_wait: int, K_run: int, F_run: int,
                 action_dim: int, hidden=256, mode="discrete"):
        """
        G: 全局状态维度
        K_wait: 等待队列view长度上限长度
        F_wait: 等待队列特征维度
        K_run: 运行队列view长度上限长度
        F_run: 运行队列特征维度
        action_dim: 动作维度
        hidden: 隐藏层维度
        mode: 输出模式（"discrete"或"continuous"）
        """
        super().__init__()
        self.mode = mode

        # 分别将等待队列和运行队列的特征编码为64维向量
        self.wait_encoder = nn.Sequential(
            nn.Linear(F_wait, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        # 等待队列注意力机制，产生一个标量 score，用来做 softmax 注意力权重
        self.wait_att = nn.Linear(64, 1)

        # running request encoder (separate weights)
        self.run_encoder = nn.Sequential(
            nn.Linear(F_run, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.run_att = nn.Linear(64, 1)

        # 把 global + pooled_wait + pooled_run 的信息混合、抽象，输出动作 Q 值或策略参数
        self.backbone = nn.Sequential(
            nn.Linear(G + 64 + 64, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU()
        )

        if self.mode == "discrete":
            self.head = nn.Linear(hidden//2, action_dim)
        else:
            self.mu = nn.Linear(hidden//2, 2)
            # learnable logstd scalar per action-dim
            self.logstd = nn.Parameter(torch.zeros(2))

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _masked_softmax(self, scores: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8):
        """
        scores: [B, K]
        mask:   [B, K], 1 for valid, 0 for pad. If mask is None -> normal softmax
        返回: weights [B, K]
        """
        if mask is None:
            return F.softmax(scores, dim=dim)

        neg_inf = -1e9
        scores_masked = scores.masked_fill(mask == 0, neg_inf)

        mask_sum = mask.sum(dim=dim, keepdim=True)
        all_pad = (mask_sum == 0)

        # 对全 pad 行单独处理，不在原 tensor 上改
        safe_scores = torch.where(all_pad, torch.zeros_like(scores_masked), scores_masked)

        weights = F.softmax(safe_scores, dim=dim)

        # 再用 mask 清零无效位置（非原地）
        weights = weights * mask.float()

        # 最后对 all-pad 行显式归零（非原地）
        weights = torch.where(all_pad, torch.zeros_like(weights), weights)

        return weights

    def forward(self, global_vec: torch.Tensor, wait_arr: torch.Tensor, run_arr: torch.Tensor,
                wait_mask: torch.Tensor = None, run_mask: torch.Tensor = None, return_attn: bool = False):
        """
        wait_mask, run_mask: tensors of 0/1 with same device/dtype as global_vec (or None)
        """
        device = global_vec.device
        dtype = global_vec.dtype

        B = global_vec.size(0)

        # --- waiting encoding & attention ---
        Kw = wait_arr.size(1)
        if Kw > 0:
            wait_flat = wait_arr.view(B * Kw, -1).to(device=device, dtype=dtype)
            wait_enc = self.wait_encoder(wait_flat).view(B, Kw, -1)  # [B, Kw, 64]
            wait_scores = self.wait_att(wait_enc).squeeze(-1)  # [B, Kw]
            if wait_mask is not None:
                wait_mask = wait_mask.to(device=device)
            wait_weights = self._masked_softmax(wait_scores, wait_mask, dim=1)  # [B, Kw]
            pooled_wait = (wait_weights.unsqueeze(-1) * wait_enc).sum(dim=1)  # [B, 64]
        else:
            pooled_wait = torch.zeros(B, 64, device=device, dtype=dtype)
            wait_weights = torch.zeros(B, 0, device=device, dtype=dtype)

        # --- running encoding & attention ---
        Kr = run_arr.size(1)
        if Kr > 0:
            run_flat = run_arr.view(B * Kr, -1).to(device=device, dtype=dtype)
            run_enc = self.run_encoder(run_flat).view(B, Kr, -1)  # [B, Kr, 64]
            run_scores = self.run_att(run_enc).squeeze(-1)  # [B, Kr]
            if run_mask is not None:
                run_mask = run_mask.to(device=device)
            run_weights = self._masked_softmax(run_scores, run_mask, dim=1)  # [B, Kr]
            pooled_run = (run_weights.unsqueeze(-1) * run_enc).sum(dim=1)  # [B, 64]
        else:
            pooled_run = torch.zeros(B, 64, device=device, dtype=dtype)
            run_weights = torch.zeros(B, 0, device=device, dtype=dtype)

        # --- concat + backbone ---
        gv = global_vec.to(device=device, dtype=dtype)
        x = torch.cat([gv, pooled_wait, pooled_run], dim=1)
        h = self.backbone(x)

        if self.mode == "discrete":
            out = self.head(h)
        else:
            mu = self.mu(h)
            # clamp logstd to avoid extreme variance
            logstd = torch.clamp(self.logstd, min=-6.0, max=1.0)
            logstd = logstd.expand_as(mu)
            out = (mu, logstd)

        if return_attn:
            return out, wait_weights, run_weights
        return out