import torch
import torch.nn as nn
import logging

import torch.nn.functional as F

try:
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

class DualAttentionNetwork(nn.Module):
    """
    双注意力网络 (Dual Attention Network) - 专为调度系统设计的深度学习模型
    
    这个网络用于强化学习中的DQN算法，它能够处理调度系统中的三种关键信息：
    1. 全局系统状态
    2. 等待队列中的请求信息
    3. 正在运行的请求信息
    
    通过注意力机制，网络可以自动学习哪些请求更重要，从而做出更好的调度决策。
    
    输入数据格式:
        global_vec: [批量大小, 全局状态维度] - 表示系统的整体状态
        wait_arr: [批量大小, 等待队列长度, 等待队列特征维度] - 等待处理的请求信息
        run_arr: [批量大小, 运行队列长度, 运行队列特征维度] - 正在运行的请求信息
        wait_mask: [批量大小, 等待队列长度] (可选) - 标记等待队列中哪些位置是有效的
        run_mask: [批量大小, 运行队列长度] (可选) - 标记运行队列中哪些位置是有效的
    """
    def __init__(self, G: int, K_wait: int, F_wait: int, K_run: int, F_run: int,
                 action_dim: int, hidden=256):
        """
        G: 全局状态维度
        K_wait: 等待队列view长度上限长度
        F_wait: 等待队列特征维度
        K_run: 运行队列view长度上限长度
        F_run: 运行队列特征维度
        action_dim: 动作维度
        hidden: 隐藏层维度
        """
        super().__init__()

        # 等待队列编码器 - 将每个等待请求的特征转换为64维的向量表示
        # 两层全连接网络，使用ReLU激活函数增加非线性能力
        self.wait_encoder = nn.Sequential(
            nn.Linear(F_wait, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        # 等待队列注意力层 - 计算每个等待请求的重要性分数
        # 输出单个标量值，表示该请求的重要程度
        self.wait_att = nn.Linear(64, 1)

        # 运行队列编码器 - 与等待队列编码器结构相同但参数独立
        self.run_encoder = nn.Sequential(
            nn.Linear(F_run, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        # 运行队列注意力层
        self.run_att = nn.Linear(64, 1)

        # 场景嵌入 + FiLM 调制（两类 RL 场景）
        self.scenario_embed = nn.Embedding(2, 8)
        # 生成缩放/偏置，用于调制 wait/run 的池化特征（各 128 维）
        self.film = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 512)  # -> gamma(256)+beta(256) -> wait/run 各 128
        )

        # 主网络骨干 - 将所有信息融合并进行高级特征提取
        # 输入是全局状态 + 等待队列池化结果 + 运行队列池化结果
        self.backbone = nn.Sequential(
            nn.Linear(G + 256 + 8, hidden),  # 融合所有特征,256 = wait_att+wait_max+run_att+run_max
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),  # 降维处理
            nn.ReLU()
        )

        self.value_head = nn.Linear(hidden // 2, 1)
        self.adv_head = nn.Linear(hidden // 2, action_dim)

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
        带掩码的softmax函数 - 处理变长序列
        由于不同样本可能有不同长度的队列，需要掩码机制来忽略填充的无效数据
        参数:
            scores: 注意力分数，形状[批量大小, 序列长度]
            mask: 掩码张量，值为1表示有效，0表示无效
            dim: 应用softmax的维度
            eps: 防止数值不稳定的小值
        返回:
            weights: 归一化后的注意力权重
        """
        # 如果没有掩码，直接使用普通softmax
        if mask is None:
            return F.softmax(scores, dim=dim)

        # 对于无效位置，将分数设为非常小的值（负无穷）
        neg_inf = -1e9
        scores_masked = scores.masked_fill(mask == 0, neg_inf)

        # 检查哪些行全是无效数据
        mask_sum = mask.sum(dim=dim, keepdim=True)
        all_pad = (mask_sum == 0)

        # 对全是无效数据的行，将分数设为0而不是负无穷，避免NaN
        safe_scores = torch.where(all_pad, torch.zeros_like(scores_masked), scores_masked)

        # 应用softmax得到权重
        weights = F.softmax(safe_scores, dim=dim)

        # 确保无效位置的权重为0
        weights = weights * mask.float()

        # 对全是无效数据的行，显式将权重设为0
        weights = torch.where(all_pad, torch.zeros_like(weights), weights)

        return weights

    def forward(self, global_vec: torch.Tensor, wait_arr: torch.Tensor, run_arr: torch.Tensor,
                wait_mask: torch.Tensor = None, run_mask: torch.Tensor = None, scenario_id: torch.Tensor = None):
        """
        网络前向传播过程 - 处理输入数据并生成输出
        """
        device = global_vec.device
        dtype = global_vec.dtype

        # 获取批量大小
        B = global_vec.size(0)
        
        # 定义负无穷，用于 Max Pooling 的掩码处理
        neg_inf = -1e9

        # 场景 id（无则默认 0）
        if scenario_id is None:
            scenario_id = torch.zeros(B, dtype=torch.long, device=device)
        
        scen = self.scenario_embed(scenario_id)  # [B, 8]
        film_params = self.film(scen)            # [B, 512]
        gamma, beta = film_params.chunk(2, dim=-1)       # 各 [B, 256]
        gamma_wait, gamma_run = gamma.chunk(2, dim=-1)   # 各 [B, 128]
        beta_wait, beta_run = beta.chunk(2, dim=-1)      # 各 [B, 128]

        # 1. 处理等待队列信息
        # 等待队列长度
        Kw = wait_arr.size(1) 
        if Kw > 0:
            # 1.1 编码
            wait_flat = wait_arr.view(B * Kw, -1).to(device=device, dtype=dtype)
            wait_enc = self.wait_encoder(wait_flat).view(B, Kw, -1)  # [B, Kw, 64]
            
            # 1.2 Attention Pooling (关注整体分布)
            wait_scores = self.wait_att(wait_enc).squeeze(-1)  # [B, Kw]
            if wait_mask is not None:
                wait_mask = wait_mask.to(device=device)
            wait_weights = self._masked_softmax(wait_scores, wait_mask, dim=1)  # [B, Kw]
            pooled_wait_att = (wait_weights.unsqueeze(-1) * wait_enc).sum(dim=1)  # [B, 64]
            
            # 1.3 Max Pooling (捕捉最极端/紧急的特征)
            if wait_mask is not None:
                # 将 mask 扩展到特征维度 [B, Kw, 64]
                mask_exp = wait_mask.unsqueeze(-1).expand_as(wait_enc)
                # 将无效位置(mask=0)填为负无穷，确保 max 不会选中它们
                wait_enc_masked = wait_enc.masked_fill(mask_exp == 0, neg_inf)
                # 取最大值
                pooled_wait_max = wait_enc_masked.max(dim=1)[0] # [B, 64]
                
                # 特殊处理：如果某一样本全是 Padding (空队列)，max 结果会是 neg_inf
                # 需要将其重置为 0
                is_all_pad = (wait_mask.sum(dim=1) == 0).unsqueeze(-1) # [B, 1]
                pooled_wait_max = pooled_wait_max.masked_fill(is_all_pad, 0.0)
            else:
                pooled_wait_max = wait_enc.max(dim=1)[0]
                
        else:
            # 队列为空时的默认值
            pooled_wait_att = torch.zeros(B, 64, device=device, dtype=dtype)
            pooled_wait_max = torch.zeros(B, 64, device=device, dtype=dtype)
            wait_weights = torch.zeros(B, 0, device=device, dtype=dtype)

        # 2. 处理运行队列信息 (Running Queue)
        Kr = run_arr.size(1)
        if Kr > 0:
            # 2.1 编码
            run_flat = run_arr.view(B * Kr, -1).to(device=device, dtype=dtype)
            run_enc = self.run_encoder(run_flat).view(B, Kr, -1)  # [B, Kr, 64]
            
            # 2.2 Attention Pooling
            run_scores = self.run_att(run_enc).squeeze(-1)  # [B, Kr]
            if run_mask is not None:
                run_mask = run_mask.to(device=device)
            run_weights = self._masked_softmax(run_scores, run_mask, dim=1)  # [B, Kr]
            pooled_run_att = (run_weights.unsqueeze(-1) * run_enc).sum(dim=1)  # [B, 64]
            
            # 2.3 Max Pooling
            if run_mask is not None:
                mask_exp = run_mask.unsqueeze(-1).expand_as(run_enc)
                run_enc_masked = run_enc.masked_fill(mask_exp == 0, neg_inf)
                pooled_run_max = run_enc_masked.max(dim=1)[0] # [B, 64]
                
                # 处理全空情况
                is_all_pad = (run_mask.sum(dim=1) == 0).unsqueeze(-1)
                pooled_run_max = pooled_run_max.masked_fill(is_all_pad, 0.0)
            else:
                pooled_run_max = run_enc.max(dim=1)[0]
        else:
            pooled_run_att = torch.zeros(B, 64, device=device, dtype=dtype)
            pooled_run_max = torch.zeros(B, 64, device=device, dtype=dtype)
            run_weights = torch.zeros(B, 0, device=device, dtype=dtype)
        

        # FiLM 调制（条件化场景）
        wait_feat = torch.cat([pooled_wait_att, pooled_wait_max], dim=1)  # [B,128]
        run_feat = torch.cat([pooled_run_att, pooled_run_max], dim=1)     # [B,128]
        wait_feat = wait_feat * (1 + gamma_wait) + beta_wait
        run_feat = run_feat * (1 + gamma_run) + beta_run

        # 3. 融合所有信息并生成输出
        gv = global_vec.to(device=device, dtype=dtype)
        
        x = torch.cat([gv, wait_feat, run_feat, scen], dim=1)  # [B, G+256+8]
        
        h = self.backbone(x)

        value = self.value_head(h)                 # [B,1]
        adv = self.adv_head(h)                     # [B,A]
        adv_mean = adv.mean(dim=1, keepdim=True)
        out = value + (adv - adv_mean)

        return out