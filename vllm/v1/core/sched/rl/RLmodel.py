import torch
import torch.nn as nn
import logging

from vllm.logger import init_logger

logger = init_logger(__name__)

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