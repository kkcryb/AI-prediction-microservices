# 实现微小扰动法，计算电价弹性系数矩阵
"""
app/services/elasticity_service.py

功能描述：
专门负责通过“微观扰动法”计算电价-负荷弹性系数。
将电力经济学逻辑与底层深度学习推理逻辑解耦。
"""

import torch
import numpy as np
import logging
from typing import Callable
from app.core.config import settings

logger = logging.getLogger(__name__)


class ElasticityService:
    def __init__(self):
        # 价格扰动比例 (如 1%)
        self.delta_p_ratio = settings.PRICE_PERTURBATION_RATIO if hasattr(settings,
                                                                          'PRICE_PERTURBATION_RATIO') else 0.01
        # 电价特征在输入维度中的索引
        self.price_idx = settings.PRICE_FEATURE_INDEX
        # 极小值，防止除以 0
        self.epsilon = 1e-8

    def calculate_elasticity(self,
                             model: torch.nn.Module,
                             x_tensor: torch.Tensor,
                             adj_tensor: torch.Tensor,
                             base_pred_np: np.ndarray) -> np.ndarray:
        """
        通过对价格特征施加扰动，二次调用模型，计算弹性系数矩阵。

        :param model: 已加载的 GCN-LSTM 模型
        :param x_tensor: 原始历史输入张量
        :param adj_tensor: 空间拓扑邻接矩阵张量
        :param base_pred_np: 基准情况下的负荷预测结果 (NumPy)
        :return: 弹性系数矩阵 (NumPy)
        """
        try:
            # 1. 拷贝原始特征，防止污染
            perturbed_x = x_tensor.clone()

            # 2. 仅对电价维度施加正向扰动（模拟电价上调）
            perturbed_x[:, :, :, self.price_idx] = perturbed_x[:, :, :, self.price_idx] * (1.0 + self.delta_p_ratio)

            # 3. 闭包内进行二次前向推理（无需梯度）
            with torch.no_grad():
                perturbed_pred_tensor = model(perturbed_x, adj_tensor)

            pert_pred_np = perturbed_pred_tensor.cpu().numpy()

            # 4. 计算负荷变化率 ΔQ / Q
            delta_q_ratio = (pert_pred_np - base_pred_np) / (base_pred_np + self.epsilon)

            # 5. 计算最终弹性系数 E = (ΔQ / Q) / (ΔP / P)
            elasticity_matrix = delta_q_ratio / self.delta_p_ratio

            # 6. 工程优化：截断极端异常值，防止下游运筹优化器(Geatpy)因为系数量级过大而发散寻找不到帕累托最优解
            elasticity_matrix = np.clip(elasticity_matrix, -10.0, 10.0)

            return elasticity_matrix

        except Exception as e:
            logger.error(f"弹性系数计算失败: {e}", exc_info=True)
            raise


# 导出单例
elasticity_service = ElasticityService()