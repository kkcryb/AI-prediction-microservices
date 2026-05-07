# 封装 GCN-LSTM 前向推理逻辑 (1小时级调用)
"""
app/services/inference_service.py

功能描述：
AI 推理服务核心。负责加载模型、执行基准负荷预测，
并调度 elasticity_service 进行弹性系数推演，最终组合返回给路由层。
"""

import torch
import numpy as np
import logging
from typing import Dict, Any

from app.models.gcn_lstm import Gcnlstm
from app.core.config import settings
from app.core.exceptions import ModelInferenceException

# 引入刚刚写好的 elasticity_service
from app.services.elasticity_service import elasticity_service

logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        logger.info(f"AI基准模型就绪，运行设备: {self.device}")

    def _load_model(self) -> torch.nn.Module:
        # (保持原有的模型加载逻辑不变...)
        model = Gcnlstm(
            num_nodes=settings.NUM_NODES,
            input_dim=settings.INPUT_DIM,
            hidden_dim=settings.HIDDEN_DIM,
            seq_len=settings.SEQ_LEN,
            pre_len=settings.PRE_LEN
        ).to(self.device)
        state_dict = torch.load(settings.MODEL_WEIGHTS_PATH, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def predict_and_evaluate_elasticity(self,
                                        historical_features: np.ndarray,
                                        adj_matrix: np.ndarray) -> Dict[str, Any]:
        """
        统筹推理与经济计算。
        """
        try:
            # 1. 准备张量
            x_tensor = torch.FloatTensor(historical_features).to(self.device)
            adj_tensor = torch.FloatTensor(adj_matrix).to(self.device)

            # 2. 核心AI任务：基准负荷预测
            with torch.no_grad():
                base_pred_tensor = self.model(x_tensor, adj_tensor)
            base_pred_np = base_pred_tensor.cpu().numpy()

            # 3. 移交经济计算任务：调用 elasticity_service
            elasticity_matrix = elasticity_service.calculate_elasticity(
                model=self.model,  # 将模型引用传给弹性服务，以供其做扰动推理
                x_tensor=x_tensor,
                adj_tensor=adj_tensor,
                base_pred_np=base_pred_np
            )

            # 4. 组装响应
            result = {
                "status": "success",
                "predicted_load": base_pred_np.tolist(),
                "elasticity_coefficient": elasticity_matrix.tolist(),
                "macro_horizon_steps": base_pred_np.shape[1]
            }

            logger.info("基准预测与弹性系数提取均计算完成。")
            return result

        except Exception as e:
            logger.error(f"推理服务执行异常: {e}", exc_info=True)
            raise ModelInferenceException(f"Inference failed: {str(e)}")


inference_service = InferenceService()