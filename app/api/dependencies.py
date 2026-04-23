# 依赖注入 (如获取数据库客户端、加载模型实例)
# app/api/dependencies.py

import logging
import torch
import joblib
import pandas as pd
from functools import lru_cache
from fastapi import HTTPException, status, Depends
from typing import Tuple, Any

# 导入配置和底层模型结构
from app.core.config import settings
from app.models.gcn_lstm import Gcnlstm  # 假设在 app/models/gcn_lstm.py 中定义的模型类
from app.services.data_client import DataClient  # 假设存在的数据中台对接服务

logger = logging.getLogger(__name__)


# ==========================================
# 1. 单例加载: 归一化组件 (Scaler)
# ==========================================
@lru_cache(maxsize=1)
def load_scaler() -> Any:
    """
    使用 lru_cache 实现单例模式，将全局特征归一化器（如MinMaxScaler/StandardScaler）常驻内存。
    """
    try:
        scaler_path = settings.SCALER_PATH
        logger.info(f"正在加载数据归一化组件 (Scaler)... 路径: {scaler_path}")
        scaler = joblib.load(scaler_path)
        logger.info("Scaler 加载成功。")
        return scaler
    except FileNotFoundError:
        logger.error(f"Scaler 文件未找到: {settings.SCALER_PATH}")
        raise RuntimeError(f"Scaler 文件未找到，请检查路径: {settings.SCALER_PATH}")
    except Exception as e:
        logger.error(f"加载 Scaler 时发生异常: {str(e)}")
        raise RuntimeError(f"Scaler 加载失败: {str(e)}")


# ==========================================
# 2. 单例加载: AI 预测大脑 (GCN-LSTM 模型)
# ==========================================
@lru_cache(maxsize=1)
def load_model() -> Tuple[torch.nn.Module, torch.device]:
    """
    使用 lru_cache 缓存加载好的 PyTorch 模型和目标计算设备。
    确保模型仅在第一次请求（或启动时预热）被加载一次。
    """
    # 自动识别计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"AI微服务当前推理设备设为: {device}")

    try:
        # 1. 加载并处理邻接矩阵 (Topology)
        logger.info(f"正在加载空间邻接矩阵... 路径: {settings.ADJ_MATRIX_PATH}")
        # 假设 csv 没有表头，全都是数值
        adj_df = pd.read_csv(settings.ADJ_MATRIX_PATH, header=None)

        # 转换为 PyTorch Tensor，并【关键步骤】直接发送到目标计算设备
        # 这能避免模型源码中 self.A = a_delta 无法被 .to(device) 覆盖导致的数据跨设备计算报错
        adj_dense = torch.tensor(adj_df.values, dtype=torch.float32).to(device)

        # 验证矩阵维度是否与节点数一致
        if adj_dense.shape[0] != settings.NUM_NODES:
            logger.warning(f"邻接矩阵维度 ({adj_dense.shape}) 与设定节点数 ({settings.NUM_NODES}) 不一致，请注意！")

        # 2. 实例化 Gcnlstm 模型
        logger.info(f"正在初始化 Gcnlstm 模型结构...")
        model = Gcnlstm(
            seq=settings.SEQ_LEN,
            n_fea=settings.NUM_FEAT,
            adj_dense=adj_dense,  # 注入计算好的邻接矩阵张量
            node=settings.NUM_NODES,
            gcn_out=settings.GCN_OUT,
            gcn_layers=settings.GCN_LAYERS,
            lstm_hidden_dim=settings.LSTM_HIDDEN_DIM,
            lstm_layers=settings.LSTM_LAYERS,
            hidden_dim=settings.HIDDEN_DIM
        )

        # 3. 加载预训练权重
        logger.info(f"正在加载模型权重... 路径: {settings.MODEL_WEIGHT_PATH}")
        state_dict = torch.load(settings.MODEL_WEIGHT_PATH, map_location=device)
        model.load_state_dict(state_dict)

        # 发送整个模型到目标设备并切换至推理模式
        model.to(device)
        model.eval()

        logger.info("Gcnlstm 模型初始化并加载权重完毕。")
        return model, device

    except FileNotFoundError as fnf:
        logger.error(f"文件未找到: {str(fnf)}")
        raise RuntimeError(f"所需文件丢失: {str(fnf)}")
    except Exception as e:
        logger.error(f"加载模型时发生异常: {str(e)}")
        raise RuntimeError(f"模型初始化失败: {str(e)}")


# ==========================================
# 3. FastAPI 依赖注入函数 (供 Route 使用)
# ==========================================

def get_scaler() -> Any:
    """依赖注入: 获取 Scaler 实例"""
    try:
        return load_scaler()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"内部服务异常(Scaler未就绪): {str(e)}"
        )


def get_model_and_device() -> Tuple[torch.nn.Module, torch.device]:
    """依赖注入: 获取 GCN_LSTM 模型与计算设备上下文"""
    try:
        return load_model()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI推理引擎加载失败: {str(e)}"
        )


def get_data_client() -> DataClient:
    """
    依赖注入: 获取后端数据中台的请求客户端。
    保证每次请求时都能拿到带有正确验证 Key 和 URL的客户端。
    """
    return DataClient(
        base_url=settings.DATA_CENTER_API_URL,
        api_key=settings.DATA_CENTER_API_KEY
    )