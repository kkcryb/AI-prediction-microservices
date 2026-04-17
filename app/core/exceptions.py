# 全局异常处理类
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# ==========================================
# 1. 自定义业务异常类
# ==========================================

class DataFetchException(Exception):
    """向后端数据中台请求数据时发生的异常"""
    def __init__(self, detail: str):
        self.detail = detail

class ModelInferenceException(Exception):
    """模型前向传播或弹性系数计算时发生的异常"""
    def __init__(self, detail: str):
        self.detail = detail

class ModelNotLoadedException(Exception):
    """模型权重尚未加载到内存时的异常"""
    def __init__(self, detail: str = "AI Model is not loaded into memory."):
        self.detail = detail

# ==========================================
# 2. 注册全局异常处理器
# ==========================================

def setup_exception_handlers(app: FastAPI) -> None:
    """
    在 main.py 中调用此函数，将自定义异常绑定到 FastAPI 实例
    """
    @app.exception_handler(DataFetchException)
    async def data_fetch_exception_handler(request: Request, exc: DataFetchException):
        logger.error(f"DataFetchException: {exc.detail}")
        return JSONResponse(
            status_code=502,  # 502 Bad Gateway (依赖的下游服务出问题)
            content={
                "code": "DATA_FETCH_ERROR",
                "message": "未能成功从数据中台获取历史数据",
                "detail": exc.detail
            }
        )

    @app.exception_handler(ModelInferenceException)
    async def model_inference_exception_handler(request: Request, exc: ModelInferenceException):
        logger.error(f"ModelInferenceException: {exc.detail}")
        return JSONResponse(
            status_code=500,  # 500 Internal Server Error
            content={
                "code": "MODEL_INFERENCE_ERROR",
                "message": "时空图模型(GCN-LSTM)推理计算失败",
                "detail": exc.detail
            }
        )

    @app.exception_handler(ModelNotLoadedException)
    async def model_not_loaded_exception_handler(request: Request, exc: ModelNotLoadedException):
        logger.error(f"ModelNotLoadedException: {exc.detail}")
        return JSONResponse(
            status_code=503,  # 503 Service Unavailable
            content={
                "code": "MODEL_NOT_READY",
                "message": "AI大脑正在初始化，暂不可用",
                "detail": exc.detail
            }
        )