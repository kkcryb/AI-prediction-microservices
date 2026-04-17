# 环境变量加载 (如 InfluxDB 地址, 模型路径等)
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # --- 基础配置 ---
    PROJECT_NAME: str = "V2G AI Prediction Microservice"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # --- 模型路径配置 ---
    # 默认指向 checkpoints 目录，Docker 部署时可通过环境变量覆盖
    MODEL_WEIGHT_PATH: str = os.getenv("MODEL_WEIGHT_PATH", "checkpoints/best_gcn_lstm.pth")
    SCALER_PATH: str = os.getenv("SCALER_PATH", "checkpoints/scaler.pkl")

    # --- 接口与数据中台对接配置 ---
    # 根据项目要求，AI组需要从后端数据中台拉取标准历史时序数据
    DATA_CENTER_API_URL: str = os.getenv("DATA_CENTER_API_URL", "http://data-center:8000/api/v1/historical-data")
    DATA_CENTER_API_KEY: str = os.getenv("DATA_CENTER_API_KEY", "default_secret_key")

    # --- 算法超参数配置 (针对微小扰动法) ---
    PERTURBATION_DELTA: float = 0.05  # 微小扰动法中电价变动的比例 (例如 5%)

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True


# 实例化配置对象，在其他文件中只需 `from app.core.config import settings` 即可使用
settings = Settings()