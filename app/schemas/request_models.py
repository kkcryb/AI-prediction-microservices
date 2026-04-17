# 定义API请求载荷格式
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class HistoricalFeature(BaseModel):
    """单时间步的历史特征数据结构"""
    timestamp: datetime = Field(..., description="时间戳")
    load: float = Field(..., description="该区域历史负荷 (kWh)")
    price: float = Field(..., description="该区域历史电价 (元/kWh)")
    pv_generation: float = Field(..., description="光伏发电量 (kWh)")
    # 可根据UrbanEV数据集扩展更多特征，如气象数据、星期几等
    temperature: Optional[float] = Field(None, description="气象温度")

class PredictRequest(BaseModel):
    """AI预测微服务请求体模型"""
    region_id: str = Field(
        ...,
        description="区域或边缘充电站的唯一标识符",
        example="shenzhen_urban_ev_zone_A"
    )
    current_time: datetime = Field(
        ...,
        description="当前调度基准时间 (如 8:00)，以此为界划分历史24h与未来24h",
        example="2026-04-18T08:00:00"
    )
    prediction_horizon: int = Field(
        24,
        description="预测未来时间步长（小时级调度默认预测未来24步）"
    )
    perturbation_ratio: float = Field(
        0.05,
        description="电价扰动法计算弹性系数时的价格变化比例（默认上浮/下浮 5%）"
    )
    historical_data: Optional[List[HistoricalFeature]] = Field(
        None,
        description="过去24小时的时序特征。若为空，则微服务内部调取 InfluxDB 获取"
    )

    class Config:
        schema_extra = {
            "example": {
                "region_id": "shenzhen_urban_ev_zone_A",
                "current_time": "2026-04-18T08:00:00",
                "prediction_horizon": 24,
                "perturbation_ratio": 0.05,
                "historical_data": None
            }
        }