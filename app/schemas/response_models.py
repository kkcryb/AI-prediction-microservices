# 定义API返回载荷格式 (含预测向量与弹性系数JSON)
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class HourlyForecast(BaseModel):
    """单小时的预测结果与弹性属性"""
    timestamp: datetime = Field(..., description="预测的目标时间戳")
    baseline_load: float = Field(
        ...,
        description="基准预测负荷 (未施加V2G调控或电价调控下的自然负荷, kWh)"
    )
    elasticity_coefficient: float = Field(
        ...,
        description="该小时的电价弹性系数 (负荷变化率 / 价格变化率)。用于表征用户对价格的敏感度"
    )
    # 可选项：为运筹优化提供鲁棒性边界
    upper_bound: Optional[float] = Field(None, description="预测置信区间上限 (95%)")
    lower_bound: Optional[float] = Field(None, description="预测置信区间下限 (95%)")


class PredictResponse(BaseModel):
    """AI预测微服务响应体模型（直供运筹决策层）"""
    region_id: str = Field(..., description="区域唯一标识符")
    forecast_start_time: datetime = Field(..., description="预测开始时间")

    forecasts: List[HourlyForecast] = Field(
        ...,
        description="未来时段（如24小时）的序列预测结果，含负荷与弹性系数"
    )

    # 聚合级指标，方便监控大盘展示
    total_predicted_load: float = Field(
        ...,
        description="未来24小时预测总负荷 (kWh)"
    )
    average_elasticity: float = Field(
        ...,
        description="全天平均电价弹性系数"
    )

    status: str = Field("success", description="接口执行状态")
    message: Optional[str] = Field(None, description="附加信息或错误提示")

    class Config:
        schema_extra = {
            "example": {
                "region_id": "shenzhen_urban_ev_zone_A",
                "forecast_start_time": "2026-04-18T08:00:00",
                "total_predicted_load": 14500.5,
                "average_elasticity": -0.32,
                "forecasts": [
                    {
                        "timestamp": "2026-04-18T08:00:00",
                        "baseline_load": 850.2,
                        "elasticity_coefficient": -0.15,
                        "upper_bound": 880.0,
                        "lower_bound": 820.0
                    }
                ],
                "status": "success",
                "message": "Inference and elasticity calculation completed."
            }
        }