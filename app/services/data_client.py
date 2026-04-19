# HTTP/InfluxDB 客户端: 向数据中台请求标准化的历史时序数据
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
import logging

# 假设使用 InfluxDB v2 的官方异步客户端
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from fastapi import HTTPException

# 导入我们刚刚定义的数据契约
from app.schemas.request_models import HistoricalFeature

# 导入系统配置 (假设在 app/core/config.py 中已定义)
# from app.core.config import settings

logger = logging.getLogger(__name__)


class DataClient:
    """
    数据访问服务类
    负责从 InfluxDB 或本地仿真数据集中拉取过去 24 小时的时空特征数据
    """

    def __init__(self):
        # 实际开发中，这些配置应从 settings 中获取
        self.influx_url = os.getenv("INFLUXDB_URL", "http://localhost:8086")
        self.influx_token = os.getenv("INFLUXDB_TOKEN", "your_super_secret_token")
        self.influx_org = os.getenv("INFLUXDB_ORG", "urbanev_org")
        self.influx_bucket = os.getenv("INFLUXDB_BUCKET", "urbanev_data")

        # 仿真模式开关：如果为 True，则从本地 CSV 读取，否则连接 InfluxDB
        self.use_simulation_data = os.getenv("USE_SIMULATION_DATA", "True").lower() == "true"
        self.local_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "training",
                                           "data")

    async def get_historical_features(
            self,
            region_id: str,
            current_time: datetime,
            history_horizon: int = 24
    ) -> List[HistoricalFeature]:
        """
        核心对外接口：获取指定区域、指定时间点往前的历史特征
        """
        start_time = current_time - timedelta(hours=history_horizon)

        try:
            if self.use_simulation_data:
                logger.info(f"[{region_id}] 从本地 CSV 仿真数据集读取 {start_time} 至 {current_time} 的数据")
                return await self._fetch_from_local_csv(region_id, start_time, current_time)
            else:
                logger.info(f"[{region_id}] 从 InfluxDB 读取 {start_time} 至 {current_time} 的数据")
                return await self._fetch_from_influxdb(region_id, start_time, current_time)
        except Exception as e:
            logger.error(f"获取历史数据失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch historical data: {str(e)}")

    async def _fetch_from_influxdb(
            self,
            region_id: str,
            start_time: datetime,
            end_time: datetime
    ) -> List[HistoricalFeature]:
        """
        从 InfluxDB 实时数据库中拉取数据 (使用 Flux 查询语言)
        """
        # 将 datetime 转换为 InfluxDB 接受的 RFC3339 格式
        start_str = start_time.isoformat() + "Z"
        end_str = end_time.isoformat() + "Z"

        # 编写 Flux 查询语句，利用 pivot 将多条 field 记录转为宽表格式
        flux_query = f"""
            from(bucket: "{self.influx_bucket}")
                |> range(start: {start_str}, stop: {end_str})
                |> filter(fn: (r) => r["_measurement"] == "station_status")
                |> filter(fn: (r) => r["region_id"] == "{region_id}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                |> sort(columns: ["_time"])
        """

        features = []
        async with InfluxDBClientAsync(url=self.influx_url, token=self.influx_token, org=self.influx_org) as client:
            query_api = client.query_api()
            # 异步执行查询
            result = await query_api.query(query=flux_query)

            # 解析 Flux 结果并映射到 Pydantic 模型
            for table in result:
                for record in table.records:
                    features.append(HistoricalFeature(
                        timestamp=record.get_time(),
                        load=float(record.values.get("load", 0.0)),
                        price=float(record.values.get("price", 0.0)),
                        pv_generation=float(record.values.get("pv_generation", 0.0)),
                        temperature=float(record.values.get("temperature")) if record.values.get(
                            "temperature") else None
                    ))

        if not features:
            logger.warning(f"InfluxDB 查询结果为空 (region: {region_id}, time: {start_time} - {end_time})")

        return features

    async def _fetch_from_local_csv(
            self,
            region_id: str,
            start_time: datetime,
            end_time: datetime
    ) -> List[HistoricalFeature]:
        """
        仿真回测模式：从 training/data 目录下的 CSV 文件中读取数据
        应对开发阶段脱机测试的需求
        """
        try:
            # 读取对应的负荷和价格数据 (你需要根据你实际的 CSV 结构进行调整)
            # 假设 volume.csv 和 e_price.csv 的行是时间，列是 region_id
            volume_path = os.path.join(self.local_data_dir, "volume.csv")
            price_path = os.path.join(self.local_data_dir, "e_price.csv")

            df_volume = pd.read_csv(volume_path, parse_dates=['timestamp'], index_col='timestamp')
            df_price = pd.read_csv(price_path, parse_dates=['timestamp'], index_col='timestamp')

            # 根据时间区间和区域切片
            df_vol_slice = df_volume.loc[start_time:end_time, region_id]
            df_price_slice = df_price.loc[start_time:end_time, region_id]

            features = []
            for ts in df_vol_slice.index:
                # 排除完全对齐的结束时间（因为通常预测时域不包含当前时刻的实际值）
                if ts == end_time:
                    continue

                features.append(HistoricalFeature(
                    timestamp=ts,
                    load=float(df_vol_slice[ts]),
                    price=float(df_price_slice[ts]),
                    # 假设仿真阶段暂无 PV 或温度数据，填 0 默认处理
                    pv_generation=0.0,
                    temperature=25.0
                ))
            return features

        except KeyError:
            raise ValueError(f"找不到区域 {region_id} 的本地数据")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"本地数据集缺失: {str(e)}")


# 实例化一个单例供 FastAPI 依赖注入使用
data_client = DataClient()