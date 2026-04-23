import os
import pandas as pd
from influxdb_client import InfluxDBClient, Point

# ====================
# 你的配置（正确无误）
# ====================
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "r9JyPrSWd43AiXptUeiVtWrKk1Ut38kjDxqBdfxeBDLVHj4wOrCgZ1_x8-Q-0oIeO9hkZKXWkSFt8PMZYiOccQ=="
INFLUX_ORG = "my-org"
INFLUX_BUCKET = "raw_data"

# 连接数据库（超稳定模式）
client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG, timeout=120000)
write_api = client.write_api()

# 数据目录（你现在的正确路径）
data_path = r"D:\python\Scripts\pythonProject21\data"

# ------------------------------
# 导入时序数据（ occupancy, volume 等）
# ------------------------------
def import_time_series(file, measurement):
    print(f"📥 导入 {file} -> {measurement}")
    full = os.path.join(data_path, file)
    df = pd.read_csv(full, index_col=0)

    for time_idx, row in df.iterrows():
        for station, value in row.items():
            try:
                p = Point(measurement)\
                    .tag("station_id", str(station))\
                    .field("value", float(value))\
                    .time(pd.to_datetime(time_idx))

                write_api.write(INFLUX_BUCKET, INFLUX_ORG, p)
            except:
                continue

    print(f"✅ {measurement} 导入完成！")

# ------------------------------
# 开始一键导入
# ------------------------------
print("🚀 开始导入所有 UrbanEV 数据\n")

import_time_series("occupancy.csv",    "occupancy")
import_time_series("volume.csv",       "volume")
import_time_series("e_price.csv",      "e_price")
import_time_series("s_price.csv",      "s_price")
import_time_series("duration.csv",     "duration")

print("\n🎉🎉🎉 全部导入成功！数据库已经有数据了！")
write_api.close()
client.close()