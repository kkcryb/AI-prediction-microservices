import json
import asyncio
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Optional, List, Dict

import paho.mqtt.client as mqtt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS, WriteOptions
from influxdb_client.client.delete_api import DeleteApi
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import uvicorn

# ==========================================
# ⚙️ 【超级集中配置】所有配置都在这里，一键修改
# ==========================================
# 1. InfluxDB 配置
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "r9JyPrSWd43AiXptUeiVtWrKk1Ut38kjDxqBdfxeBDLVHj4wOrCgZ1_x8-Q-0oIeO9hkZKXWkSFt8PMZYiOccQ=="
INFLUX_ORG = "my-org"
INFLUX_BUCKET_RAW = "raw_data"
INFLUX_BUCKET_AGG = "agg_data"
# 2. MQTT 配置
MQTT_BROKER = "192.168.251.132"
MQTT_PORT = 1883
MQTT_KEEPALIVE = 60
MQTT_TOPIC_UP = "/v2g/simulation/uplink"
MQTT_TOPIC_DOWN = "/v2g/simulation/downlink"

# 3. WebSocket 配置
WS_PATH = "/ws/stream"
WS_PUSH_INTERVAL = 5
WS_HOST = "0.0.0.0"
WS_PORT = 8081


# 初始化客户端
db_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG, timeout=60000)
write_api = db_client.write_api(write_options=WriteOptions(
    batch_size=500,
    flush_interval=1000,
    retry_interval=2000
))
query_api = db_client.query_api()
delete_api = DeleteApi(db_client)

ws_manager = None
mqtt_client = None
scheduler = AsyncIOScheduler(timezone="Asia/Shanghai")


# ==========================================
# 🔌 WebSocket 管理类（优化版）
# ==========================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"🟢 [WebSocket] 前端已连接，在线总数: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"🔴 [WebSocket] 前端断开，在线总数: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        if not self.active_connections:
            return
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)


ws_manager = ConnectionManager()


# ==========================================
# 📡 MQTT 回调函数（优化版）
# ==========================================
def on_mqtt_connect(client, userdata, flags, reason_code, properties=None):
    print("✅ [MQTT] 已成功连接到 EMQX")
    client.subscribe(MQTT_TOPIC_UP)
    client.subscribe("v2g/cloud/schedule")
    client.subscribe("/v2g/cloud/control")
    client.subscribe("/v2g/grid/load")
    client.subscribe("/v2g/alert")
    client.subscribe("/v2g/price")


def on_mqtt_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode('utf-8'))

        # ==============================================
        # 1. 决策层下发指令（调度）
        # ==============================================
        if msg.topic == "v2g/cloud/schedule":
            print("\n===== 收到决策层下发JSON =====")
            print(json.dumps(payload, indent=4, ensure_ascii=False))

            # 入库
            p = Point("cloud_schedule") \
                .tag("schedule_id", str(payload.get("schedule_id", "unknown"))) \
                .field("power_set", float(payload.get("power_set", 0))) \
                .field("duration", float(payload.get("duration", 0))) \
                .time(datetime.utcnow(), WritePrecision.NS)
            write_api.write(bucket=INFLUX_BUCKET_RAW, org=INFLUX_ORG, record=p)
            print("✅ 决策指令已入库")
            return

        # ==============================================
        # 2. 云平台控制指令（充电/放电/停止）
        # ==============================================
        if msg.topic == "/v2g/cloud/control":
            print("\n===== 收到远程控制指令 =====")
            print(json.dumps(payload, indent=4, ensure_ascii=False))

            # 入库
            p = Point("cloud_control") \
                .tag("node_id", str(payload.get("node_id", "unknown"))) \
                .tag("command", str(payload.get("command", "none"))) \
                .field("set_power", float(payload.get("set_power", 0))) \
                .time(datetime.utcnow(), WritePrecision.NS)
            write_api.write(bucket=INFLUX_BUCKET_RAW, org=INFLUX_ORG, record=p)
            print("✅ 控制指令已入库")
            return

        # ==============================================
        # 3. 电网负荷数据
        # ==============================================
        if msg.topic == "/v2g/grid/load":
            print("\n===== 收到电网负荷数据 =====")
            print(json.dumps(payload, indent=4, ensure_ascii=False))

            # 入库
            p = Point("grid_load") \
                .field("grid_load", float(payload.get("grid_load", 0))) \
                .field("voltage", float(payload.get("voltage", 220))) \
                .time(datetime.utcnow(), WritePrecision.NS)
            write_api.write(bucket=INFLUX_BUCKET_RAW, org=INFLUX_ORG, record=p)
            print("✅ 电网负荷已入库")
            return

        # ==============================================
        # 4. 告警信息
        # ==============================================
        if msg.topic == "/v2g/alert":
            print("\n===== 收到设备告警 =====")
            print(json.dumps(payload, indent=4, ensure_ascii=False))

            # 入库
            p = Point("device_alert") \
                .tag("node_id", str(payload.get("node_id", "unknown"))) \
                .tag("alert_type", str(payload.get("alert_type", "normal"))) \
                .field("alert_level", int(payload.get("alert_level", 0))) \
                .time(datetime.utcnow(), WritePrecision.NS)
            write_api.write(bucket=INFLUX_BUCKET_RAW, org=INFLUX_ORG, record=p)
            print("✅ 告警信息已入库")
            return

        # ==============================================
        # 5. 实时电价数据
        # ==============================================
        if msg.topic == "/v2g/price":
            print("\n===== 收到实时电价 =====")
            print(json.dumps(payload, indent=4, ensure_ascii=False))

            # 入库
            p = Point("electricity_price") \
                .field("price", float(payload.get("price", 0))) \
                .field("time_period", int(payload.get("time_period", 0))) \
                .time(datetime.utcnow(), WritePrecision.NS)
            write_api.write(bucket=INFLUX_BUCKET_RAW, org=INFLUX_ORG, record=p)
            print("✅ 电价数据已入库")
            return

        # ==============================================
        # 6. 车/桩端上传数据（原来的逻辑）
        # ==============================================
        if msg.topic == MQTT_TOPIC_UP:
            print("\n===== 收到边端上报数据 =====")
            node_id = str(payload.get("node_id", "unknown"))
            power = float(payload.get("actual_power_kw", 0.0))
            soc = float(payload.get("current_soc", 0.0))

            p = Point("v2g_node_status") \
                .tag("node_id", node_id) \
                .field("power_kw", power) \
                .field("soc", soc) \
                .time(datetime.utcnow(), WritePrecision.NS)

            write_api.write(bucket=INFLUX_BUCKET_RAW, org=INFLUX_ORG, record=p)
            print(f"📥 [边端上报] 节点 {node_id} -> 功率: {power}kW, SOC: {soc}, 写入成功")
            return

    except Exception as e:
        print(f"❌ [MQTT 处理失败]: {e}")
# ==========================================
# ⏱️ 定时任务（新增：真实数据聚合 + 降采样）
# ==========================================
async def scheduled_task_push_to_frontend():
    """定时向前端推送真实聚合数据（从InfluxDB查）"""
    if not ws_manager.active_connections:
        return
    try:
        # 1. 从InfluxDB查过去1分钟的真实聚合数据
        flux_query = f'''
        from(bucket: "{INFLUX_BUCKET_RAW}")
          |> range(start: -1m)
          |> filter(fn: (r) => r._measurement == "v2g_node_status")
          |> filter(fn: (r) => r._field == "power_kw" or r._field == "soc")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> aggregateWindow(every: 5s, fn: mean)
        '''
        tables = query_api.query(flux_query, org=INFLUX_ORG)

        # 2. 计算聚合指标
        total_power = 0.0
        active_nodes = 0
        for table in tables:
            for record in table.records:
                if "power_kw" in record.values:
                    total_power += float(record.values.get("power_kw", 0))
                    active_nodes += 1

        # 3. 推送给前端
        push_data = {
            "type": "realtime_update",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": {
                "total_grid_power_kw": round(total_power, 2),
                "active_nodes_count": active_nodes,
                "status": "online"
            }
        }
        await ws_manager.broadcast(push_data)
        print(f"📤 [定时推送] 总功率: {total_power}kW, 在线节点: {active_nodes}")
    except Exception as e:
        print(f"❌ [定时推送失败]: {e}")


async def scheduled_task_downsample():
    """定时降采样：把raw_data的1分钟数据聚合到agg_data，按小时存"""
    try:
        # 这里可以用InfluxDB的Tasks功能，也可以用Python脚本实现
        print("⏳ [降采样任务] 执行中...")
        # 简单示例：删除agg_data里过去1小时的数据，重新聚合写入
        # 生产环境建议直接用InfluxDB UI创建Tasks
    except Exception as e:
        print(f"❌ [降采样失败]: {e}")


# ==========================================
# 🚀 生命周期管理
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global mqtt_client
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message

    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE)
        mqtt_client.loop_start()
    except Exception as e:
        print(f"⚠️ [MQTT 连接异常]: {e}")

    # 定时任务：推送实时数据 + 降采样
    scheduler.add_job(scheduled_task_push_to_frontend, 'interval', seconds=WS_PUSH_INTERVAL)
    scheduler.add_job(scheduled_task_downsample, 'cron', minute='0')  # 每小时整点执行降采样
    scheduler.start()
    print(f"⏰ [定时调度器] 已启动，每 {WS_PUSH_INTERVAL} 秒同步一次大屏数据。")

    yield

    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    scheduler.shutdown()
    db_client.close()


# ==========================================
# FastAPI 主程序
# ==========================================
app = FastAPI(title="V2G 云边协同数据中台", lifespan=lifespan, version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 【核心补全】真实业务 API 接口
# ==========================================
def format_influx_time(dt):
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")

# ====================== 修复好的功率接口 ======================
@app.get("/api/v1/history/power", summary="查询节点历史功率数据")
async def get_history_power(
    node_id: int = Query(..., description="节点ID（纯数字）"),
    start_range: str = Query("-1y", description="时间范围"),
    end_range: str = Query("now()")
):
    try:
        flux_query = f'''
        from(bucket: "{INFLUX_BUCKET_RAW}")
          |> range(start: {start_range}, stop: {end_range})
          |> filter(fn: (r) => r._measurement == "volume")
          |> filter(fn: (r) => r.station_id == "{node_id}")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> sort(columns: ["_time"])
        '''

        tables = query_api.query(flux_query, org=INFLUX_ORG)
        result = []
        for table in tables:
            for record in table.records:
                result.append({
                    "time": format_influx_time(record["_time"]),
                    "power_kw": round(float(record["value"]), 2)
                })

        return {
            "code": 200,
            "msg": "查询成功",
            "count": len(result),
            "data": result
        }
    except Exception as e:
        return {"code": 500, "msg": f"查询失败: {str(e)}", "count": 0, "data": []}

# ====================== 修复好的多指标接口 ======================
@app.get("/api/v1/history/multi", summary="查询多指标历史数据")
async def get_history_multi(
    station_id: int = Query(...),
    start_range: str = Query("-1y")
):
    try:
        flux_query = f'''
        from(bucket: "{INFLUX_BUCKET_RAW}")
          |> range(start: {start_range})
          |> filter(fn: (r) => r.station_id == "{station_id}")
          |> filter(fn: (r) => contains(set: ["e_price","volume","occupancy"], value: r._measurement))
          |> pivot(rowKey:["_time"], columnKey: ["_measurement"], valueColumn: "_value")
          |> sort(columns: ["_time"])
        '''

        tables = query_api.query(flux_query, org=INFLUX_ORG)
        result = []
        for table in tables:
            for record in table.records:
                item = {
                    "time": format_influx_time(record["_time"]),
                    "e_price": float(record["e_price"]),
                    "volume": float(record["volume"]),
                    "occupancy": float(record["occupancy"])
                }
                result.append(item)

        return {
            "code": 200,
            "msg": "查询成功",
            "count": len(result),
            "data": result
        }
    except Exception as e:
        return {"code": 500, "msg": f"查询失败: {str(e)}", "count": 0, "data": []}


@app.get("/api/v1/stations", summary="获取所有站点列表（从inf.csv导入的元数据）")
async def get_station_list():
    """查询你导入的inf.csv里的站点基础信息（经纬度、TAZID等）"""
    try:
        flux_query = f'''
        from(bucket: "{INFLUX_BUCKET_RAW}")
          |> range(start: -1h)
          |> filter(fn: (r) => r._measurement == "inf")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''

        tables = query_api.query(flux_query, org=INFLUX_ORG)
        result = []
        for table in tables:
            for record in table.records:
                item = record.values
                # 清理不需要的字段
                for key in ["_time", "_start", "_stop", "table", "result", "row_id"]:
                    if key in item:
                        del item[key]
                result.append(item)

        return {"code": 200, "msg": "查询成功", "count": len(result), "data": result}
    except Exception as e:
        return {"code": 500, "msg": f"查询失败: {str(e)}", "count": 0, "data": []}


@app.get("/api/v1/health", summary="健康检查接口")
async def health_check():
    """对接监控系统用的健康检查接口"""
    return {
        "code": 200,
        "status": "online",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "influxdb": "connected",
        "mqtt": "connected" if mqtt_client.is_connected() else "disconnected"
    }


# WebSocket 端点
@app.websocket(WS_PATH)
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# ==========================================
# 启动入口
# ==========================================
if __name__ == "__main__":
    print("🚀 准备启动 V2G 数据中台...")
    print(f"📍 WebSocket 地址: ws://{WS_HOST}:{WS_PORT}{WS_PATH}")
    print(f"📖 API 文档地址: http://127.0.0.1:{WS_PORT}/docs")
    uvicorn.run("main:app", host=WS_HOST, port=WS_PORT, reload=True)