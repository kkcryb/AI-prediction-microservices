# AI-prediction-microservices

📁 项目目录结构 (AI 算法与预测)
本项目基于 FastAPI 与 PyTorch 构建，采用模型训练与微服务推理彻底解耦的设计。完整的项目目录结构如下：
```text
AI_Prediction_Microservice/
│
├── app/                                 # FastAPI 微服务核心代码目录
│   ├── __init__.py
│   ├── main.py                          # FastAPI 应用入口文件，挂载路由和启动事件
│   ├── core/                            # 核心配置模块
│   │   ├── __init__.py
│   │   ├── config.py                    # 环境变量加载 (如 InfluxDB 地址, 模型路径等)
│   │   └── exceptions.py                # 全局异常处理类
│   │
│   ├── api/                             # Web API 路由层 (对接运筹优化组)
│   │   ├── __init__.py
│   │   ├── dependencies.py              # 依赖注入 (如获取数据库客户端、加载模型实例)
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── predict.py               # 核心接口: 接收24h数据，返回未来24h预测与弹性系数矩阵
│   │       └── health.py                # 健康检查接口 (Docker/K8s探针使用)
│   │
│   ├── services/                        # 业务逻辑层 (连接API与底层模型)
│   │   ├── __init__.py
│   │   ├── inference_service.py         # 封装 GCN-LSTM 前向推理逻辑 (1小时级调用)
│   │   ├── elasticity_service.py        # 实现微小扰动法，计算电价弹性系数矩阵
│   │   └── data_client.py               # HTTP/InfluxDB 客户端: 向数据中台请求标准化的历史时序数据
│   │
│   ├── models/                          # 深度学习模型定义层 (PyTorch)
│   │   ├── __init__.py
│   │   ├── gcn_lstm.py                  # 定义融合了图卷积与LSTM网络结构的核心类
│   │   └── graph_builder.py             # 构建空间拓扑矩阵 (Station-to-Grid 邻接矩阵 A) 
│   │
│   └── schemas/                         # Pydantic 数据验证模型 (输入输出格式定义)
│       ├── __init__.py
│       ├── request_models.py            # 定义API请求载荷格式
│       └── response_models.py           # 定义API返回载荷格式 (含预测向量与弹性系数JSON)
│
├── training/                            # 线下模型训练流水线 (不参与线上微服务运行)
│   ├── __init__.py
│   ├── dataset.py                       # PyTorch Dataset 构建，处理从中台拉取的批次数据
│   ├── train_gcn_lstm.py                # 模型训练主脚本 (包含前向传播、损失计算、反向传播)
│   ├── evaluate.py                      # 模型评估脚本 (计算 WMAPE, RMSE, MAE 等指标)
│   └── hyperparam_tuning.py             # 超参数搜索脚本
│
├── checkpoints/                         # 模型权重存储目录
│   ├── best_gcn_lstm.pth                # 训练产出的最优模型权重文件，供 app/ 加载
│   └── scaler.pkl                       # 数据归一化/反归一化相关的参数文件
│
├── tests/                               # 单元测试与集成测试
│   ├── test_api.py                      # 测试 FastAPI 接口响应
│   ├── test_models.py                   # 测试模型输入输出张量维度是否正确
│   └── test_elasticity.py               # 测试扰动法计算逻辑的准确性
│
├── docs/                                # 文档目录
│   └── api_reference.md                 # 供“运筹优化组”调用的 API 接口说明文档
│
├── Dockerfile                           # 容器化构建文件，打包环境与代码
├── requirements.txt                     # Python 依赖清单 (PyTorch, FastAPI, Pandas, etc.)
└── .env.example                         # 环境变量示例文件
