# GPU 训练平台使用指南

> 本文档面向算法与研发同学，介绍如何在 GPU 训练平台上发起训练任务及查看任务详情。

**平台入口**：[https://tianqin.tmeoa.com/train/dashboard](https://tianqin.tmeoa.com/train/dashboard)

---

## 目录

- [一、发起任务](#一发起任务)
  - [1.1 基本信息](#11-基本信息)
  - [1.2 资源配置](#12-资源配置)
  - [1.3 存储与环境](#13-存储与环境)
  - [1.4 启动命令](#14-启动命令)
  - [1.5 环境变量](#15-环境变量)
  - [1.6 任务时长预估](#16-任务时长预估)
- [二、查看任务详情](#二查看任务详情)
  - [2.1 任务状态](#21-任务状态)
  - [2.2 资源信息](#22-资源信息)
  - [2.3 实时日志](#23-实时日志)
  - [2.4 TensorBoard](#24-tensorboard)
  - [2.5 任务管理](#25-任务管理)

---

## 一、发起任务

进入「发起任务」页面，按以下步骤配置并提交训练任务。

### 1.0 镜像规范

> ⚠️ **重要**：所有训练镜像必须内置 `/app/entrypoint.sh` 作为统一分布式训练入口脚本。

**镜像要求：**

- 脚本位置：`/app/entrypoint.sh`（必须）
- 脚本权限：可执行（`chmod +x /app/entrypoint.sh`）
- 系统依赖：必须安装 `nfs-common` 和 `iproute2`（用于挂载 CFS 共享文件系统）
- Dockerfile 示例：

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 系统依赖（NFS 客户端，用于挂载 CFS 共享文件系统）
RUN apt-get update && apt-get install -y --no-install-recommends \
        nfs-common \
        iproute2 \
    && rm -rf /var/lib/apt/lists/*

# 复制入口脚本
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# 设置入口
ENTRYPOINT ["/app/entrypoint.sh"]
```

**entrypoint.sh 核心职责：**

| 职责 | 说明 |
|------|------|
| CFS 文件系统挂载 | 通过 `CFS_MOUNTS` 环境变量自动挂载存储盘 |
| 多机节点发现与选举 | 通过共享文件自动选举 master 节点 |
| NCCL 网络环境配置 | 自动检测网卡、设置 NCCL 参数 |
| 导出分布式环境变量 | `NODE_RANK`、`MASTER_IP`、`NUM_GPUS` 等供用户命令使用 |
| 执行用户自定义命令 | 通过 `USER_COMMAND` 环境变量执行用户的训练命令 |

### 1.1 基本信息

| 字段 | 说明 | 示例 |
|------|------|------|
| **任务名称** | 必填，用于标识任务 | `rank_cross_domain_v3` |
| **任务详情** | 描述任务目标、数据集、模型架构等 | 可填写实验目的、版本说明 |
| **运行镜像** | 下拉选择预置镜像 | 暂不支持自定义镜像，如需新增请联系管理员 |

**注意事项：**
- 任务名称建议采用有意义的命名规范，便于后续检索
- 任务详情可填写 1000 字符以内的备注信息
- 如需要特定镜像版本，需提前联系管理员配置

### 1.2 资源配置

按地域和机型查看可用节点，输入需要申请的节点数量。

| 字段 | 说明 |
|------|------|
| **集群** | 默认集群 |
| **地域** | 上海-临港 / 上海-松江 |
| **GPU 型号** | H20 |
| **可用节点 / 总数** | 实时显示当前可用资源 |
| **选择** | 勾选需要使用的资源 |
| **申请数量** | 输入申请的节点数 |

**资源配置示例：**

| 集群 | 地域 | GPU 型号 | 可用节点 / 总数 | 选择 |
|------|------|----------|-----------------|------|
| 默认集群 | 上海-临港 | H20 | 6 / 6 | ⭕ |
| 默认集群 | 上海-松江 | H20 | 0 / 2 | ⭕ |

> 💡 **提示**：选择资源后，在「申请数量」列输入所需节点数。

### 1.3 存储与环境

#### 挂载 CFS

- **cfs-hooke**（默认）：yibo 的盘 · 上海
- 默认盘必须挂载，其他盘可按需开启

**路径说明：**
- CFS 挂载后，可在容器内通过 `/cfs-hooke` 访问数据
- 建议将训练数据、模型输出、日志统一存放在 CFS 目录下，便于持久化和团队协作

### 1.4 启动命令

启动命令支持两种方式：

#### 方式一：脚本形式（推荐）

```bash
# 推荐 1：指定绝对路径
# 脚本头部指定 shell 解释器 #!/bin/bash 或者 #!/usr/bin/env
/cfs/start.sh

# 脚本形式 指定 bash
/bin/bash /data/start.sh
```

#### 方式二：直接写 shell 命令

```bash
torchrun xxx
```

#### 完整启动命令示例

**多机分布式训练（推荐使用环境变量）：**

```bash
# 方式一：脚本形式（推荐，便于管理和参数调整）
/bin/bash /cfs-hooke/scripts/train.sh

# 方式二：直接使用 torchrun（环境变量由 entrypoint.sh 自动导出）
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --data_dir=/cfs-hooke/data/train \
    --output_dir=/cfs-hooke/outputs/$TASK_ID \
    --tensorboard_dir=$TENSORBOARD_LOGGING_DIR
```

> 💡 **提示**：`$NUM_GPUS`、`$NUM_NODES`、`$NODE_RANK`、`$MASTER_ADDR`、`$MASTER_PORT` 等变量由 `/app/entrypoint.sh` 自动导出，在启动命令中可直接使用，无需手动计算。

#### 环境变量说明

**系统环境变量**（不可覆盖，由平台自动生成）：

| 变量名 | 说明 |
|--------|------|
| `CFS_MOUNTS` | CFS 挂载配置（JSON 格式），由 entrypoint.sh 解析并执行挂载 |
| `USER_COMMAND` | 用户的启动命令，由 entrypoint.sh 执行 |
| `NUM_NODES` | 节点数量，用于多机模式下等待所有节点注册 |
| `NUM_GPUS` | 每个节点 GPU 卡数，多机模式默认 `8`，本地模式自动检测 |
| `TASK_ID` | 任务字符串 ID，如 `prod_12`，用于隔离不同任务的选举文件 |

**业务环境变量**（由 `/app/entrypoint.sh` 自动导出，可在用户命令中直接使用）：

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `NODE_RANK` | 当前节点的 rank（0 = master），由共享文件选举得来 | — |
| `MASTER_ADDR` / `MASTER_IP` | 分布式训练 master 地址，由选举得来 | — |
| `MASTER_PORT` | 分布式训练 master 端口 | `29500` |
| `NUM_GPUS` | 每节点 GPU 数量 | 多机 `8`，本地自动检测 |
| `NUM_NODES` | 节点总数 | 自动检测 |
| `CONNECTION_TIMEOUT` | 分布式训练连接超时 | `1800s` |
| `TENSORBOARD_LOGGING_DIR` | TensorBoard 日志目录，由挂载盘目录自动生成 | — |

**NCCL 相关变量**（由 entrypoint.sh 自动配置）：

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `NCCL_SOCKET_IFNAME` | 通信网卡名，自动检测 | 自动 |
| `NCCL_DEBUG` | NCCL 日志级别 | `WARN` |
| `NCCL_IB_DISABLE` | 禁用 InfiniBand | `0` |
| `NCCL_NET_GDR_LEVEL` | GPUDirect RDMA 级别 | `5` |

**entrypoint.sh 可选环境变量**（高级配置，一般无需手动设置）：

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `MODE` | 运行模式：`multi`（多机）或 `local`（单机） | `multi` |
| `DISCOVERY_FILE` | 共享文件路径 | 自动拼接：`<CFS挂载点>/distributed-discovery/<TASK_ID>/discovery` |
| `DISCOVERY_PREFIX` | 选举目录前缀 | `distributed-discovery` |
| `NODE_WAIT_TIMEOUT` | 等待所有节点注册的超时秒数 | `1800s` |
| `NET_IFACE` | 指定获取本机 IP 的网卡名 | 自动检测 |

### 1.5 自定义环境变量

- 最多支持 **10 个**自定义环境变量
- 变量名仅支持**大写字母、数字和下划线**
- 点击「+ 添加环境变量」按钮添加

**常见用途：**
- 设置训练超参数（如 `BATCH_SIZE`、`LEARNING_RATE`）
- 配置实验标识（如 `EXPERIMENT_ID`）
- 传入敏感信息（如 `API_KEY`）

### 1.6 任务时长预估

| 字段 | 说明 |
|------|------|
| **预计运行时长** | 输入预估的任务运行时间 |
| **单位** | 时 / 分 / 秒 |

> 💡 **用途**：用于调度优先级评估和超时阻断判断

**填写建议：**
- 根据历史相似任务的实际运行时间预估
- 建议留有一定余量，避免因超时导致任务被强制终止
- 如果实际运行超过预估时长，任务可能被系统标记或终止

### 提交任务

配置完成后，点击页面右下角 **「提交审批」** 按钮发起任务。

---

## 二、查看任务详情

任务提交后，可在任务列表中查看任务状态，点击任务进入详情页。

### 2.1 任务状态

任务详情页顶部显示当前任务状态：

| 状态 | 说明 |
|------|------|
| **排队中** | 等待资源分配 |
| **运行中** | 任务正在执行 |
| **已完成** | 任务正常结束 |
| **失败** | 任务执行出错 |
| **已终止** | 被手动停止或超时终止 |

### 2.2 资源信息

显示任务申请的资源详情：

```
2节点 × NVIDIA-H20-8-384C-2304Gi
```

包含信息：
- 节点数量
- GPU 型号（H20）
- GPU 卡数（8 卡）
- 内存配置（384C-2304Gi）
- 所属集群（默认集群）

### 2.3 实时日志

点击 **「查看日志」** 可查看任务运行的实时日志：

- 支持实时刷新
- 可查看 stdout / stderr 输出
- 日志保留时长遵循平台策略

**常见用途：**
- 监控训练进度
- 排查报错信息
- 查看 loss、acc 等训练指标输出

### 2.4 TensorBoard

点击 **「TensorBoard」** 可打开可视化监控面板：

- 自动关联 `TENSORBOARD_LOGGING_DIR` 指定的日志目录
- 支持查看 loss 曲线、learning rate 变化、模型 graph 等
- 多节点训练自动聚合数据

**TensorBoard 日志路径示例：**
```
/cfs-hooke/tensorboard/lyra_train_prod_12
```

### 2.5 任务管理

根据任务状态，可执行以下操作：

| 操作 | 适用状态 | 说明 |
|------|----------|------|
| **终止任务** | 排队中 / 运行中 | 立即停止任务，释放资源 |
| **重新提交** | 失败 / 已终止 | 基于当前配置快速创建新任务 |
| **查看配置** | 所有状态 | 查看任务发起时的完整配置 |

---

## 三、最佳实践

### 3.1 任务命名规范

建议采用以下格式：

```
{模型名}_{数据集}_{实验目的}_{版本}

# 示例
gpt2_corpus_pretrain_v3
rank_cross_domain_ablation_test
```

### 3.2 目录组织建议

```
/cfs-hooke/
├── data/               # 数据集
│   └── corpus/
├── models/             # 模型代码
│   └── gpt2/
├── outputs/            # 训练输出
│   └── {task_id}/
├── checkpoints/        # 模型检查点
│   └── {task_id}/
└── logs/               # 日志文件
    └── {task_id}/
```

### 3.3 启动脚本模板

```bash
#!/bin/bash
# start.sh

# 打印环境信息（便于调试）
echo "=== Environment Info ==="
echo "TASK_ID: $TASK_ID"
echo "NUM_NODES: $NUM_NODES"
echo "NUM_GPUS: $NUM_GPUS"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

# 激活环境（如需要）
source /opt/conda/etc/profile.d/conda.sh
conda activate myenv

# 启动训练
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --config configs/train.yaml \
    --output_dir=/cfs-hooke/outputs/$TASK_ID \
    --tensorboard_dir=$TENSORBOARD_LOGGING_DIR
```

### 3.4 常见问题

**Q1: 任务一直显示「排队中」怎么办？**
- 检查资源看板确认当前是否有可用节点
- 确认预估运行时长是否合理（过长的任务可能调度优先级较低）
- 联系管理员确认是否有资源配额限制

**Q2: 任务启动后很快就失败了？**
- 查看「查看日志」中的错误输出
- 常见原因：启动命令路径错误、依赖缺失、数据路径不存在
- 确认 CFS 挂载是否正确

**Q3: 如何保存训练好的模型？**
- 建议将 checkpoint 保存到 `/cfs-hooke/checkpoints/$TASK_ID/`
- CFS 数据持久化，任务结束后仍可访问

---

*文档版本：v1.0*  
*最后更新：2026-04-29*
