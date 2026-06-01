#!/usr/bin/env bash
# ============================================================
# 通用分布式训练入口 — /app/entrypoint.sh
#
# 负责：
#   1. CFS 文件系统挂载（通过 CFS_MOUNTS 环境变量）
#   2. 多机节点发现与选举（通过共享文件）
#   3. NCCL 网络环境自动配置
#   4. 导出分布式环境变量供用户命令使用
#   5. 执行用户自定义命令（通过 USER_COMMAND 环境变量）
#
# 支持两种模式：
#
# 【模式 1】多机模式（默认）— 通过共享文件自动发现 master
#   docker run --rm --privileged --net=host \
#     --gpus all --shm-size=32g \
#     -e CFS_MOUNTS='...' \
#     -e TASK_ID=my-task \
#     -e USER_COMMAND='bash /workspace/scripts/train.sh' \
#     my-image:latest
#
# 【模式 2】本地单机模式 — 无需共享文件，自动检测本机 GPU
#   docker run --rm --privileged --net=host \
#     --gpus all --shm-size=32g \
#     -e MODE=local \
#     -e USER_COMMAND='bash /workspace/scripts/train.sh' \
#     my-image:latest
#
# 环境变量：
#   MODE              - 运行模式: multi（默认）或 local
#   USER_COMMAND      - 用户自定义启动命令（必须）
#   CFS_MOUNTS        - CFS 挂载配置（JSON 格式）
#   TASK_ID           - 任务 ID，用于隔离不同任务的选举文件
#   DISCOVERY_FILE    - 共享文件路径（默认自动拼接: <CFS挂载点>/distributed-discovery/<TASK_ID>/discovery）
#   DISCOVERY_PREFIX  - 选举目录前缀（默认 distributed-discovery）
#   NUM_GPUS            - 每台机器使用的 GPU 数量（默认 8，local 模式默认自动检测）
#   NUM_NODES           - 节点总数（默认自动检测，local 模式固定为 1）
#   MASTER_PORT         - 通信端口（默认 29500）
#   CONNECTION_TIMEOUT  - 连接超时秒数（默认 1800）
#   NODE_WAIT_TIMEOUT   - 等待所有节点注册的超时秒数（默认 1800）
#   NET_IFACE           - 用于获取本机 IP 的网卡名（默认自动检测）
#
# 导出给 USER_COMMAND 的环境变量：
#   NODE_RANK           - 当前节点的 rank（0 = master）
#   MASTER_IP           - master 节点的 IP 地址
#   NUM_GPUS            - 每节点 GPU 数量
#   NUM_NODES           - 节点总数
#   MASTER_PORT         - 通信端口
#   CONNECTION_TIMEOUT  - 连接超时
#   NCCL_SOCKET_IFNAME / NCCL_DEBUG 等 NCCL 相关变量
# ============================================================

set -euo pipefail

# ---------- 颜色输出 ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }

# ---------- 默认值 ----------
MODE="${MODE:-multi}"
USER_COMMAND="${USER_COMMAND:-}"
TASK_ID="${TASK_ID:-}"
DISCOVERY_FILE="${DISCOVERY_FILE:-}"
DISCOVERY_PREFIX="${DISCOVERY_PREFIX:-distributed-discovery}"
NUM_GPUS="${NUM_GPUS:-}"
NUM_NODES="${NUM_NODES:-}"
MASTER_PORT="${MASTER_PORT:-29500}"
CONNECTION_TIMEOUT="${CONNECTION_TIMEOUT:-1800}"
NODE_WAIT_TIMEOUT="${NODE_WAIT_TIMEOUT:-1800}"
NET_IFACE="${NET_IFACE:-}"

# 内部变量
NODE_RANK=""
MASTER_IP=""
MY_IP=""
USER_CMD_EXIT_CODE=0

# ---------- 检查 USER_COMMAND ----------
if [[ -z "$USER_COMMAND" ]]; then
    fail "必须设置 USER_COMMAND 环境变量"
    fail "示例: -e USER_COMMAND='torchrun --nproc_per_node=8 train.py'"
    exit 1
fi

# ---------- Debug 模式：跳过所有前置逻辑，直接执行 USER_COMMAND ----------
if [[ "$MODE" == "debug" ]]; then
    echo ""
    echo -e "${BOLD}=========================================="
    echo "       Debug 模式"
    echo -e "==========================================${NC}"
    echo ""
    info "跳过 CFS 挂载、节点发现等前置步骤"
    info "USER_COMMAND: $USER_COMMAND"
    echo ""
    info "💡 进入容器后，执行以下命令获取完整分布式环境变量："
    info "   source /app/setup_env.sh"
    echo ""
    exec bash -c "$USER_COMMAND"
fi

# ---------- 获取本机 IP ----------
get_my_ip() {
    local ip=""

    # 优先使用指定网卡
    if [[ -n "$NET_IFACE" ]]; then
        ip=$(ip -4 addr show "$NET_IFACE" 2>/dev/null | grep -oP 'inet \K[\d.]+' | head -1 || true)
        if [[ -n "$ip" ]]; then
            echo "$ip"
            return
        fi
        warn "指定网卡 $NET_IFACE 未找到 IP，尝试自动检测..."
    fi

    # 自动检测：取第一个非 loopback 的 IPv4 地址
    ip=$(hostname -I 2>/dev/null | awk '{print $1}' || true)
    if [[ -n "$ip" ]]; then
        echo "$ip"
        return
    fi

    # 兜底
    ip=$(ip -4 route get 8.8.8.8 2>/dev/null | grep -oP 'src \K[\d.]+' | head -1 || true)
    echo "${ip:-127.0.0.1}"
}

# ---------- 自动检测 GPU 数量 ----------
detect_gpu_count() {
    local count
    count=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
    if [[ "$count" -gt 0 ]]; then
        echo "$count"
    else
        echo "1"
    fi
}

# ---------- 通过共享文件发现 master（多机模式）----------
discover_master() {
    echo ""
    echo -e "${BOLD}=========================================="
    echo "       共享文件自动发现 Master"
    echo -e "==========================================${NC}"
    echo ""

    MY_IP=$(get_my_ip)
    info "本机 IP: $MY_IP"
    info "共享文件: $DISCOVERY_FILE"

    local lock_file="${DISCOVERY_FILE}.lock"
    local discovery_dir
    discovery_dir=$(dirname "$DISCOVERY_FILE")

    # 确保目录存在
    if [[ ! -d "$discovery_dir" ]]; then
        fail "共享目录不存在: $discovery_dir"
        fail "请确保 NFS 挂载正确，或手动创建该目录"
        exit 1
    fi

    # 使用 flock 文件锁进行原子操作，注册本机 IP 并获取 node_rank
    info "正在注册节点..."

    local result
    result=$(
        flock -w 30 "$lock_file" bash -c "
            if [[ ! -f '$DISCOVERY_FILE' ]] || [[ ! -s '$DISCOVERY_FILE' ]]; then
                # 文件不存在或为空 → 我是第一个 → 我是 master（rank=0）
                echo '$MY_IP' > '$DISCOVERY_FILE'
                echo 'RANK=0:MASTER=$MY_IP'
            else
                # 检查自己是否已注册
                existing_rank=\$(grep -nxF '$MY_IP' '$DISCOVERY_FILE' | head -1 | cut -d: -f1)
                if [[ -n \"\$existing_rank\" ]]; then
                    # 已注册（可能是重启），rank = 行号 - 1
                    my_rank=\$((existing_rank - 1))
                    master_ip=\$(head -1 '$DISCOVERY_FILE' | tr -d '[:space:]')
                    echo \"RANK=\$my_rank:MASTER=\$master_ip\"
                else
                    # 追加注册，rank = 当前行数
                    my_rank=\$(wc -l < '$DISCOVERY_FILE')
                    echo '$MY_IP' >> '$DISCOVERY_FILE'
                    master_ip=\$(head -1 '$DISCOVERY_FILE' | tr -d '[:space:]')
                    echo \"RANK=\$my_rank:MASTER=\$master_ip\"
                fi
            fi
        "
    )

    # 解析结果: RANK=<n>:MASTER=<ip>
    if [[ "$result" =~ ^RANK=([0-9]+):MASTER=(.+)$ ]]; then
        NODE_RANK="${BASH_REMATCH[1]}"
        MASTER_IP="${BASH_REMATCH[2]}"
        if [[ "$NODE_RANK" == "0" ]]; then
            ok "✅ 本机是 MASTER（第一个写入共享文件）→ node_rank=0"
            ok "   Master IP: $MASTER_IP"
        else
            ok "✅ 本机是 WORKER → node_rank=$NODE_RANK"
            ok "   Master IP: $MASTER_IP（从共享文件读取）"
        fi
    else
        fail "共享文件发现逻辑异常，结果: $result"
        exit 1
    fi

    echo ""
    info "共享文件内容:"
    cat "$DISCOVERY_FILE" 2>/dev/null | while read -r line; do
        echo "    $line"
    done
    echo ""

    # ---- 等待所有节点注册 ----
    if [[ -n "$NUM_NODES" ]] && [[ "$NUM_NODES" -gt 0 ]]; then
        # 用户指定了 NUM_NODES，等待直到共享文件中有足够的节点
        info "等待 ${NUM_NODES} 个节点全部注册（超时: ${NODE_WAIT_TIMEOUT}s）..."
        local waited=0
        while true; do
            local registered
            registered=$(wc -l < "$DISCOVERY_FILE" 2>/dev/null || echo "0")
            if [[ "$registered" -ge "$NUM_NODES" ]]; then
                ok "✅ 所有 ${NUM_NODES} 个节点已注册"
                break
            fi
            if [[ "$waited" -ge "$NODE_WAIT_TIMEOUT" ]]; then
                warn "等待超时！当前已注册 ${registered}/${NUM_NODES} 个节点"
                warn "将以 ${registered} 个节点继续启动"
                NUM_NODES="$registered"
                break
            fi
            if (( waited % 10 == 0 )); then
                info "已注册 ${registered}/${NUM_NODES} 个节点，等待中...（${waited}/${NODE_WAIT_TIMEOUT}s）"
            fi
            sleep 2
            waited=$((waited + 2))
        done
    else
        # 用户未指定 NUM_NODES，等待一段时间让所有节点注册，然后以实际注册数启动
        info "未指定 NUM_NODES，等待其他节点注册（${NODE_WAIT_TIMEOUT}s 内无新节点则启动）..."
        local last_count=0
        local stable_time=0
        local stable_threshold=30  # 连续 30 秒无新节点注册则认为所有节点已就绪
        local waited=0
        while true; do
            local registered
            registered=$(wc -l < "$DISCOVERY_FILE" 2>/dev/null || echo "0")
            if [[ "$registered" -gt "$last_count" ]]; then
                last_count="$registered"
                stable_time=0
                info "新节点注册！当前共 ${registered} 个节点"
            else
                stable_time=$((stable_time + 2))
            fi
            # 至少需要 1 个节点，且稳定一段时间
            if [[ "$registered" -ge 1 ]] && [[ "$stable_time" -ge "$stable_threshold" ]]; then
                NUM_NODES="$registered"
                ok "✅ ${stable_threshold}s 内无新节点注册，以 ${NUM_NODES} 个节点启动"
                break
            fi
            if [[ "$waited" -ge "$NODE_WAIT_TIMEOUT" ]]; then
                NUM_NODES="$registered"
                warn "等待超时，以 ${NUM_NODES} 个节点启动"
                break
            fi
            if (( waited % 10 == 0 )); then
                info "当前 ${registered} 个节点，等待中...（${waited}/${NODE_WAIT_TIMEOUT}s）"
            fi
            sleep 2
            waited=$((waited + 2))
        done
    fi

    echo ""
    info "最终节点列表（共 ${NUM_NODES} 个）:"
    local line_num=0
    while IFS= read -r line; do
        echo "    node_rank=${line_num}: ${line}"
        line_num=$((line_num + 1))
    done < "$DISCOVERY_FILE"
    echo ""
}

# ---------- 本地单机模式初始化 ----------
setup_local_mode() {
    echo ""
    echo -e "${BOLD}=========================================="
    echo "       本地单机模式"
    echo -e "==========================================${NC}"
    echo ""

    NODE_RANK=0
    MASTER_IP="127.0.0.1"
    MY_IP="127.0.0.1"
    NUM_NODES=1

    # 自动检测 GPU 数量（如果用户未指定）
    if [[ -z "$NUM_GPUS" ]]; then
        NUM_GPUS=$(detect_gpu_count)
        info "自动检测到 ${NUM_GPUS} 个 GPU"
    fi

    ok "✅ 本地单机模式"
    ok "   Master IP: $MASTER_IP (localhost)"
    ok "   GPU 数量: $NUM_GPUS"
    ok "   节点数: $NUM_NODES"
    echo ""
}

# ---------- 清理共享文件（仅多机模式 master 在正常退出时清理）----------
cleanup_discovery() {
    if [[ "$MODE" == "multi" ]] && [[ "$NODE_RANK" == "0" ]] && [[ -n "${DISCOVERY_FILE:-}" ]]; then
        if [[ "$USER_CMD_EXIT_CODE" -eq 0 ]]; then
            info "Master 正常退出，清理共享文件: $DISCOVERY_FILE"
            rm -f "$DISCOVERY_FILE" "${DISCOVERY_FILE}.lock" 2>/dev/null || true
        else
            warn "Master 异常退出（exit_code=$USER_CMD_EXIT_CODE），保留共享文件: $DISCOVERY_FILE"
            warn "如需手动清理: rm -f $DISCOVERY_FILE ${DISCOVERY_FILE}.lock"
        fi
    fi
}
trap cleanup_discovery EXIT

# ---------- 系统信息采集 ----------
print_system_info() {
    echo ""
    echo "=========================================="
    echo "       系统信息"
    echo "=========================================="
    echo ""

    info "主机名: $(hostname)"
    info "本机 IP: $MY_IP"
    info "运行模式: $MODE"
    info "node_rank: $NODE_RANK"
    info "MASTER_IP: $MASTER_IP"
    info "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # PyTorch + GPU 环境
    python3 -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 编译版本: {torch.version.cuda}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU 数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
print(f'NCCL 版本: {\".\".join(map(str, torch.cuda.nccl.version()))}')
" 2>/dev/null || warn "PyTorch 环境检查失败"
    echo ""
}

# ---------- 配置 NCCL 环境 ----------
setup_nccl_env() {
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
    export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-5}"

    # 自动检测 NCCL_SOCKET_IFNAME（如果未设置）
    if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
        local auto_iface=""

        # 方法 1: 使用 ip 命令（需要 iproute2）
        if command -v ip &>/dev/null; then
            auto_iface=$(ip -4 -o addr show scope global 2>/dev/null | awk '{print $2}' | head -1 || true)
        fi

        # 方法 2: 从 /proc/net/route 解析默认路由网卡（不依赖外部命令）
        if [[ -z "$auto_iface" ]] && [[ -f /proc/net/route ]]; then
            auto_iface=$(awk '$2 == "00000000" {print $1; exit}' /proc/net/route 2>/dev/null || true)
        fi

        # 方法 3: 从 /sys/class/net 中找第一个非 lo 的网卡
        if [[ -z "$auto_iface" ]] && [[ -d /sys/class/net ]]; then
            for iface_dir in /sys/class/net/*; do
                local iface_name
                iface_name=$(basename "$iface_dir")
                if [[ "$iface_name" != "lo" ]] && [[ -d "$iface_dir" ]]; then
                    auto_iface="$iface_name"
                    break
                fi
            done
        fi

        if [[ -n "$auto_iface" ]]; then
            export NCCL_SOCKET_IFNAME="$auto_iface"
            info "自动检测 NCCL_SOCKET_IFNAME=$auto_iface"
        else
            warn "未检测到可用网卡，NCCL 将自行选择（可能失败）"
            warn "建议: 手动设置 -e NCCL_SOCKET_IFNAME=<网卡名> 或安装 iproute2"
        fi
    else
        info "使用指定的 NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
    fi

    # 检查 MASTER_IP 是否为合法 IPv4 地址
    if [[ -z "$MASTER_IP" ]]; then
        fail "MASTER_IP 为空！"
        fail "请检查网络配置或手动设置 MASTER_IP 环境变量"
        exit 1
    fi
    if ! echo "$MASTER_IP" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$'; then
        warn "MASTER_IP='$MASTER_IP' 不是标准 IPv4 地址，尝试 DNS 解析..."
        local resolved_ip
        resolved_ip=$(getent hosts "$MASTER_IP" 2>/dev/null | awk '{print $1}' | head -1 || true)
        if [[ -n "$resolved_ip" ]]; then
            info "DNS 解析成功: $MASTER_IP -> $resolved_ip"
            MASTER_IP="$resolved_ip"
        else
            fail "无法解析 MASTER_IP='$MASTER_IP'，请手动指定 IP 地址"
            exit 1
        fi
    fi
    info "最终 MASTER_IP: $MASTER_IP"
}

# ---------- 导出分布式环境变量 ----------
export_dist_env() {
    export NODE_RANK
    export MASTER_IP
    export MASTER_ADDR="$MASTER_IP"
    export MASTER_PORT
    export NUM_GPUS
    export NUM_NODES
    export CONNECTION_TIMEOUT

    echo ""
    echo "=========================================="
    echo "       分布式环境变量"
    echo "=========================================="
    echo ""
    info "NODE_RANK=$NODE_RANK"
    info "MASTER_IP=$MASTER_IP"
    info "MASTER_ADDR=$MASTER_IP"
    info "MASTER_PORT=$MASTER_PORT"
    info "NUM_GPUS=$NUM_GPUS"
    info "NUM_NODES=$NUM_NODES"
    info "CONNECTION_TIMEOUT=$CONNECTION_TIMEOUT"
    echo ""

    # 持久化环境变量到文件，供 docker exec 进入的 shell 使用
    _persist_env
}

# ---------- 持久化环境变量到文件（供 docker exec 使用）----------
_persist_env() {
    local env_file="/app/.dist_env"
    local profile_file="/etc/profile.d/dist_env.sh"

    # 收集需要持久化的环境变量
    cat > "$env_file" << ENVEOF
# 由 entrypoint.sh 自动生成，供 docker exec 使用
export NODE_RANK="${NODE_RANK}"
export MASTER_IP="${MASTER_IP}"
export MASTER_ADDR="${MASTER_IP}"
export MASTER_PORT="${MASTER_PORT}"
export NUM_GPUS="${NUM_GPUS}"
export NUM_NODES="${NUM_NODES}"
export CONNECTION_TIMEOUT="${CONNECTION_TIMEOUT}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-5}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"
export MY_IP="${MY_IP}"
export MODE="${MODE}"
export TASK_ID="${TASK_ID:-}"
ENVEOF

    # 如果有 TMPDIR 也写入
    if [[ -n "${TMPDIR:-}" ]]; then
        echo "export TMPDIR=\"${TMPDIR}\"" >> "$env_file"
        echo "export TEMP=\"${TEMP:-}\"" >> "$env_file"
        echo "export TMP=\"${TMP:-}\"" >> "$env_file"
    fi

    # 复制到 /etc/profile.d/ 让 login shell 自动加载
    mkdir -p /etc/profile.d 2>/dev/null || true
    cp "$env_file" "$profile_file" 2>/dev/null || true

    # 同时写入 /root/.bashrc（非 login shell 也能加载）
    local bashrc="/root/.bashrc"
    local source_line="[ -f /app/.dist_env ] && source /app/.dist_env"
    if [[ -f "$bashrc" ]]; then
        if ! grep -qF "$source_line" "$bashrc" 2>/dev/null; then
            echo "$source_line" >> "$bashrc"
        fi
    else
        echo "$source_line" > "$bashrc"
    fi

    info "环境变量已持久化到 $env_file（docker exec 可自动加载）"
}

# ---------- 执行用户命令 ----------
run_user_command() {
    echo ""
    echo -e "${BOLD}=========================================="
    echo "       执行用户命令"
    echo -e "==========================================${NC}"
    echo ""

    if [[ "$MODE" == "local" ]]; then
        info "模式: 本地单机, ${NUM_GPUS} GPU"
    else
        info "模式: 分布式, ${NUM_GPUS} GPU × ${NUM_NODES} 节点 = $((NUM_GPUS * NUM_NODES)) 进程"
    fi
    info "USER_COMMAND: $USER_COMMAND"
    echo ""

    # 不让 set -e 直接杀掉脚本，手动捕获退出码
    set +e
    eval "$USER_COMMAND"
    USER_CMD_EXIT_CODE=$?
    set -e

    if [[ "$USER_CMD_EXIT_CODE" -ne 0 ]]; then
        fail "用户命令退出，exit_code=$USER_CMD_EXIT_CODE"
        if [[ "$NUM_NODES" -gt 1 ]]; then
            fail "多机模式常见原因:"
            fail "  1. 其他节点未启动或未及时连接"
            fail "  2. 端口 ${MASTER_PORT} 被占用或防火墙阻断"
            fail "  3. Master IP ${MASTER_IP} 从其他节点不可达"
            fail "  4. NCCL 网络接口配置不正确"
        fi
        return $USER_CMD_EXIT_CODE
    fi
}

# ---------- 汇总 ----------
print_summary() {
    echo ""
    echo "=========================================="
    echo "       执行完成"
    echo "=========================================="
    echo ""
    info "本机 IP: $MY_IP"
    info "运行模式: $MODE"
    info "node_rank: $NODE_RANK"
    info "MASTER_IP: $MASTER_IP"
    if [[ "$MODE" == "local" ]]; then
        info "模式: 本地单机, ${NUM_GPUS} GPU"
    else
        info "模式: 分布式, ${NUM_GPUS} GPU × ${NUM_NODES} 节点"
    fi
    info "退出码: $USER_CMD_EXIT_CODE"
    echo ""
}

# ==========================================
#                 主流程
# ==========================================

echo ""
echo -e "${BOLD}=========================================="
echo "       通用分布式训练入口"
echo -e "==========================================${NC}"
echo ""
info "MODE=$MODE"
info "TASK_ID=${TASK_ID:-<未设置>}"
info "USER_COMMAND=$USER_COMMAND"
echo ""

# ---- 将临时目录指向 /data1（如果存在），避免容器 /tmp 空间不足 ----
if [[ -d "/data1" ]] || mkdir -p /data1 2>/dev/null; then
    LOCAL_TMPDIR="/data1/tmp"
    mkdir -p "$LOCAL_TMPDIR" 2>/dev/null || true
    export TMPDIR="$LOCAL_TMPDIR"
    export TEMP="$LOCAL_TMPDIR"
    export TMP="$LOCAL_TMPDIR"
    info "临时目录已切换到: $LOCAL_TMPDIR"
fi

# ---- 安装并配置 CFS Turbo 客户端 ----
info "正在安装 CFS Turbo 客户端..."
wget -q http://mirrors.tencentyun.com/install/cfsturbo-client/tools/cfs_turbo_client_setup
bash cfs_turbo_client_setup -c
ok "CFS Turbo 客户端安装完成"

# ---- 挂载 CFS 文件系统（通过 CFS_MOUNTS 环境变量配置）----
CFS_OK=false
CFS_FIRST_MOUNT=""
if [[ -n "${CFS_MOUNTS:-}" ]]; then
    info "通过 CFS_MOUNTS 环境变量挂载文件系统..."
    # 先提取第一个挂载点路径（用于拼接 DISCOVERY_FILE）
    CFS_FIRST_MOUNT=$(python3 -c "
import json, os
data = json.loads(os.environ.get('CFS_MOUNTS', '{}'))
mounts = data.get('mounts', [])
if mounts:
    print(mounts[0]['mount_point'])
" 2>/dev/null || true)

    if python3 << 'MOUNT_EOF'
import json, subprocess, os, sys

mounts_json = os.environ.get("CFS_MOUNTS", "{}")
data = json.loads(mounts_json)

for m in data.get("mounts", []):
    mount_point = m["mount_point"]
    subprocess.run(["mkdir", "-p", mount_point], check=True)
    subprocess.run(m["command"], shell=True, check=True)
    print(f"[mount] {m['id']} -> {mount_point}")

print(f"[mount] All {len(data.get('mounts', []))} volumes mounted.")
MOUNT_EOF
    then
        ok "CFS 挂载完成"
        CFS_OK=true
    else
        warn "CFS 挂载失败！请检查 CFS_MOUNTS 环境变量配置"
        warn "多机模式需要共享文件系统来选举 master"
    fi
elif [[ "$MODE" == "multi" ]]; then
    warn "多机模式但未设置 CFS_MOUNTS 环境变量"
    warn "请确保共享文件目录已手动挂载"
fi

# ---- 自动拼接 DISCOVERY_FILE（如果用户未显式指定）----
if [[ -z "$DISCOVERY_FILE" ]]; then
    if [[ -n "$CFS_FIRST_MOUNT" ]] && [[ -n "$TASK_ID" ]]; then
        DISCOVERY_FILE="${CFS_FIRST_MOUNT}/${DISCOVERY_PREFIX}/${TASK_ID}/discovery"
        info "自动拼接选举文件路径: $DISCOVERY_FILE"
    elif [[ -n "$CFS_FIRST_MOUNT" ]]; then
        warn "未设置 TASK_ID，使用默认选举路径"
        DISCOVERY_FILE="${CFS_FIRST_MOUNT}/${DISCOVERY_PREFIX}/default/discovery"
        info "选举文件路径: $DISCOVERY_FILE"
    else
        DISCOVERY_FILE="/cfs-hooke/${DISCOVERY_PREFIX}/${TASK_ID:-default}/discovery"
        warn "未检测到 CFS 挂载点，使用兜底路径: $DISCOVERY_FILE"
    fi
fi
echo ""

# ---- 模式分发 ----
case "$MODE" in
    "local")
        setup_local_mode
        ;;
    "multi")
        # 多机模式：NUM_GPUS 默认 8
        if [[ -z "$NUM_GPUS" ]]; then
            NUM_GPUS=8
        fi

        # 确保共享目录存在
        mkdir -p "$(dirname "$DISCOVERY_FILE")" 2>/dev/null || true
        info "共享发现文件: $DISCOVERY_FILE"

        discover_master
        ;;
    *)
        fail "未知模式: $MODE（可选: multi, local）"
        exit 1
        ;;
esac

# ---- 配置 NCCL 环境 ----
setup_nccl_env

# ---- 导出分布式环境变量 ----
export_dist_env

# ---- 系统信息 ----
print_system_info

# ---- 执行用户命令 ----
run_user_command || exit $USER_CMD_EXIT_CODE

# ---- 汇总 ----
print_summary
