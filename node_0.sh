#!/usr/bin/env bash
set -euo pipefail

# Simulated multi-node rank 0 on one physical machine.
# Start this first; it owns the rendezvous/discovery file and uses GPUs 4,5.

NETWORK_NAME="${NETWORK_NAME:-stableaudio-multinode}"
IMAGE="${IMAGE:-harbor.music.woa.com/lixinghe/stableaudio3:v0.3}"
DISCOVERY_DIR="${DISCOVERY_DIR:-/tmp/stableaudio-multinode-discovery}"
HOST_GPUS="${NODE0_GPUS:-4,5}"
GPU_REQUEST="\"device=${HOST_GPUS}\""

NODE_NAME="${NODE_NAME:-stableaudio-node0}"
TASK_ID="${TASK_ID:-default}"
DISCOVERY_MOUNT="${DISCOVERY_MOUNT:-/cfs-hooke/distributed-discovery/${TASK_ID}}"
DISCOVERY_FILE="${DISCOVERY_FILE:-${DISCOVERY_MOUNT}/discovery}"
MASTER_PORT="${MASTER_PORT:-29500}"
NUM_NODES="${NUM_NODES:-2}"
NUM_GPUS="${NUM_GPUS:-2}"
NODE_WAIT_TIMEOUT="${NODE_WAIT_TIMEOUT:-1800}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

docker network create "$NETWORK_NAME" >/dev/null 2>&1 || true
mkdir -p "$DISCOVERY_DIR"
rm -f "$DISCOVERY_DIR/discovery" "$DISCOVERY_DIR/discovery.lock"
docker rm -f "$NODE_NAME" >/dev/null 2>&1 || true

docker run -d --gpus "$GPU_REQUEST" --cpus=64 --memory 256G --shm-size 64G \
    --privileged \
    --name "$NODE_NAME" \
    --network "$NETWORK_NAME" \
    -v "$DISCOVERY_DIR:$DISCOVERY_MOUNT" \
    -p 8080:8080 \
    -e MODE=multi \
    -e TASK_ID="$TASK_ID" \
    -e DISCOVERY_FILE="$DISCOVERY_FILE" \
    -e NUM_GPUS="$NUM_GPUS" \
    -e NUM_NODES="$NUM_NODES" \
    -e MASTER_PORT="$MASTER_PORT" \
    -e NODE_WAIT_TIMEOUT="$NODE_WAIT_TIMEOUT" \
    -e NVIDIA_VISIBLE_DEVICES="$HOST_GPUS" \
    -e CUDA_VISIBLE_DEVICES="$HOST_GPUS" \
    -e TRAIN_EXTRA_ARGS="$TRAIN_EXTRA_ARGS" \
    -e USER_COMMAND='mkdir -p /kk3essw1 &&
    mount -t nfs -o vers=3,nolock,proto=tcp,noresvport 21.99.249.13:/kk3essw1 /kk3essw1 &&
    cd /kk3essw1/lixinghe/stable-audio-tools/ &&
    mkdir -p /cfs-r3ufsqcb &&
    mount -t lustre 30.170.137.65@tcp:/r3ufsqcb/cfs /cfs-r3ufsqcb &&
    export NVCC_PREPEND_FLAGS="" &&
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}" &&
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" &&
    nvidia-smi &&
    eval "$(conda shell.bash hook)" &&
    conda activate stableaudio &&
    bash train_dit.sh ${TRAIN_EXTRA_ARGS:-} &&
    sleep infinity' \
    "$IMAGE"

echo "Started node 0:"
echo "  container: $NODE_NAME"
echo "  image:     $IMAGE"
echo "  network:   $NETWORK_NAME"
echo "  discovery: $DISCOVERY_DIR -> $DISCOVERY_FILE"
echo "  host GPUs: $HOST_GPUS"
echo "  topology:  $NUM_GPUS GPUs x $NUM_NODES nodes"
echo "  logs:      docker logs -f $NODE_NAME"
echo "  ckpts:     /kk3essw1/lixinghe/stable-audio-tools/checkpoints/stable_audio_3_0_master/<timestamp>/checkpoints/"
echo ""
echo "Start node 1 in another shell:"
echo "  bash node_1.sh"
echo ""
echo "Logs:"
echo "  docker logs -f $NODE_NAME"
