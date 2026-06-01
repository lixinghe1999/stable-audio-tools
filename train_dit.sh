#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# stable_audio_3_0_master Training Script
# GPU platform compatible: single GPU, multi-GPU, and multi-node.
# ------------------------------------------------------------

# ---------- Configuration ----------
PRETRANSFORM_CKPT_PATH="${PRETRANSFORM_CKPT_PATH:-SAME-L/model.safetensors}"
MODEL_CONFIG="${MODEL_CONFIG:-stable_audio_tools/configs/model_configs/txt2audio/stable_audio_3_0_master.json}"
DATASET_CONFIG="${DATASET_CONFIG:-stable_audio_tools/configs/dataset_configs/paired_latent_jsonl_train.json}"
VAL_DATASET_CONFIG="${VAL_DATASET_CONFIG:-stable_audio_tools/configs/dataset_configs/paired_letent_jsonl_val.json}"
NAME="${NAME:-stable_audio_3_0_master}"
SAVE_DIR="${SAVE_DIR:-${TENSORBOARD_LOGGING_DIR:-./checkpoints}}"

BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10000}"
SEED="${SEED:-42}"

# Use environment variables exported by /app/entrypoint.sh.
# These are commonly auto-set by GPU platforms:
#   NUM_GPUS, NUM_NODES, NODE_RANK, MASTER_ADDR, MASTER_PORT
NUM_GPUS="${NUM_GPUS:-8}"
NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

LOGGER="tensorboard"
STRATEGY="${STRATEGY:-deepspeed}"

echo ">>> Training output"
echo "    stdout/stderr: docker logs -f <node-container>"
echo "    save root:     $SAVE_DIR"
echo "    tensorboard:   $SAVE_DIR/$NAME/<time-stamp>/"
echo "    checkpoints:   $SAVE_DIR/$NAME/<time-stamp>/checkpoints/"
echo ""

TRAIN_ARGS=(
    train.py
    --pretransform-ckpt-path "$PRETRANSFORM_CKPT_PATH"
    --model-config "$MODEL_CONFIG"
    --dataset-config "$DATASET_CONFIG"
    --val-dataset-config "$VAL_DATASET_CONFIG"
    --name "$NAME"
    --logger "$LOGGER"
    --strategy "$STRATEGY"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --num-nodes "$NUM_NODES"
    --checkpoint-every "$CHECKPOINT_EVERY"
    --save-dir "$SAVE_DIR"
    --seed "$SEED"
    "$@"
)

# ---------- Single-GPU / Single-Node ----------
if [ "$NUM_GPUS" -eq 1 ] && [ "$NUM_NODES" -eq 1 ]; then
    echo ">>> Single-GPU training"
    python "${TRAIN_ARGS[@]}"
    exit 0
fi

# ---------- Multi-GPU / Multi-Node (torchrun) ----------
echo ">>> Distributed training: ${NUM_GPUS} GPUs x ${NUM_NODES} nodes"

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --nnodes="$NUM_NODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "${TRAIN_ARGS[@]}"
