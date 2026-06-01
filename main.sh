# apt-get update
# apt-get install -y build-essential
# pip install -U deepspeed


# (main code) Paste it into the platform.
set -euo pipefail
mkdir -p /kk3essw1
mount -t nfs -o vers=3,nolock,proto=tcp,noresvport 21.99.249.13:/kk3essw1 /kk3essw1
mkdir -p /cfs-r3ufsqcb
mount -t lustre 30.170.137.65@tcp:/r3ufsqcb/cfs /cfs-r3ufsqcb
cd /kk3essw1/lixinghe/stable-audio-tools/
export NVCC_PREPEND_FLAGS=""
eval "$(conda shell.bash hook)"
conda activate stableaudio
bash train_dit.sh > "train_dit_node_${NODE_RANK:-0}.log" 2>&1

# running local test
sudo docker rm setup
sudo docker run -it --gpus all --cpus=64 --memory 256G --shm-size 64G \
    --privileged \
    --name setup \
    -p 8080:8080 \
    -e MODE=local \
    -e USER_COMMAND='
    set -euo pipefail &&
    mkdir -p /kk3essw1 &&
    mount -t nfs -o vers=3,nolock,proto=tcp,noresvport 21.99.249.13:/kk3essw1 /kk3essw1 &&
    cd /kk3essw1/lixinghe/stable-audio-tools/ &&
    mkdir -p /cfs-r3ufsqcb &&
    mount -t lustre 30.170.137.65@tcp:/r3ufsqcb/cfs /cfs-r3ufsqcb &&
    export NVCC_PREPEND_FLAGS="" &&
    eval "$(conda shell.bash hook)" &&
    conda activate stableaudio && 
    bash train_dit.sh > "train_dit_node_${NODE_RANK:-0}.log" 2>&1 &&
    sleep infinity' \
    harbor.music.woa.com/lixinghe/stableaudio3:v0.3

# setup and test only (no training)
sudo docker rm setup
sudo docker run -it --gpus all --cpus=64 --memory 256G --shm-size 64G \
    --privileged \
    --name setup \
    -p 8080:8080 \
    -e MODE=local \
    -e USER_COMMAND='mkdir -p /kk3essw1 &&
    mount -t nfs -o vers=3,nolock,proto=tcp,noresvport 21.99.249.13:/kk3essw1 /kk3essw1 &&
    cd /kk3essw1/lixinghe/stable-audio-tools/ &&
    mkdir -p /cfs-r3ufsqcb &&
    mount -t lustre 30.170.137.65@tcp:/r3ufsqcb/cfs /cfs-r3ufsqcb &&
    export NVCC_PREPEND_FLAGS="" &&
    eval "$(conda shell.bash hook)" &&
    conda activate stableaudio && 
    sleep infinity' \
    harbor.music.woa.com/lixinghe/stableaudio3:v0.3
