FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# 系统依赖（NFS 客户端，用于挂载 CFS 共享文件系统）
# netbase: provides /etc/protocols and /etc/services, required by mount.nfs
#          to resolve proto=tcp via getprotobyname(); missing in minimal images
RUN apt-get update && apt-get install -y --no-install-recommends \
        nfs-common \
        rpcbind \
        netbase \
        iproute2 \
        wget curl \
    && rm -rf /var/lib/apt/lists/*

# 复制入口脚本
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# 设置入口
ENTRYPOINT ["/app/entrypoint.sh"]