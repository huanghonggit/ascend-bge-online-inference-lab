#!/usr/bin/env bash
# Run on the assigned Linux host. The two physical card IDs must belong to this user.
set -euo pipefail
IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12
CONTAINER_NAME=${CONTAINER_NAME:-bge-online-lab}
HOST_WORKSPACE=${HOST_WORKSPACE:-$HOME/bge-online-workspace}
NPU_FIRST=${NPU_FIRST:?Set your first assigned physical NPU ID}
NPU_SECOND=${NPU_SECOND:?Set your second assigned physical NPU ID}
[[ "$NPU_FIRST" =~ ^[0-9]+$ && "$NPU_SECOND" =~ ^[0-9]+$ && "$NPU_FIRST" != "$NPU_SECOND" ]]
[[ -c /dev/davinci${NPU_FIRST} && -c /dev/davinci${NPU_SECOND} ]]
if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  printf 'Container already exists: %s. Reuse it; this script will not replace it.\n' "$CONTAINER_NAME" >&2
  exit 1
fi
mkdir -p "$HOST_WORKSPACE"
docker pull "$IMAGE"
docker run -d --network host --name "$CONTAINER_NAME" \
  --device /dev/davinci${NPU_FIRST} --device /dev/davinci${NPU_SECOND} \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v "$HOST_WORKSPACE":/workspace \
  --security-opt label=disable --shm-size=8g --cpus=16 \
  -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 -e TOKENIZERS_PARALLELISM=false \
  -w /workspace "$IMAGE" sleep infinity
docker inspect --format '{{json .HostConfig.Devices}}' "$CONTAINER_NAME"
