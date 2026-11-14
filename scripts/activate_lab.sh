# Source this file in the Linux container: source scripts/activate_lab.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH="${LAB_VENV_DIR:-/opt/bge-venv}/bin:$PATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
