# 加载 CANN 环境变量（路径需根据实际安装位置调整）
install_path=/usr/local/Ascend
source $install_path/ascend-toolkit/set_env.sh
source $install_path/nnal/atb/set_env.sh

# llmtuner env
source /llm_workspace_1P/robin/miniconda3/bin/activate llmtuner
