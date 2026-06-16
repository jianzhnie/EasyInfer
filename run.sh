#!/bin/bash

# reload docker
systemctl daemon-reload
systemctl enable --now docker
systemctl is-active --quiet docker

# exec docker
docker exec -it vllm-ascend-0.18-env /bin/bash
docker exec -it mindspeed-llm-env /bin/bash
docker exec -it mindspeed-llm-26-env /bin/bash
docker exec -it npuslim-env /bin/bash
docker exec -it vllm-ascend-env /bin/bash

# Find null nodes
bash /home/jianzhnie/llmtuner/llm/tools/find_null_nodes.sh \
    /home/jianzhnie/llmtuner/llm/tools/ip_list.txt

# load images 
bash /home/jianzhnie/llmtuner/llm/EasyInfer/scripts/docker/manage_docker_containers.sh start \
    --file /home/jianzhnie/llmtuner/llm/EasyInfer/available_nodes.txt

# start docker env
bash /home/jianzhnie/llmtuner/llm/EasyInfer/scripts/docker/manage_npuslim_containers.sh start \
    --file /home/jianzhnie/llmtuner/llm/EasyInfer/node_list.txt

# Start Ray Cluster
# bash /home/jianzhnie/llmtuner/llm/EasyInfer/scripts/ray_cluster/start_ray_cluster.sh start \
#     --file /home/jianzhnie/llmtuner/llm/EasyInfer/node_list.txt

# Start Ray Cluster for npuslim
bash /home/jianzhnie/llmtuner/llm/EasyInfer/scripts/ray_cluster/start_npuslim_ray_cluster.sh start \
    --file /home/jianzhnie/llmtuner/llm/EasyInfer/node_list.txt

# start kimi2_pcl
nohup bash /llm_workspace_1P/robin/EasyInfer/examples/kimi2_pcl.sh > /llm_workspace_1P/robin/EasyInfer/kimi2_pcl.log 2>&1 &
nohup bash /llm_workspace_1P/robin/EasyInfer/examples/lm_eval.sh > /llm_workspace_1P/robin/EasyInfer/lm_eval.log 2>&1 &

## Start Longcat Flash Chat
nohup bash /home/jianzhnie/llmtuner/llm/EasyInfer/examples/longcat_flash-chat.sh  > /home/jianzhnie/llmtuner/llm/EasyInfer/longcat_flash-chat3.log 2>&1 &

## evaluate
nohup bash /home/jianzhnie/llmtuner/llm/EasyInfer/examples/lm_eval.sh > /home/jianzhnie/llmtuner/llm/EasyInfer/lm_eval3.log 2>&1 &

