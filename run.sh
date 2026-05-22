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

# Start Docker Container
bash /llm_workspace_1P/robin/EasyInfer/scripts/docker/prepare_docker_nodes.sh start 
bash /llm_workspace_1P/robin/EasyInfer/scripts/cluster/start_ray_cluster.sh start \
    --file /llm_workspace_1P/robin/EasyInfer/scripts/node_list.txt

# Start Ray Cluster
bash /llm_workspace_1P/robin/EasyInfer/scripts/ray_cluster/start_npuslim_ray_cluster.sh start \
    --file /llm_workspace_1P/robin/EasyInfer/scripts/node_list.txt

# start kimi2_pcl
nohup bash /llm_workspace_1P/robin/EasyInfer/examples/kimi2_pcl.sh > /llm_workspace_1P/robin/EasyInfer/kimi2_pcl.log 2>&1 &
nohup bash /llm_workspace_1P/robin/EasyInfer/examples/lm_eval.sh > /llm_workspace_1P/robin/EasyInfer/lm_eval.log 2>&1 &

## Start Longcat Flash Chat
nohup bash /llm_workspace_1P/robin/EasyInfer/examples/longcat_flash-chat.sh > /llm_workspace_1P/robin/EasyInfer/longcat_flash-chat.log 2>&1 &