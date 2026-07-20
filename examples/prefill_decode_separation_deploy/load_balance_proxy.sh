unset http_proxy
unset https_proxy

node_p0_ip="10.18.1.10"
node_p1_ip="10.18.1.11"
node_p2_ip="10.18.1.12"
node_p3_ip="10.18.1.13"

node_d0_ip="10.18.1.14"
node_d1_ip="10.18.1.15"
node_d2_ip="10.18.1.16"
node_d3_ip="10.18.1.17"

python load_balance_proxy_server_example.py \
    --port 8000 \
    --host 0.0.0.0 \
    --prefiller-hosts \
       $node_p0_ip \
       $node_p1_ip \
       $node_p2_ip \
       $node_p3_ip \
    --prefiller-ports \
       9081 9081 \
       9081 9081 \
    --decoder-hosts \
      $node_d0_ip \
      $node_d0_ip \
      $node_d1_ip \
      $node_d1_ip \
      $node_d2_ip \
      $node_d2_ip \
      $node_d3_ip \
      $node_d3_ip \
    --decoder-ports \
      9900 9901 9900 9901 \
      9900 9901 9900 9901 
