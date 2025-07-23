#!/usr/bin/env bash

CONFIG=$1

#PORT=${PORT:-$(2)}
PORT=-$2

#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
echo $(dirname "$0")
#GPUS=2
#CUDA_VISIBLE_DEVICES="2,3" python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
CUDA_VISIBLE_DEVICES="$2" python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch

