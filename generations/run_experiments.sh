#!/bin/bash

# GPU 0
CUDA_VISIBLE_DEVICES=0 python consume.py --model p160m --generations 10 --lineage synthetic --real-percent 10 --resume-gen 5 &
CUDA_VISIBLE_DEVICES=0 python consume.py --model p160m --generations 10 --lineage synthetic --real-percent 25 --resume-gen 5

# GPU 1
CUDA_VISIBLE_DEVICES=1 python consume.py --model p160m --generations 10 --lineage synthetic --real-percent 50 --resume-gen 5 
CUDA_VISIBLE_DEVICES=1 python consume.py --model p160m --generations 10 --lineage synthetic --real-percent 75 --resume-gen 7 &
CUDA_VISIBLE_DEVICES=1 python consume.py --model p160m --generations 10 --lineage synthetic --real-percent 90 --resume-gen 5 &

wait
echo "All experiments finished."