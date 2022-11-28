#!/bin/bash
batch_size=$1
master_ip=$2
workers=$3
log=$4
source /home/ubuntu/ptorch/bin/activate
torchrun --nnodes $workers --nproc_per_node 1 --rdzv_backend c10d --rdzv_endpoint $master_ip --rdzv_id 54321 --role trainer torchrec_dlrm/dlrm_main.py --num_embeddings_per_feature "1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572" --embedding_dim 48 --pin_memory --over_arch_layer_sizes "1024,1024,512,256,1" --dense_arch_layer_sizes "512,256,48" --epochs 1 --shuffle_batches --in_memory_binary_criteo_path /home/ubuntu/torchrec_dataset 2>&1 | tee out