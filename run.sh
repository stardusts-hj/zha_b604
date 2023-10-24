#!/bin/bash

OUTDIR = '/mnt/share/lbh/'
DATADIR = '/mnt/datasets'

python -m torch.distributed.launch --nproc_per_node=4 train.py \
 --world_size 4 --batch_size 8 --lr 0.0001 \
 --data_path ${DATADIR}/vfidataset --
 --config reproduce --logdir ${OUTDIR}/output_submit