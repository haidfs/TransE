#!/usr/bin/env bash
if [ $# != 1 ] ; then
echo "USAGE: $0 model: 0-simple 1-manger 2-queue"
echo " e.g.: $0 0"
exit 1;
fi
if [[ "$1" -eq 0 ]]; then
mode="Simple"
elif [ "$1" == 1 ]; then
mode="Manager"
elif [ "$1" == 2 ]; then
mode="MpQueue"
else
echo "Illegel argument!"
exit 1;
fi
CUDA_VISIBLE_DEVICES=0 \
python TrainMain.py --embedding_dim 100 --margin_value 1 --normal_form "L1" --batch_size 10000 --learning_rate 0.003 --n_generator 24 --max_epoch 5000 --multi_process $mode