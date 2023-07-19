export CUDA_VISIBLE_DEVICES=1
python -W ignore finetune.py     \
 -a qmobilenetv2                 \
 --workers 32                    \
 --test_batch 512                \
 --gpu_id 1                \
 --free_high_bit False           \
 --linear_quantization           \
 --eval                          \
 --pretrained
