export CUDA_VISIBLE_DEVICES=2
python -W ignore finetune.py     \
 -a resnet50                     \
 --workers 32                    \
 --test_batch 512                \
 --free_high_bit False           \
 --eval                          \
 --pretrained
