name=cylinder_asym_networks
gpuid=1

CUDA_VISIBLE_DEVICES=${gpuid}  python -u train_cylinder_asym_naive.py \
2>&1 | tee /root/phd/log/${name}_logs_tee.txt