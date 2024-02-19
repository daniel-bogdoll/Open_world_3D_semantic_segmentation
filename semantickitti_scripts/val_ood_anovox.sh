name=cylinder_asym_networks
gpuid=1

CUDA_VISIBLE_DEVICES=${gpuid}  python -u val_cylinder_asym_ood_anovox.py \
--config 'config/anovox_val.yaml'