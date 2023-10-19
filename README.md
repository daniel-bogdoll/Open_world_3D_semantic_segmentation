
# Open-world semantic segmentation for Lidar Point Clouds
Official implementation of "Open-world semantic segmentation for Lidar Point Clouds", ECCV 2022.
After saving the corresponding inference result files using this repository, please use [semantic_kitti_api](https://github.com/Jun-CEN/semantic_kitti_api) and [nuScenes_api](https://github.com/Jun-CEN/nuScenes_api) to evaluate the performance.

## Installation

1. Clone the repository
2. Setup a [venv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) with `python3 -m venv env`
3. Source the virtual environment with `source env/bin/activate`
4. Update pip with `pip install --upgrade pip`
5. Install all dependencies with `pip install -r requirements.txt`

### Test environment

- RTX 3080 Mobile (8 GB)
- Ubuntu 20.04.6 LTS (running in WSL2 under Windows 11)
- Python 3.8.10
- NVIDIA-SMI 545.23.05 (Driver Version: 545.84 CUDA Version: 12.3)

## Data Preparation

### SemanticKITTI
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
    ├──sequences
        ├── 00/           
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |       ├── 000000.label
        |       ├── 000001.label
        |       └── ...
        ├── 08/ # for validation
        ├── 11/ # 11-21 for testing
        └── 21/
	    └── ...
```

### nuScenes
```
./
├── ...
├── v1.0-trainval
├── v1.0-test
├── samples
├── sweeps
├── maps
└── lidarseg/
    ├──v1.0-trainval/
    ├──v1.0-mini/
    ├──v1.0-test/
    ├──nuscenes_infos_train.pkl
    ├──nuscenes_infos_val.pkl
    ├──nuscenes_infos_test.pkl
└── panoptic/
    ├──v1.0-trainval/
    ├──v1.0-mini/
    ├──v1.0-test/
```

## Checkpoints
We provide the checkpoints of open-set model and incremental learning model here: [checkpoints](https://drive.google.com/drive/folders/1GopqXwTen7jcq1q4tI0_BY20AEMVX4bN?usp=share_link)


## Open-set semantic segmentation
### Training for SemanticKITTI
All scripts for SemanticKITTI dataset is in `./semantickitti_scripts`.
#### MSP/Maxlogit method
```
./train_naive.sh
```
#### Upper bound
```
./train_upper.sh
```
#### RCF - Predictive Distribution Calibration
- Change the path of pretrained naive model in `/config/semantickitti_ood_basic.yaml`, line 63.

- Change the coefficient lamda_1 in `/config/semantickitti_ood_basic.yaml`, line 70.

- Change the dummy classifier number in `/train_cylinder_asym_ood_basic.py`, line 198.
```
./train_ood_basic.sh
```
#### RCF - Unknown Object Synthesis

- Change the path of pretrained naive model in `/config/semantickitti_ood_final.yaml`, line 63.

- Change lamda_1, lamda_2 in `/config/semantickitti_ood_final.yaml`, line 70, 71.

- Change the dummy classifier number in `/train_cylinder_asym_ood_final.py`, line 198.
```
./train_ood_final.sh
```

#### MC-Dropout

```
./train_dropout.sh
```

### Evaluation for SemanticKITTI
We save the in-distribution prediction labels and uncertainty scores for every points in the validation set, 
and these files will be used to calculate the closed-set mIoU and open-set metrics including AUPR, AURPC, and FPR95.
#### MSP/Maxlogit
- Change the trained model path (Naive method) in `/config/semantickitti.yaml`, line 63.

- Change the saving path of in-distribution prediction results and uncertainty scores in `val_cylinder_asym.py`, line 112, 114, 116.
```
./val.sh
```

#### Upper bound
- Change the trained model path (Placeholder method) in `/config/semantickitti.yaml`, line 63.

- Change the saving path of in-distribution prediction results and uncertainty scores in `val_cylinder_asym_upper.py`, line 115, 117.

```
./val_upper.sh
```

#### RCF
- Change the trained model path (Placeholder method) in `/config/semantickitti_ood_final.yaml`, line 63.

- Change the saving path of in-distribution prediction results and uncertainty scores in `val_cylinder_asym_ood.py`, line 124, 125.

```
./val_ood.sh
```

#### MC-Dropout
```
./val_dropout.sh
```

### Training for nuScenes
All scripts for nuScenes dataset are in `./nuScenes_scripts`
#### MSP/Maxlogit method
```
./train_nusc_naive.sh
```
#### Upper bound
```
./train_nusc.sh
```
#### RCF - Predictive Distribution Calibration
- Change the path of pretrained naive model in `/config/nuScenes_ood_basic.yaml`, line 63.

- Change the coefficient lamda_1 in `/config/nuScenes_ood_basic.yaml`, line 70.

- Change the dummy classifier number in `/train_cylinder_asym_nuscenes_ood_basic.py`, line 197.
```
./train_nusc_ood_basic.sh
```
#### RCF - Unknown Object Synthesis

- Change the path of pretrained naive model in `/config/nuScenes_ood_final.yaml`, line 63.

- Change lamda_1, lamda_2 in `/config/nuScenes_ood_final.yaml`, line 70, 71.

- Change the dummy classifier number in `/train_cylinder_asym_nuscenes_ood_final.py`, line 197.
```
./train_nusc_ood_final.sh
```

#### MC-Dropout
```
./train_nusc_dropout.sh
```

### Evaluation for nuScenes

#### MSP/Maxlogit
- Change the trained model path (Naive method) in `/config/nuScenes.yaml`, line 63.

- Change the saving path of in-distribution prediction results and uncertainty scores in `val_cylinder_asym_nusc.py`, line 112, 114, 116.
```
./val_nusc.sh
```

#### Upper bound
- Change the trained model path (Naive method) in `/config/nuScenes.yaml`, line 63.

- Change the saving path of in-distribution prediction results and uncertainty scores in `val_cylinder_asym_nusc_upper.py`, line 121, 123.
```
./val_nusc_upper.sh
```

#### RCF
- Change the trained model path (Placeholder method) in `/config/nuScenes_ood_final.yaml`, line 63.

- Change the saving path of in-distribution prediction results and uncertainty scores in `val_cylinder_asym_nusc_ood.py`, line 125, 126.

```
./val_nusc_ood.sh
```

#### MC-Dropout
```
./val_nusc_dropout.sh
```

## Incremental learning
### Training for SemanticKITTI
All scripts for SemanticKITTI dataset is in `./semantickitti_scripts`.

First, use the trained base model to generate and save the pseudo labels of the training set:
```
./val_generate_incre_labels.sh
```
- Change the trained model path in `/config/semantickitti_ood_generate_incre_labels.yaml`, line 63.
- Change the save path of pseudo labels in `val_cylinder_asym_generate_incre_labels.py`, line 116.

Then, change the loading path of pseudo labels in `/dataloader/pc_dataset.py`, line 177.

Now, conduct incremental learing using pseudo labels:
```
./train_incre.sh
```
- Change the trained model path in `/config/semantickitti_ood_incre.yaml`, line 63.

### Evaluation for SemanticKITTI
For validation set:
```
./val_incre.sh
```
For test set:
```
./test_incre.sh
```
- Change the `collate_fn=collate_fn_BEV_val_test` in `/builder/data_builder.py`, line 70.
- Change the save path in `/dataloader/pc_dataset.py`, line 95.
- Upload the generated files into the [evalution server](https://competitions.codalab.org/competitions/20331#participate).
### Training for nuScenes
All scripts for nuScenes dataset are in `./nuScenes_scripts`.
#### For new class 1(barrier)

First, generate and save the pseudo labels of the training set:
```
./val_nusc_generate_incre_labels.sh
```
- Change the trained model path in `/config/nuScenes_ood_generate_incre_labels.yaml`, line 63.
- Change the save path of pseudo labels in `val_cylinder_asym_nusc_generate_incre_labels.py`, line 120.

Then, change the loading path of pseudo labels in `/dataloader/pc_dataset.py`, line 266.

Now, conduct incremental learing using pseudo labels:
```
./train_nusc_incre.sh
```

#### For new class 5(construction-vehicle), 8(traffic-cone), 9(trailer)
First, generate and save the pseudo labels of the training set:
```
./val_nusc_generate_incre_labels.sh
```
- Change the python file from `val_cylinder_asym_nusc_generate_incre_labels.py` into
`val_cylinder_asym_nusc_generate_incre_labels_1.py`.
- Change the incremental class in `val_cylinder_asym_nusc_generate_incre_labels_1.py`, line 153.
- Change the trained model and save path similar with new class 1.

Then, change the loading path of pseudo labels in `/dataloader/pc_dataset.py`, line 266.

Now, conduct incremental learing using pseudo labels:
```
./train_nusc_incre.sh
```
- Change the trained model path in `/config/nuScenes_ood_incre.yaml`, line 63.

### Evaluation for nuScenes
For validation set:
```
./val_incre.sh
```
For test set:
- Change the NuScenes version from `nusc = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=True)`
to `nusc = NuScenes(version='v1.0-test', dataroot=data_path, verbose=True)`
```
./test_incre.sh
```
Then upload the generated files into the [evaluation server](https://eval.ai/web/challenges/challenge-page/720/my-submission).
