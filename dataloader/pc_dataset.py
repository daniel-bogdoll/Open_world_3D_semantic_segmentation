# -*- coding:utf-8 -*-
# author: Xinge
# @file: pc_dataset.py

import os
import numpy as np
from torch.utils import data
import yaml
import pickle
import open3d as o3d
# from nuscenes.eval.lidarseg.utils import get_samples_in_eval_set

REGISTERED_PC_DATASET_CLASSES = {}


def register_dataset(cls, name=None):
    global REGISTERED_PC_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_PC_DATASET_CLASSES, f"exist class: {REGISTERED_PC_DATASET_CLASSES}"
    REGISTERED_PC_DATASET_CLASSES[name] = cls
    return cls


def get_pc_model_class(name):
    global REGISTERED_PC_DATASET_CLASSES
    assert name in REGISTERED_PC_DATASET_CLASSES, f"available class: {REGISTERED_PC_DATASET_CLASSES}"
    return REGISTERED_PC_DATASET_CLASSES[name]

@register_dataset
class AnoVox_val(data.Dataset):
    def __init__(self, data_path, imageset="val", return_ref=False, label_mapping="anovox-label.yaml", nusc=None):
        self.root = data_path
        self.return_ref = return_ref
        with open(label_mapping, "r") as stream:
            anovox_yaml = yaml.safe_load(stream)
        # self.learning_map = anovox_yaml["learning_map"]
        self.COLOR_PALETTE = anovox_yaml["color_map"]
        self.remap = np.array(anovox_yaml["to_SemKITTI"])
        self.remap_remap = anovox_yaml["learning_map"]
        self.datapath_list()
        # print('The size of %s data is %d'%(split,len(self.points_datapath)))


    def __len__(self):
        "Denotes the total number of samples"
        return len(self.labels_datapath)

    def datapath_list(self):
        def sorter(file_path):
            identifier = (os.path.basename(file_path).split('.')[0]).split('_')[-1]
            return int(identifier)
        print("root", self.root)
        self.points_datapath = []
        self.labels_datapath = []
        # self.instance_datapath = []

        for i, scenario in enumerate(os.listdir(self.root)):
            if scenario == 'Scenario_Configuration_Files':
                continue
            if i > 814: # 4070 / 5 = 814
                break
            point_dir = os.path.join(self.root, scenario, 'PCD')

            # print("point dir:", os.listdir(point_dir))
            # os.listdir(point_dir).sort()
            sem_point_dir = os.path.join(self.root, scenario, "SEMANTIC_PCD")
            # os.listdir(sem_point_dir).sort()
            try:
                self.points_datapath += [os.path.join(point_dir, point_file) for point_file in os.listdir(point_dir)]
                # self.points_datapath = sorted(points_datapath, key=sorter)
                self.labels_datapath += [os.path.join(sem_point_dir, sem_point_file) for sem_point_file in os.listdir(sem_point_dir)]
                # self.labels_datapath = sorted(labels_datapath, key=sorter)
            except:
                pass
        self.points_datapath = sorted(self.points_datapath, key=sorter)
        self.labels_datapath = sorted(self.labels_datapath, key=sorter)

        # print("points datapath:", self.points_datapath)


        # for seq in self.seq_ids[split]:
        #     point_seq_path = os.path.join(self.root, 'dataset', 'sequences', seq, 'velodyne')
        #     point_seq_bin = os.listdir(point_seq_path)
        #     point_seq_bin.sort()
        #     self.points_datapath += [ os.path.join(point_seq_path, point_file) for point_file in point_seq_bin ]

        #     try:
        #         label_seq_path = os.path.join(self.root, 'dataset', 'sequences', seq, 'labels')
        #         point_seq_label = os.listdir(label_seq_path)
        #         point_seq_label.sort()
        #         self.labels_datapath += [ os.path.join(label_seq_path, label_file) for label_file in point_seq_label ]
        #     except:
        #         pass

        #     try:
        #         instance_seq_path = os.path.join(self.root, 'dataset', 'sequences', seq, 'instances')
        #         point_seq_instance = os.listdir(instance_seq_path)
        #         point_seq_instance.sort()
        #         self.instance_datapath += [ os.path.join(instance_seq_path, instance_file) for instance_file in point_seq_instance ]
        #     except:
        #         pass

    def __getitem__(self, index):
        pcd = self.points_datapath[index]
        pcd = o3d.io.read_point_cloud(pcd)
        points_set = np.asarray(pcd.points)
        intensities = np.asarray(pcd.colors)[:,0].reshape(-1,1)


        semantic_pcd = self.labels_datapath[index]
        semantic_pcd = o3d.io.read_point_cloud(semantic_pcd)
        # semantic_points = np.asarray(semantic_pcd.points)
        color_labels = np.asarray(semantic_pcd.colors)

        # transform color labels to labels as integer value
        sem_labels = (np.asarray(color_labels) * 255.0).astype(np.uint8)
        new_labels = np.arange(len(sem_labels))
        for i, value in enumerate(sem_labels): # convert color into label
            color_index = np.where((self.COLOR_PALETTE == value).all(axis = 1))
            new_labels[i] = color_index[0][0]
        new_labels = self.remap[new_labels]
        new_labels = np.array([self.remap_remap[label] for label in new_labels])
        sem_labels = new_labels.reshape(-1,1)
        data_tuple = (points_set[:, :3], sem_labels.astype(np.uint8)) # instance_data.astype(np.uint8))
        if self.return_ref:
            # data_tuple += (intensities,)
            dummy_intensities = np.ones(new_labels.shape)
            dummy_intensities = dummy_intensities - 0.01
            data_tuple += (dummy_intensities,)

        return data_tuple


    # def __getitem__(self, index):
    #     raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
    #     if self.imageset == "test":
    #         annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
    #     else:
    #         annotated_data = np.fromfile(
    #             self.im_idx[index].replace("velodyne", "labels")[:-3] + "label", dtype=np.uint32
    #         ).reshape((-1, 1))
    #         semantic_data = annotated_data & 0xFFFF
    #         instance_data = annotated_data >> 16
    #         semantic_data = np.vectorize(self.learning_map.__getitem__)(semantic_data)

    #     data_tuple = (raw_data[:, :3], semantic_data.astype(np.uint8), instance_data.astype(np.uint8))
    #     if self.return_ref:
    #         data_tuple += (raw_data[:, 3],)
    #     return data_tuple
        # return {'points_cluster': points_set, 'semantic_label': sem_labels, 'scan_file': self.points_datapath[index]}

        # return {'points_cluster': points_set, 'scan_file': self.points_datapath[index]}


    def __len__(self):
        return len(self.points_datapath)


@register_dataset
class AnoVox_train(data.Dataset):
    def __init__(self, data_path, imageset="train", return_ref=False, label_mapping="anovox-label.yaml", nusc=None):
        self.root = data_path
        self.imageset = imageset
        self.return_ref = return_ref
        with open(label_mapping, "r") as stream:
            anovox_yaml = yaml.safe_load(stream)
        # self.learning_map = anovox_yaml["learning_map"]
        self.COLOR_PALETTE = anovox_yaml["color_map"]
        self.remap = np.array(anovox_yaml["to_SemKITTI"])
        self.train_remap = anovox_yaml["learning_map"]
        self.datapath_list()
        print("length: ", len(self.labels_datapath))

        # print('The size of %s data is %d'%(split,len(self.points_datapath)))


    def __len__(self):
        "Denotes the total number of samples"
        return len(self.labels_datapath)

    def datapath_list(self):
        def sorter(file_path):
            identifier = (os.path.basename(file_path).split('.')[0]).split('_')[-1]
            return int(identifier)
        print("root", self.root)
        self.points_datapath = []
        self.labels_datapath = []
        # self.instance_datapath = []

        for i, scenario in enumerate(os.listdir(self.root)):
            if scenario == 'Scenario_Configuration_Files':
                continue
            if i > 814 and self.imageset == "val": # 4070 / 5 = 814
                break
            point_dir = os.path.join(self.root, scenario, 'PCD')

            # print("point dir:", os.listdir(point_dir))
            # os.listdir(point_dir).sort()
            sem_point_dir = os.path.join(self.root, scenario, "SEMANTIC_PCD")
            # os.listdir(sem_point_dir).sort()
            try:
                self.points_datapath += [os.path.join(point_dir, point_file) for point_file in os.listdir(point_dir)]
                # self.points_datapath = sorted(points_datapath, key=sorter)
                # print("points datapath:", self.points_datapath)
                self.labels_datapath += [os.path.join(sem_point_dir, sem_point_file) for sem_point_file in os.listdir(sem_point_dir)]
                # self.labels_datapath = sorted(labels_datapath, key=sorter)
            except:
                pass
        self.points_datapath = sorted(self.points_datapath, key=sorter)
        self.labels_datapath = sorted(self.labels_datapath, key=sorter)

        # print("points datapath:", self.points_datapath)


        # for seq in self.seq_ids[split]:
        #     point_seq_path = os.path.join(self.root, 'dataset', 'sequences', seq, 'velodyne')
        #     point_seq_bin = os.listdir(point_seq_path)
        #     point_seq_bin.sort()
        #     self.points_datapath += [ os.path.join(point_seq_path, point_file) for point_file in point_seq_bin ]

        #     try:
        #         label_seq_path = os.path.join(self.root, 'dataset', 'sequences', seq, 'labels')
        #         point_seq_label = os.listdir(label_seq_path)
        #         point_seq_label.sort()
        #         self.labels_datapath += [ os.path.join(label_seq_path, label_file) for label_file in point_seq_label ]
        #     except:
        #         pass

        #     try:
        #         instance_seq_path = os.path.join(self.root, 'dataset', 'sequences', seq, 'instances')
        #         point_seq_instance = os.listdir(instance_seq_path)
        #         point_seq_instance.sort()
        #         self.instance_datapath += [ os.path.join(instance_seq_path, instance_file) for instance_file in point_seq_instance ]
        #     except:
        #         pass

    def __getitem__(self, index):
        pcd = self.points_datapath[index]
        pcd = o3d.io.read_point_cloud(pcd)
        points_set = np.asarray(pcd.points)
        intensities = np.asarray(pcd.colors)[:,0].reshape(-1,1)

        semantic_pcd = self.labels_datapath[index]
        semantic_pcd = o3d.io.read_point_cloud(semantic_pcd)
        # semantic_points = np.asarray(semantic_pcd.points)
        color_labels = np.asarray(semantic_pcd.colors)

        # transform color labels to labels as integer value
        sem_labels = (np.asarray(color_labels) * 255.0).astype(np.uint8)
        new_labels = np.arange(len(sem_labels))
        for i, value in enumerate(sem_labels): # convert color into label
            color_index = np.where((self.COLOR_PALETTE == value).all(axis = 1))
            new_labels[i] = color_index[0][0]
        new_labels = self.remap[new_labels]
        new_labels = np.array([self.train_remap[label] for label in new_labels])
        sem_labels = new_labels.reshape(-1,1)
        data_tuple = (points_set[:, :3], sem_labels.astype(np.uint8)) # instance_data.astype(np.uint8))
        if self.return_ref:
            # dummy_intensities = np.ones(new_labels.shape)
            # dummy_intensities = dummy_intensities - 0.01
            data_tuple += (intensities,)
        return data_tuple


@register_dataset
class SemKITTI_demo(data.Dataset):
    def __init__(
        self, data_path, imageset="demo", return_ref=True, label_mapping="semantic-kitti.yaml", demo_label_path=None
    ):
        with open(label_mapping, "r") as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml["learning_map"]
        self.imageset = imageset
        self.return_ref = return_ref

        self.im_idx = []
        self.im_idx += absoluteFilePaths(data_path)
        self.label_idx = []
        if self.imageset == "val":
            print(demo_label_path)
            self.label_idx += absoluteFilePaths(demo_label_path)

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        if self.imageset == "demo":
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        elif self.imageset == "val":
            annotated_data = np.fromfile(self.label_idx[index], dtype=np.uint32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
        if self.return_ref:
            data_tuple += (raw_data[:, 3],)
        return data_tuple


@register_dataset
class SemKITTI_sk(data.Dataset):
    def __init__(self, data_path, imageset="train", return_ref=False, label_mapping="semantic-kitti.yaml", nusc=None):
        self.return_ref = return_ref
        with open(label_mapping, "r") as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml["learning_map"]
        self.imageset = imageset
        if imageset == "train":
            split = semkittiyaml["split"]["train"]
        elif imageset == "val":
            split = semkittiyaml["split"]["valid"]
        elif imageset == "test":
            split = semkittiyaml["split"]["test"]
        else:
            raise Exception("Split must be train/val/test")

        self.im_idx = []
        for i_folder in split:
            self.im_idx += absoluteFilePaths("/".join([data_path, str(i_folder).zfill(2), "velodyne"]))
        print("length: ", len(self.im_idx))


    def __len__(self):
        "Denotes the total number of samples"
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        if self.imageset == "test":
            path_save = self.im_idx[index].replace("velodyne", "predictions")
            path_save = path_save.replace("bin", "label")
            path_save = path_save.replace("dataset", "predictions_test/predictions_incre_latest")
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(
                self.im_idx[index].replace("velodyne", "labels")[:-3] + "label", dtype=np.uint32
            ).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)
            # print("max: ", annotated_data.max())
            # print("min: ", annotated_data.min())

        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
        if self.return_ref:
            data_tuple += (raw_data[:, 3],)

        if self.imageset == "test":
            return data_tuple, path_save
        else:
            return data_tuple


@register_dataset
class SemKITTI_sk_panop(data.Dataset):
    def __init__(self, data_path, imageset="train", return_ref=False, label_mapping="semantic-kitti.yaml", nusc=None):
        self.return_ref = return_ref
        with open(label_mapping, "r") as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml["learning_map"]
        self.imageset = imageset
        if imageset == "train":
            split = semkittiyaml["split"]["train"]
        elif imageset == "val":
            split = semkittiyaml["split"]["valid"]
        elif imageset == "test":
            split = semkittiyaml["split"]["test"]
        else:
            raise Exception("Split must be train/val/test")

        self.im_idx = []
        for i_folder in split:
            self.im_idx += absoluteFilePaths("/".join([data_path, str(i_folder).zfill(2), "velodyne"]))

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        if self.imageset == "test":
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(
                self.im_idx[index].replace("velodyne", "labels")[:-3] + "label", dtype=np.uint32
            ).reshape((-1, 1))
            semantic_data = annotated_data & 0xFFFF
            instance_data = annotated_data >> 16
            semantic_data = np.vectorize(self.learning_map.__getitem__)(semantic_data)

        data_tuple = (raw_data[:, :3], semantic_data.astype(np.uint8), instance_data.astype(np.uint8))
        if self.return_ref:
            data_tuple += (raw_data[:, 3],)
        return data_tuple


@register_dataset
class SemKITTI_sk_panop_incre(data.Dataset):
    def __init__(self, data_path, imageset="train", return_ref=False, label_mapping="semantic-kitti.yaml", nusc=None):
        self.return_ref = return_ref
        with open(label_mapping, "r") as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml["learning_map"]
        self.imageset = imageset
        if imageset == "train":
            split = semkittiyaml["split"]["train"]
        elif imageset == "val":
            split = semkittiyaml["split"]["valid"]
        elif imageset == "test":
            split = semkittiyaml["split"]["test"]
        else:
            raise Exception("Split must be train/val/test")

        self.im_idx = []
        for i_folder in split:
            self.im_idx += absoluteFilePaths("/".join([data_path, str(i_folder).zfill(2), "velodyne"]))

        self.pred_names = []
        pred_paths = "/harddisk/jcenaa/semantic_kitti/predictions/sequences/08/predictions_base_train"
        # populate the label names
        seq_pred_names = [
            os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(pred_paths)) for f in fn if ".label" in f
        ]
        seq_pred_names.sort()
        self.pred_names.extend(seq_pred_names)

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        if self.imageset == "test":
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(
                self.im_idx[index].replace("velodyne", "labels")[:-3] + "label", dtype=np.uint32
            ).reshape((-1, 1))
            semantic_data = annotated_data & 0xFFFF
            instance_data = annotated_data >> 16
            semantic_data = np.vectorize(self.learning_map.__getitem__)(semantic_data)

        distill_label_path = self.pred_names[index]
        distill_label = np.fromfile(distill_label_path, dtype=int32)
        distill_label = distill_label.reshape([-1, 1])

        data_tuple = (
            raw_data[:, :3],
            semantic_data.astype(np.uint8),
            instance_data.astype(np.uint8),
            distill_label.astype(np.uint8),
        )
        if self.return_ref:
            data_tuple += (raw_data[:, 3],)
        return data_tuple


@register_dataset
class SemKITTI_nusc_panop(data.Dataset):
    def __init__(self, data_path, imageset="train", return_ref=False, label_mapping="nuscenes.yaml", nusc=None):
        self.return_ref = return_ref

        with open(imageset, "rb") as f:
            data = pickle.load(f)

        with open(label_mapping, "r") as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml["learning_map"]

        self.nusc_infos = data["infos"]
        self.data_path = data_path
        self.nusc = nusc

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.nusc_infos)

    def __getitem__(self, index):
        info = self.nusc_infos[index]
        lidar_path = info["lidar_path"][16:]
        lidar_sd_token = self.nusc.get("sample", info["token"])["data"]["LIDAR_TOP"]
        lidarseg_labels_filename = os.path.join(
            self.nusc.dataroot, self.nusc.get("panoptic", lidar_sd_token)["filename"]
        )

        points_label = np.load(lidarseg_labels_filename)["data"].reshape([-1, 1])
        sem_label = (points_label // 1000).astype(np.uint8)
        inst_label = (points_label % 1000).astype(np.uint8)
        sem_label = np.vectorize(self.learning_map.__getitem__)(sem_label)
        points = np.fromfile(os.path.join(self.data_path, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])

        data_tuple = (points[:, :3], sem_label.astype(np.uint8), inst_label.astype(np.uint8))
        if self.return_ref:
            data_tuple += (points[:, 3],)
        return data_tuple


@register_dataset
class SemKITTI_nusc_panop_incre(data.Dataset):
    def __init__(self, data_path, imageset="train", return_ref=False, label_mapping="nuscenes.yaml", nusc=None):
        self.return_ref = return_ref

        with open(imageset, "rb") as f:
            data = pickle.load(f)

        with open(label_mapping, "r") as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml["learning_map"]

        self.nusc_infos = data["infos"]
        self.data_path = data_path
        self.nusc = nusc

        self.pred_names = []
        pred_paths = os.path.join(self.data_path, "predictions", "predictions_incre158_train")
        # populate the label names
        seq_pred_names = [
            os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(pred_paths)) for f in fn if ".label" in f
        ]
        seq_pred_names.sort()
        self.pred_names.extend(seq_pred_names)

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.nusc_infos)

    def __getitem__(self, index):
        info = self.nusc_infos[index]
        lidar_path = info["lidar_path"][16:]
        lidar_sd_token = self.nusc.get("sample", info["token"])["data"]["LIDAR_TOP"]
        lidarseg_labels_filename = os.path.join(
            self.nusc.dataroot, self.nusc.get("panoptic", lidar_sd_token)["filename"]
        )

        points_label = np.load(lidarseg_labels_filename)["data"].reshape([-1, 1])
        sem_label = (points_label // 1000).astype(np.uint8)
        inst_label = (points_label % 1000).astype(np.uint8)
        sem_label = np.vectorize(self.learning_map.__getitem__)(sem_label)
        points = np.fromfile(os.path.join(self.data_path, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])

        distill_label_path = self.pred_names[index]
        distill_label = np.fromfile(distill_label_path, dtype=int32)
        distill_label = distill_label.reshape([-1, 1])

        data_tuple = (
            points[:, :3],
            sem_label.astype(np.uint8),
            inst_label.astype(np.uint8),
            distill_label.astype(np.uint8),
        )
        if self.return_ref:
            data_tuple += (points[:, 3],)
        return data_tuple


@register_dataset
class SemKITTI_nusc(data.Dataset):
    def __init__(self, data_path, imageset="train", return_ref=False, label_mapping="nuscenes.yaml", nusc=None):
        self.return_ref = return_ref

        with open(imageset, "rb") as f:
            data = pickle.load(f)

        with open(label_mapping, "r") as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml["learning_map"]

        self.nusc_infos = data["infos"]
        self.data_path = data_path
        self.nusc = nusc
        self.imageset = imageset

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.nusc_infos)

    def __getitem__(self, index):
        info = self.nusc_infos[index]
        lidar_path = info["lidar_path"][16:]
        points = np.fromfile(os.path.join(self.data_path, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])

        lidar_sd_token = self.nusc.get("sample", info["token"])["data"]["LIDAR_TOP"]

        if self.imageset.find("test") != -1:
            points_label = np.expand_dims(np.zeros_like(points[:, 0], dtype=int), axis=1)
        else:
            lidarseg_labels_filename = os.path.join(
                self.nusc.dataroot, self.nusc.get("lidarseg", lidar_sd_token)["filename"]
            )

            points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            points_label = np.vectorize(self.learning_map.__getitem__)(points_label)

        data_tuple = (points[:, :3], points_label.astype(np.uint8))
        if self.return_ref:
            data_tuple += (points[:, 3],)

        if self.imageset.find("test") != -1:
            return data_tuple, lidar_sd_token
        else:
            return data_tuple


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


def SemKITTI2train(label):
    if isinstance(label, list):
        return [SemKITTI2train_single(a) for a in label]
    else:
        return SemKITTI2train_single(label)


def SemKITTI2train_single(label):
    remove_ind = label == 0
    label -= 1
    label[remove_ind] = 255
    return label


from os.path import join


@register_dataset
class SemKITTI_sk_multiscan(data.Dataset):
    def __init__(self, data_path, imageset="train", return_ref=False, label_mapping="semantic-kitti-multiscan.yaml"):
        self.return_ref = return_ref
        with open(label_mapping, "r") as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml["learning_map"]
        self.imageset = imageset
        self.data_path = data_path
        if imageset == "train":
            split = semkittiyaml["split"]["train"]
        elif imageset == "val":
            split = semkittiyaml["split"]["valid"]
        elif imageset == "test":
            split = semkittiyaml["split"]["test"]
        else:
            raise Exception("Split must be train/val/test")

        multiscan = 2  # additional two frames are fused with target-frame. Hence, 3 point clouds in total
        self.multiscan = multiscan
        self.im_idx = []

        self.calibrations = []
        self.times = []
        self.poses = []

        self.load_calib_poses()

        for i_folder in split:
            self.im_idx += absoluteFilePaths("/".join([data_path, str(i_folder).zfill(2), "velodyne"]))

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.im_idx)

    def load_calib_poses(self):
        """
        load calib poses and times.
        """

        ###########
        # Load data
        ###########

        self.calibrations = []
        self.times = []
        self.poses = []

        for seq in range(0, 22):
            seq_folder = join(self.data_path, str(seq).zfill(2))

            # Read Calib
            self.calibrations.append(self.parse_calibration(join(seq_folder, "calib.txt")))

            # Read times
            self.times.append(np.loadtxt(join(seq_folder, "times.txt"), dtype=np.float32))

            # Read poses
            poses_f64 = self.parse_poses(join(seq_folder, "poses.txt"), self.calibrations[-1])
            self.poses.append([pose.astype(np.float32) for pose in poses_f64])

    def parse_calibration(self, filename):
        """read calibration file with given filename

        Returns
        -------
        dict
            Calibration matrices as 4x4 numpy arrays.
        """
        calib = {}

        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()

        return calib

    def parse_poses(self, filename, calibration):
        """read poses file with per-scan poses from given filename

        Returns
        -------
        list
            list of poses as 4x4 numpy arrays.
        """
        file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses

    def fuse_multi_scan(self, points, pose0, pose):
        # pose = poses[0][idx]

        hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
        # new_points = hpoints.dot(pose.T)
        new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)

        new_points = new_points[:, :3]
        new_coords = new_points - pose0[:3, 3]
        # new_coords = new_coords.dot(pose0[:3, :3])
        new_coords = np.sum(np.expand_dims(new_coords, 2) * pose0[:3, :3], axis=1)
        new_coords = np.hstack((new_coords, points[:, 3:]))

        return new_coords

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        origin_len = len(raw_data)
        if self.imageset == "test":
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(
                self.im_idx[index].replace("velodyne", "labels")[:-3] + "label", dtype=int32
            ).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            # annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        number_idx = int(self.im_idx[index][-10:-4])
        dir_idx = int(self.im_idx[index][-22:-20])

        pose0 = self.poses[dir_idx][number_idx]

        if number_idx - self.multiscan >= 0:
            for fuse_idx in range(self.multiscan):
                plus_idx = fuse_idx + 1

                pose = self.poses[dir_idx][number_idx - plus_idx]

                newpath2 = self.im_idx[index][:-10] + str(number_idx - plus_idx).zfill(6) + self.im_idx[index][-4:]
                raw_data2 = np.fromfile(newpath2, dtype=np.float32).reshape((-1, 4))

                if self.imageset == "test":
                    annotated_data2 = np.expand_dims(np.zeros_like(raw_data2[:, 0], dtype=int), axis=1)
                else:
                    annotated_data2 = np.fromfile(
                        newpath2.replace("velodyne", "labels")[:-3] + "label", dtype=int32
                    ).reshape((-1, 1))
                    annotated_data2 = annotated_data2 & 0xFFFF  # delete high 16 digits binary

                raw_data2 = self.fuse_multi_scan(raw_data2, pose0, pose)

                if len(raw_data2) != 0:
                    raw_data = np.concatenate((raw_data, raw_data2), 0)
                    annotated_data = np.concatenate((annotated_data, annotated_data2), 0)

        annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))

        if self.return_ref:
            data_tuple += (raw_data[:, 3], origin_len)  # origin_len is used to indicate the length of target-scan

        return data_tuple


# load Semantic KITTI class info


def get_SemKITTI_label_name(label_mapping):
    with open(label_mapping, "r") as stream:
        semkittiyaml = yaml.safe_load(stream)
    SemKITTI_label_name = dict()
    for i in sorted(list(semkittiyaml["learning_map"].keys()))[::-1]:
        SemKITTI_label_name[semkittiyaml["learning_map"][i]] = semkittiyaml["labels"][i]

    return SemKITTI_label_name


def get_nuScenes_label_name(label_mapping):
    with open(label_mapping, "r") as stream:
        nuScenesyaml = yaml.safe_load(stream)
    nuScenes_label_name = dict()
    for i in sorted(list(nuScenesyaml["learning_map"].keys()))[::-1]:
        val_ = nuScenesyaml["learning_map"][i]
        nuScenes_label_name[val_] = nuScenesyaml["labels_16"][val_]

    return nuScenes_label_name


def get_anovox_label_name(label_mapping):
    with open(label_mapping, "r") as stream:
        anovoxyaml = yaml.safe_load(stream)
    # anovox_label_name = dict()
    # for i in sorted(list(anovoxyaml["labels"].keys()))[::-1]:
    #     val_ = anovoxyaml["labels"][i]
    #     anovox_label_name[val_] = anovoxyaml["labels"][val_]
    label_dict = anovoxyaml["labels"]
    # return anovox_label_name
    return label_dict