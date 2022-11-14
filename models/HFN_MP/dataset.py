import logging
import os
import zipfile
import functools

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T

from data.data_loader import parse_data_list
from data.transforms import VisualTransform
from data.transforms import get_augmentation_transforms
from data.zip_dataset import ZipDataset as _ZipDataset
from data.video_dataset import VideoDataset as _VideoDataset


class ZipDataset(_ZipDataset):
    def __init__(self, zip_file_path, face_label, frame_index, transform=None, num_frames=1, train=True):
        _ZipDataset.__init__(self, zip_file_path, face_label, transform=transform, num_frames=num_frames)

    def __getitem__(self, index):
        index, tensor, target, self.zip_file_path = _ZipDataset.__getitem__(self, index)
        target['depth'] =  self.face_label * torch.ones([1, 32, 32, ], dtype=torch.float32)
        return index, tensor, target, self.zip_file_path


class ZipDatasetPixelMC(ZipDataset):
    """
        ZipDataset with Pixel-wise target (Multi channel)
    """

    def __init__(self, zip_file_path, face_label, frame_index, transform, num_frames=1000, config=None):
        super(ZipDatasetPixelMC, self).__init__(zip_file_path, face_label, frame_index, transform, num_frames)

        self.config = config

    def __getitem__(self, index):
    
        im = self.__read_image_from_zip__(index)
        pixel_maps_size = 32

        pixel_maps_size = [32, 16, 8, 4, 2]
        pixel_maps = []
        for s in pixel_maps_size:
            pixel_maps.append(self.face_label * torch.ones([s, s]))

        channels = channel_list(im, self.config)
        channels = [ self.transform(im) for im in channels ]

        channels = torch.cat(channels, 0)
        target = {
            'face_label': self.face_label,
            'pixel_maps': pixel_maps
        }
        return index, channels, target, self.zip_file_path


class VideoDataset(_VideoDataset):
    def __init__(self, zip_file_path, face_label, frame_index, transform=None, num_frames=1, train=True):
        _VideoDataset.__init__(self, zip_file_path, face_label, frame_index, transform=transform, num_frames=num_frames)

    def __getitem__(self, index):
        index, tensor, target, self.vid_file_path = _VideoDataset.__getitem__(self, index)
        target['depth'] =  self.face_label * torch.ones([1, 32, 32, ], dtype=torch.float32)
        return index, tensor, target, self.vid_file_path


class VideoDatasetPixelMC(VideoDataset):
    """
        VideoDataset with Pixel-wise target (Multi channel)
    """

    def __init__(self, zip_file_path, face_label, frame_index, transform, num_frames=1000, config=None):
        super(VideoDatasetPixelMC, self).__init__(zip_file_path, face_label, frame_index, transform, num_frames)

        self.config = config

    def __getitem__(self, index):
        cap = cv2.VideoCapture(self.vid_file_path)
        cap.set(1, self.frame_index)
        ret, im = cap.read()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        pixel_maps_size = [32, 16, 8, 4, 2]
        pixel_maps = []
        for s in pixel_maps_size:
            pixel_maps.append(self.face_label * torch.ones([s, s]))
            
        channels = channel_list(im, self.config)
        channels = [self.transform(im) for im in channels]

        channels = torch.cat(channels, 0)
        target = {
            'face_label':  self.face_label,
            'pixel_maps': pixel_maps,
        }
        return index, channels, target, self.vid_file_path


def channel_list(bgr_im, config=None):
    channel_list = []
    if config.MODEL.CHANNELS.RGB:
        channel_list.append(bgr_im)
    if config.MODEL.CHANNELS.HSV:
        img_hsv = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2HSV)
        channel_list.append(img_hsv)

    if config.MODEL.CHANNELS.YCRCB:
        img_ycrcb = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2YCrCb)

        channel_list.append(img_ycrcb)

    if config.MODEL.CHANNELS.LAB:
        lab_image = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2LAB)
        channel_list.append(lab_image)

    if config.MODEL.CHANNELS.YUV:
        yuv_image = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2YUV)
        channel_list.append(yuv_image)

    if config.MODEL.CHANNELS.XYZ:
        xyz_image = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2XYZ)
        channel_list.append(xyz_image)

    return channel_list


def get_dataset_from_list(data_list_path, dataset_cls, train=True, transform=None, num_frames=1, root_dir='', ):
    # TODO: hard code it right now
    data_file_list, face_labels, frame_indices = parse_data_list(data_list_path)

    num_file = data_file_list.size
    dataset_list = []

    for i in range(num_file):
        face_label = int(face_labels.get(i)==0) # 0 means real face and non-zero represents spoof
        file_path = data_file_list.get(i)
        frame_index = frame_indices.get(i)
        # zip_path = os.path.join(file_path, root_path)
        file_path = root_dir + file_path
        if not os.path.exists(file_path):
            logging.warning("Skip {} (not exists)".format(file_path))
            continue
        else:
            dataset = dataset_cls(file_path, face_label, frame_index, transform, num_frames=num_frames)
            if len(dataset) == 0:
                logging.warning("Skip {} (zero elements)".format(file_path))
                continue
            else:
                dataset_list.append(dataset)
    final_dataset = torch.utils.data.ConcatDataset(dataset_list)

    return final_dataset


def get_zip_data_loader(config):
    

    aug_transform = get_augmentation_transforms(config)
    train_data_transform = VisualTransform(config, aug_transform)
    test_data_transform = VisualTransform(config)
    root_dir = os.path.join(config.DATA.ROOT_DIR, config.DATA.SUB_DIR)
    Dataset = functools.partial(ZipDatasetPixelMC, config=config)

    batch_size = config.DATA.BATCH_SIZE
    if config.DATA.TEST:
        tgt_test_data = get_dataset_from_list(config.DATA.TEST, Dataset, transform=test_data_transform,
                                              root_dir=root_dir, num_frames=config.TEST.NUM_FRAMES)


        tgt_dataloader = DataLoader(tgt_test_data, batch_size=batch_size, shuffle=False, drop_last=True)
        return tgt_dataloader

    src1_train_data_fake_list = config.DATA.TRAIN_SRC_FAKE_1
    src1_train_data_real_list = config.DATA.TRAIN_SRC_REAL_1

    src2_train_data_fake_list = config.DATA.TRAIN_SRC_FAKE_2
    src2_train_data_real_list = config.DATA.TRAIN_SRC_REAL_2

    src3_train_data_fake_list = config.DATA.TRAIN_SRC_FAKE_3
    src3_train_data_real_list = config.DATA.TRAIN_SRC_REAL_3

    num_frames = config.TRAIN.NUM_FRAMES
    src1_train_data_fake = get_dataset_from_list(src1_train_data_fake_list, Dataset, transform=train_data_transform, num_frames=config.TRAIN.NUM_FRAMES, root_dir=root_dir)
    src1_train_data_real = get_dataset_from_list(src1_train_data_real_list, Dataset, transform=train_data_transform, num_frames=config.TRAIN.NUM_FRAMES, root_dir=root_dir)
    src2_train_data_fake = get_dataset_from_list(src2_train_data_fake_list, Dataset, transform=train_data_transform, num_frames=config.TRAIN.NUM_FRAMES, root_dir=root_dir)
    src2_train_data_real = get_dataset_from_list(src2_train_data_real_list, Dataset, transform=train_data_transform, num_frames=config.TRAIN.NUM_FRAMES, root_dir=root_dir)
    src3_train_data_fake = get_dataset_from_list(src3_train_data_fake_list, Dataset, transform=train_data_transform, num_frames=config.TRAIN.NUM_FRAMES, root_dir=root_dir)
    src3_train_data_real = get_dataset_from_list(src3_train_data_real_list, Dataset, transform=train_data_transform, num_frames=config.TRAIN.NUM_FRAMES, root_dir=root_dir)

    tgt_test_data_list = config.DATA.TARGET_DATA
    tgt_test_data = get_dataset_from_list(tgt_test_data_list, Dataset, transform=test_data_transform, root_dir=root_dir, num_frames = config.TEST.NUM_FRAMES)

    print('Load Target Data')

    src1_train_dataloader_fake = DataLoader(src1_train_data_fake,
                                            batch_size=batch_size, shuffle=True)
    src1_train_dataloader_real = DataLoader(src1_train_data_real,
                                            batch_size=batch_size, shuffle=True)
    src2_train_dataloader_fake = DataLoader(src2_train_data_fake,
                                            batch_size=batch_size, shuffle=True)
    src2_train_dataloader_real = DataLoader(src2_train_data_real,
                                            batch_size=batch_size, shuffle=True)
    src3_train_dataloader_fake = DataLoader(src3_train_data_fake,
                                            batch_size=batch_size, shuffle=True)
    src3_train_dataloader_real = DataLoader(src3_train_data_real,
                                            batch_size=batch_size, shuffle=True)
    tgt_dataloader = DataLoader(tgt_test_data, batch_size=batch_size, shuffle=False, drop_last=True)
    return src1_train_dataloader_fake, src1_train_dataloader_real, \
           src2_train_dataloader_fake, src2_train_dataloader_real, \
           src3_train_dataloader_fake, src3_train_dataloader_real, \
           tgt_dataloader


def get_video_data_loader(config):


    aug_transform = get_augmentation_transforms(config)
    train_data_transform = VisualTransform(config, aug_transform)
    test_data_transform = VisualTransform(config)
    root_dir = os.path.join(config.DATA.ROOT_DIR, config.DATA.SUB_DIR)
    Dataset = functools.partial(VideoDatasetPixelMC, config=config)

    batch_size = config.DATA.BATCH_SIZE
    if config.DATA.TEST:
        tgt_test_data = get_dataset_from_list(config.DATA.TEST, Dataset, transform=test_data_transform,
                                              root_dir=root_dir, num_frames=config.TEST.NUM_FRAMES)


        tgt_dataloader = DataLoader(tgt_test_data, batch_size=batch_size, shuffle=False, drop_last=True)
        return tgt_dataloader

    src1_train_data_fake_list = config.DATA.TRAIN_SRC_FAKE_1
    src1_train_data_real_list = config.DATA.TRAIN_SRC_REAL_1

    src2_train_data_fake_list = config.DATA.TRAIN_SRC_FAKE_2
    src2_train_data_real_list = config.DATA.TRAIN_SRC_REAL_2

    src3_train_data_fake_list = config.DATA.TRAIN_SRC_FAKE_3
    src3_train_data_real_list = config.DATA.TRAIN_SRC_REAL_3

    src4_train_data_fake_list = config.DATA.TRAIN_SRC_FAKE_4
    src4_train_data_real_list = config.DATA.TRAIN_SRC_REAL_4

    num_frames = config.TRAIN.NUM_FRAMES
    src1_train_data_fake = get_dataset_from_list(src1_train_data_fake_list, Dataset, transform=train_data_transform, num_frames=config.TRAIN.NUM_FRAMES, root_dir=root_dir)
    src1_train_data_real = get_dataset_from_list(src1_train_data_real_list, Dataset, transform=train_data_transform, num_frames=config.TRAIN.NUM_FRAMES, root_dir=root_dir)
    src2_train_data_fake = get_dataset_from_list(src2_train_data_fake_list, Dataset, transform=train_data_transform, num_frames=config.TRAIN.NUM_FRAMES, root_dir=root_dir)
    src2_train_data_real = get_dataset_from_list(src2_train_data_real_list, Dataset, transform=train_data_transform, num_frames=config.TRAIN.NUM_FRAMES, root_dir=root_dir)
    src3_train_data_fake = get_dataset_from_list(src3_train_data_fake_list, Dataset, transform=train_data_transform, num_frames=config.TRAIN.NUM_FRAMES, root_dir=root_dir)
    src3_train_data_real = get_dataset_from_list(src3_train_data_real_list, Dataset, transform=train_data_transform, num_frames=config.TRAIN.NUM_FRAMES, root_dir=root_dir)
    src4_train_data_fake = get_dataset_from_list(src4_train_data_fake_list, Dataset, transform=train_data_transform, num_frames=config.TRAIN.NUM_FRAMES, root_dir=root_dir)
    src4_train_data_real = get_dataset_from_list(src4_train_data_real_list, Dataset, transform=train_data_transform, num_frames=config.TRAIN.NUM_FRAMES, root_dir=root_dir)

    tgt_test_data_list = config.DATA.TARGET_DATA
    tgt_test_data = get_dataset_from_list(tgt_test_data_list, Dataset, transform=test_data_transform, root_dir=root_dir, num_frames = config.TEST.NUM_FRAMES)

    
    print('Load Target Data')

    src1_train_dataloader_fake = DataLoader(src1_train_data_fake,
                                            batch_size=batch_size, shuffle=True)
    src1_train_dataloader_real = DataLoader(src1_train_data_real,
                                            batch_size=batch_size, shuffle=True)
    src2_train_dataloader_fake = DataLoader(src2_train_data_fake,
                                            batch_size=batch_size, shuffle=True)
    src2_train_dataloader_real = DataLoader(src2_train_data_real,
                                            batch_size=batch_size, shuffle=True)
    src3_train_dataloader_fake = DataLoader(src3_train_data_fake,
                                            batch_size=batch_size, shuffle=True)
    src3_train_dataloader_real = DataLoader(src3_train_data_real,
                                            batch_size=batch_size, shuffle=True)
    src4_train_dataloader_fake = DataLoader(src4_train_data_fake,
                                            batch_size=batch_size, shuffle=True)
    src4_train_dataloader_real = DataLoader(src4_train_data_real,
                                            batch_size=batch_size, shuffle=True)
    tgt_dataloader = DataLoader(tgt_test_data, batch_size=batch_size, shuffle=False, drop_last=True)
    return src1_train_dataloader_fake, src1_train_dataloader_real, \
           src2_train_dataloader_fake, src2_train_dataloader_real, \
           src3_train_dataloader_fake, src3_train_dataloader_real, \
           src4_train_dataloader_fake, src4_train_dataloader_real, \
           tgt_dataloader