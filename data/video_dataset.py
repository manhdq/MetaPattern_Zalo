import cv2
import numpy as np
import torch
import zipfile
from torch.utils.data import Dataset


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, vid_file_path, face_label, frame_index, transform, num_frames=1000):
        self.vid_file_path = vid_file_path
        self.transform = transform
        self.decode_flag = cv2.IMREAD_UNCHANGED
        self.face_label = face_label
        self.frame_index = frame_index

        # self.image_list_in_video = []

        self.len = 1

    def __getitem__(self, index):
        cap = cv2.VideoCapture(self.vid_file_path)
        cap.set(1, self.frame_index)
        ret, im = cap.read()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        tensor = self.transform(im)
        tensor = tensor.to(torch.float)
        target = {
            'face_label':  self.face_label
        }
        return index, tensor, target, self.vid_file_path

    def __len__(self):
        return self.len