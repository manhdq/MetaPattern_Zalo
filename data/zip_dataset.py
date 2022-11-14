import cv2
import numpy as np
import torch
import zipfile
from torch.utils.data import Dataset



class ZipDataset(torch.utils.data.Dataset):
    def __init__(self, zip_file_path, face_label, frame_index, transform, num_frames=1000):
        self.zip_file_path = zip_file_path
        self.transform = transform
        self.decode_flag = cv2.IMREAD_UNCHANGED
        self.face_label = face_label

        self.image_list_in_zip = []
        with zipfile.ZipFile(self.zip_file_path, "r") as zip:
            lst = zip.namelist()
            exts = ['png', 'jpg']
            for ext in exts:
                self.image_list_in_zip += list(filter(lambda x: x.lower().endswith(ext), lst))

        if len(self.image_list_in_zip) > num_frames:
            sample_indices = np.linspace(0, len(self.image_list_in_zip)-1, num=num_frames, dtype=int)
            self.image_list_in_zip = [self.image_list_in_zip[id] for id in sample_indices]

        self.len = len(self.image_list_in_zip)

    def __read_image_from_zip__(self, index):
        image_name_in_zip = self.image_list_in_zip[index]
        with zipfile.ZipFile(self.zip_file_path, "r") as zip:
            bytes_ = zip.read(image_name_in_zip)
            bytes_ = np.frombuffer(bytes_, dtype=np.uint8)
            im = cv2.imdecode(bytes_, self.decode_flag)  # cv2 image
            return im

    def __getitem__(self, index):
        im = self.__read_image_from_zip__(index) # cv2 image, format [H, W, C], BGR
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # im = im.transpose((2,0,1))
        tensor = self.transform(im)
        tensor = tensor.to(torch.float)
        target = {
            'face_label':  self.face_label
        }
        return index, tensor, target, self.zip_file_path

    def __len__(self):
        return self.len