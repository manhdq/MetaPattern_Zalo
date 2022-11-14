import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import torch
import zip_dataset
from transforms import VisualTransform, get_augmentation_transforms
import torchvision.transforms as transforms
import logging

import pdb


def parse_data_list(data_list_path):
    if not isinstance(data_list_path, str):
        csv = [pd.read_csv(data_path) for data_path in data_list_path]
        csv = pd.concat(csv).reset_index()
    else:
        csv = pd.read_csv(data_list_path)
    data_list = csv.get('fname')
    frame_indices = csv.get('frame_index')
    face_labels = csv.get('liveness_score')

    return data_list, face_labels, frame_indices


def get_dataset_from_list(data_list_path, dataset_cls, transform, num_frames=1000, root_dir=''):
    
    data_file_list, face_labels = parse_data_list(data_list_path)

    num_file = data_file_list.size
    dataset_list = []

    for i in range(num_file):
        face_label = int(face_labels.get(i)==0) # 0 means real face and non-zero represents spoof
        file_path = data_file_list.get(i)

        zip_path = root_dir + file_path
        if not os.path.exists(zip_path):
            logging.warning("Skip {} (not exists)".format(zip_path))
            continue
        else:
            dataset = dataset_cls(zip_path, face_label, transform=transform, num_frames=num_frames)
            if len(dataset) == 0:
                logging.warning("Skip {} (zero elements)".format(zip_path))
                continue
            else:
                dataset_list.append(dataset)
    final_dataset = torch.utils.data.ConcatDataset(dataset_list)
    return final_dataset