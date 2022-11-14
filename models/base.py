import os

import numpy as np
import pandas as pd
import torch
import importlib
from tqdm import tqdm

from utils.utils import AverageMeter

pd.set_option('display.max_columns', None)

import logging

from data.transforms import VisualTransform, get_augmentation_transforms
from data.data_loader import get_dataset_from_list
from test import metric_report_from_dict

from tensorboardX import SummaryWriter


class BaseTrainer(object):
    """
    Trainer encapsulates all the logic necessary for
    training the Recurrent Attention Model.

    All hyperparameters are provided by the user in the
    config file.
    """

    def __init__(self, config):
        """
        Construct a new Trainer instance.
        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator
        """

        self.config = config
        self.global_step = 1
        self.start_epoch = 1

        # Training control config
        self.epochs = self.config.TRAIN.EPOCHS
        self.batch_size = self.config.DATA.BATCH_SIZE

        self.counter = 0

        #  # Meanless at this version
        self.epochs = self.config.TRAIN.EPOCHS
        self.val_freq = config.TRAIN.VAL_FREQ
        #
        # # Network config

        self.val_metrcis = {
            'HTER@0.5': 1.0,
            'EER': 1.0,
            'MIN_HTER': 1.0,
            'AUC': 0
        }

        # Optimizer config
        self.momentum = self.config.TRAIN.MOMENTUM
        self.init_lr = self.config.TRAIN.INIT_LR
        self.lr_patience = self.config.TRAIN.LR_PATIENCE
        self.train_patience = self.config.TRAIN.PATIENCE

        self.network = self.get_network()
        self.get_loss_function()
        self.train_mode = True

    def set_train_mode(self, train_mode=True):
        self.train_mode = train_mode
        return self.train_mode

    def get_network(self):
        self.network = None

    def get_loss_function(self):
        self.loss = None

    def get_optimizer(self):
        self.optimizer = None

    def __get_dataset___(self):
        self.Dataset = importlib.import_module('data.zip_dataset').__dict__[self.config.DATA.DATASET]
        return self.Dataset

    def get_dataloader(self):
        config = self.config

        batch_size = config.DATA.BATCH_SIZE
        num_workers = 0 if config.DEBUG else config.DATA.NUM_WORKERS

        dataset_root_dir = config.DATA.ROOT_DIR
        dataset_subdir = config.DATA.SUB_DIR  # 'EXT0.2'
        dataset_dir = os.path.join(dataset_root_dir, dataset_subdir)

        test_data_transform = VisualTransform(config)
        Dataset = self.__get_dataset___()

        if not self.train_mode:
            assert config.DATA.TEST, "Please provide at least a data_list"
            test_dataset = get_dataset_from_list(config.DATA.TEST, Dataset, test_data_transform, num_frames=config.DATA.NUM_FRAMES, root_dir=dataset_dir)
            self.test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size, num_workers=num_workers,
                                                                shuffle=False, drop_last=True)

        else:
            assert config.DATA.TRAIN, "CONFIG.DATA.TRAIN should be provided"
            aug_transform = get_augmentation_transforms(config)
            train_data_transform = VisualTransform(config, aug_transform)
            train_dataset = get_dataset_from_list(config.DATA.TRAIN, Dataset, train_data_transform, num_frames=config.DATA.NUM_FRAMES,
                                                  root_dir=dataset_dir)
            self.train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size, num_workers=num_workers,
                                                                 shuffle=True, pin_memory=True, drop_last=True)

            assert config.DATA.VAL, "CONFIG.DATA.VAL should be provided"
            val_dataset = get_dataset_from_list(config.DATA.VAL, Dataset, test_data_transform, num_frames=config.DATA.NUM_FRAMES, root_dir=dataset_dir)
            self.val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size, num_workers=num_workers,
                                                               shuffle=False, pin_memory=True, drop_last=True)

    def init_weight(self):
        pass

    def validate(self, epoch, val_data_loader):
        val_results = self.test(val_data_loader)
        val_loss = val_results['avg_loss']
        scores_gt_dict = val_results['scores_gt']
        scores_pred_dict = val_results['scores_pred']
        # log to tensorboard
        if self.tensorboard:
            self.tensorboard.add_scalar('loss/val_total', val_loss, self.global_step)

        frame_metric_dict, video_metric_dict = metric_report_from_dict(scores_pred_dict, scores_gt_dict, 0.5)
        df_frame = pd.DataFrame(frame_metric_dict, index=[0])
        df_video = pd.DataFrame(video_metric_dict, index=[0])

        logging.info("Frame level metrics: \n" + str(df_frame))
        logging.info("Video level metrics: \n" + str(df_video))

        return frame_metric_dict

    def test(self, test_data_loader):
        avg_test_loss = AverageMeter()
        scores_pred_dict = {}
        face_label_gt_dict = {}
        self.network.eval()
        with torch.no_grad():
            for data in tqdm(test_data_loader):
                network_input, target, video_ids = data[1], data[2], data[3]

                output_prob = self.inference(network_input.cuda())
                test_loss = self._total_loss_caculation(output_prob, target)
                pred_score = self._get_score_from_prob(output_prob)
                avg_test_loss.update(test_loss.item(), network_input.size()[0])

                gt_dict, pred_dict = self._collect_scores_from_loader(scores_pred_dict, face_label_gt_dict,
                                                                      target['face_label'].numpy(), pred_score,
                                                                      video_ids
                                                                      )

        test_results = {
            'scores_gt': gt_dict,
            'scores_pred': pred_dict,
            'avg_loss': avg_test_loss.avg,
        }
        return test_results

    def _collect_scores_from_loader(self, gt_dict, pred_dict, ground_truths, pred_scores, video_ids):
        batch_size = ground_truths.shape[0]

        for i in range(batch_size):
            video_name = video_ids[i]
            if video_name not in pred_dict.keys():
                pred_dict[video_name] = list()
            if video_name not in gt_dict.keys():
                gt_dict[video_name] = list()

            pred_dict[video_name] = np.append(pred_dict[video_name], pred_scores[i])
            gt_dict[video_name] = np.append(gt_dict[video_name], ground_truths[i])

        return gt_dict, pred_dict

    def inference(self, *args, **kargs):
        """
            Input images
            Output prob and scores
        """
        output_prob = self.network(*args, **kargs)  # By default: a binary classifier network
        return output_prob

    def _total_loss_caculation(self, output_prob, target):
        face_label = target['face_label'].cuda()
        return self.loss(output_prob, face_label)

    def _get_score_from_prob(self, output_prob):
        output_scores = torch.softmax(output_prob, 1)
        output_scores = output_scores.cpu().numpy()[:, 1]
        return output_scores


    def load_batch_data(self):
        pass
