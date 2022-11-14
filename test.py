from utils.metrics import get_eer_stats, get_hter_at_thr, get_min_hter
import os
import logging
import numpy as np
import pandas as pd
import importlib
import torch
from config.defaults import get_cfg_defaults
pd.set_option('display.max_columns', None)


import sklearn.metrics as metrics

torch.backends.cudnn.benchmark = True


import argparse

def default_argument_parser():
    """
        Create arg parser
    """
    parser = argparse.ArgumentParser("Args parser for training")
    parser.add_argument("--config", default="", metavar="FILE", required=True, help="path to config file")
    parser.add_argument("--trainer", default="", required=True, help="trainer")
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    #
    # parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
             "See config references at "
             "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def setup(args):
    """
        Perform some basic common setups at the beginning of a job, including:
    """
    cfg = get_cfg_defaults()
    trainer_lib = importlib.import_module('models.' + args.trainer)

    if 'custom_cfg' in trainer_lib.__all__:
        cfg.merge_from_other_cfg(trainer_lib.custom_cfg)

    if args.config:
        cfg.merge_from_file(args.config)

    if args.opts:
        cfg.merge_from_list(args.opts)

    assert cfg.TEST.CKPT, "A checkpoint should be provided"

    cfg.DEBUG = args.debug
    cfg.freeze()

    testing_output_dir = os.path.join(cfg.OUTPUT_DIR, 'test', cfg.TEST.TAG)

    if not os.path.exists(testing_output_dir):
        os.makedirs(testing_output_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(filename)s-%(funcName)s-%(lineno)d:%(message)s",
                        datefmt='%a-%d %b %Y %H:%M:%S',
                        handlers=[logging.FileHandler(os.path.join(testing_output_dir, 'test.log'), 'a', 'utf-8'),
                                  logging.StreamHandler()]
                        )

    config_saved_path = os.path.join(testing_output_dir, "test_config.yaml")
    logging.info("Config:\n"+str(cfg))

    with open(config_saved_path, "w") as f:
        f.write(cfg.dump())
        logging.info("Full config saved to {}".format(config_saved_path))

    return trainer_lib, cfg


def metric_report_from_dict(scores_pred_dict, scores_gt_dict, thr):
    frame_pred_list = list()
    frame_label_list = list()
    video_pred_list = list()
    video_label_list = list()

    for key in scores_pred_dict.keys():
        num_frames = len(scores_pred_dict[key])
        avg_single_video_pred = sum(scores_pred_dict[key]) /num_frames
        avg_single_video_label = sum(scores_gt_dict[key]) / num_frames

        video_pred_list = np.append(video_pred_list, avg_single_video_pred)
        video_label_list = np.append(video_label_list, avg_single_video_label)

        frame_pred_list = np.append(frame_pred_list, scores_pred_dict[key])
        frame_label_list = np.append(frame_label_list, scores_gt_dict[key])

    frame_metrics = metric_report(frame_pred_list, frame_label_list, thr)
    video_metrics = metric_report(video_pred_list, video_label_list, thr)
    return frame_metrics, video_metrics


def metric_report(scores_pred, scores_gt, thr):
    
    fpr, tpr, threshold = metrics.roc_curve(scores_gt, scores_pred)
    auc = metrics.auc(fpr, tpr)

    eer, eer_thr = get_eer_stats(scores_pred, scores_gt)
    hter, far, frr = get_hter_at_thr(scores_pred, scores_gt, thr)
    hter05, far05, frr05 = get_hter_at_thr(scores_pred, scores_gt, 0.5)
    min_hter, hter_thr, far_at_thr, frr_at_thr = get_min_hter(scores_pred, scores_gt)

    metric_dict = {
        'AUC':auc,
        'EER': eer,
        'EER_THR': eer_thr,
        'HTER@THR': hter,
        'FAR@THR': far,
        'FRR@THR': frr,
        'THR': thr,
        'HTER@0.5': hter05,
        'FAR@0.5': far05,
        'FRR@0.5': frr05,
        'MIN_HTER': min_hter,
        'MIN_HTER_THR': hter_thr,
        'MIN_FAR_THR': far_at_thr,
        'MIN_FRR_THR': frr_at_thr,

    }
    return metric_dict