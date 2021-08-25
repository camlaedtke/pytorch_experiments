import numpy as np
import albumentations
from utils.data_utils import get_labels, label_mapping
from utils.transformations import (normalize, ComposeSingle, ComposeDouble, re_normalize, 
                                   FunctionWrapperSingle, FunctionWrapperDouble, 
                                   AlbuSeg2d, multi_scale_aug, random_brightness)


labels = get_labels()
id2label =      { label.id      : label for label in labels }


def get_transforms_training(config):
    transforms_training = ComposeDouble([
        FunctionWrapperDouble(
            multi_scale_aug, 
            scale_factor=16, 
            crop_size=config['CROP_SIZE'], 
            base_size=config['BASE_SIZE'][1], 
            ignore_label=config['IGNORE_LABEL'], 
            both=True),
        AlbuSeg2d(albumentations.HorizontalFlip(p=0.5)),
        FunctionWrapperDouble(label_mapping, label_map=id2label, input=False, target=True),
        FunctionWrapperDouble(random_brightness, input=True, target=False),
        FunctionWrapperDouble(
            normalize, 
            mean=config['DATASET_MEAN'], 
            std=config['DATASET_STD'],
            input=True, 
            target=False),
        FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
    ])
    return transforms_training


def get_transforms_validation(config):

    transforms_validation = ComposeDouble([
        FunctionWrapperDouble(multi_scale_aug, valid=True, crop_size=config['CROP_SIZE'], both=True),
        FunctionWrapperDouble(label_mapping, label_map=id2label, input=False, target=True),
        FunctionWrapperDouble(
            normalize, 
            mean=config['DATASET_MEAN'], 
            std=config['DATASET_STD'],
            input=True, 
            target=False),
        FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
    ])
    return transforms_validation


def get_transforms_evaluation(config):

    transforms_evaluation = ComposeDouble([
        FunctionWrapperDouble(label_mapping, label_map=id2label, input=False, target=True),
        FunctionWrapperDouble(
            normalize, 
            mean=config['DATASET_MEAN'], 
            std=config['DATASET_STD'],
            input=True, 
            target=False),
        FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
    ])
    return transforms_evaluation


def get_transforms_video(config):
    transforms_video = ComposeSingle([
        FunctionWrapperSingle(
            normalize, 
            mean=config['DATASET_MEAN'], 
            std=config['DATASET_STD']),
        FunctionWrapperSingle(np.moveaxis, source=-1, destination=0),
    ])
    return transforms_video