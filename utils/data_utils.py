# +
from __future__ import print_function, absolute_import, division

import os
import sys
import cv2
import glob
import torch
import pathlib
import numpy as np
from torch import nn
from PIL import Image
import torch.nn.functional as F
from skimage.io import imread
import matplotlib.pyplot as plt
from collections import namedtuple
from typing import List, Callable, Tuple
from sklearn.externals._pilutil import bytescale

def re_normalize(inp: np.ndarray, low: int = 0, high: int = 255):
    """Normalize the data to a certain range. Default: [0-255]"""
    inp_out = bytescale(inp, low=low, high=high)
    return inp_out



# +
def get_labels():

    # a label and all meta information
    Label = namedtuple( 'Label' , [

        'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                        # We use them to uniquely name a class

        'id'          , # An integer ID that is associated with this label.
                        # The IDs are used to represent the label in ground truth images
                        # An ID of -1 means that this label does not have an ID and thus
                        # is ignored when creating ground truth images (e.g. license plate).
                        # Do not modify these IDs, since exactly these IDs are expected by the
                        # evaluation server.

        'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                        # ground truth images with train IDs, using the tools provided in the
                        # 'preparation' folder. However, make sure to validate or submit results
                        # to our evaluation server using the regular IDs above!
                        # For trainIds, multiple labels might have the same ID. Then, these labels
                        # are mapped to the same class in the ground truth images. For the inverse
                        # mapping, we use the label that is defined first in the list below.
                        # For example, mapping all void-type classes to the same ID in training,
                        # might make sense for some approaches.
                        # Max value is 255!

        'category'    , # The name of the category that this label belongs to

        'categoryId'  , # The ID of this category. Used to create ground truth images
                        # on category level.

        'hasInstances', # Whether this label distinguishes between single instances or not

        'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                        # during evaluations or not

        'color'       , # The color of this label
        ] )


    #--------------------------------------------------------------------------------
    # A list of all labels
    #--------------------------------------------------------------------------------

    # Please adapt the train IDs as appropriate for your approach.
    # Note that you might want to ignore labels with ID 255 during training.
    # Further note that the current train IDs are only a suggestion. You can use whatever you like.
    # Make sure to provide your results using the original IDs and not the training IDs.
    # Note that many IDs are ignored in evaluation and thus you never need to predict these!

    labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    ]
    
    return labels

labels = get_labels()
id2label =      { label.id      : label for label in labels }
trainid2label = { label.trainId : label for label in labels }

# -

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, config: dict, split="train", transform=None, labels=True):
        self.config = config
        self.split = split
        self.labels = labels
        self.crop_size = config['CROP_SIZE']
        self.base_size = config['BASE_SIZE']
        
        search_image_files = os.path.join(
            config['cityscapes_dir'],
            config['trainval_image_dir'], 
            split, '*', 
            config['trainval_input_pattern'])

        if labels:
            search_annot_files = os.path.join(
                config['cityscapes_dir'],
                config['trainval_label_dir'], 
                split, '*', 
                config['trainval_annot_pattern'])
        
        
        # root directory
        root = pathlib.Path.cwd() 

        input_path = str(root / search_image_files)
        if labels:
            target_path = str(root / search_annot_files)
        
        self.inputs = [pathlib.PurePath(file) for file in sorted(glob.glob(search_image_files))]
        if labels:
            self.targets = [pathlib.PurePath(file) for file in sorted(glob.glob(search_annot_files))]
        
        # print("{} images".format(len(self.inputs)))
        # print("{} masks".format(len(self.targets)))
        
        self.transform = transform
        self.inputs_dtype = torch.float32
        if labels:
            self.targets_dtype = torch.int64
        
        self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 
                                   1.0166, 0.9969, 0.9754, 1.0489,
                                   0.8786, 1.0023, 0.9539, 0.9843, 
                                   1.1116, 0.9037, 1.0865, 1.0955, 
                                   1.0865, 1.1529, 1.0507]).cuda()
       

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        
        # Select the sample
        input_ID = self.inputs[index]
        if self.labels:
            target_ID = self.targets[index]
        name = os.path.splitext(os.path.basename(input_ID))[0]

        # Load input and target
        if self.labels:
            x, y = imread(str(input_ID)), imread(str(target_ID))
        else:
            x = imread(str(input_ID))
        size = x.shape
            
        # Preprocessing
        if (self.transform is not None) and self.labels:
            x, y = self.transform(x, y)
        elif self.transform is not None:
            x = self.transform(x)

        # Typecasting
        if self.labels:
            x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)
            y = y.squeeze()
            return x, y, np.array(size), name
        else:
            x = torch.from_numpy(x).type(self.inputs_dtype)
            return x, np.array(size), name
       
    
    def inference(self, model, image):
        # assume input image is channels first
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        # convert to channels last for resizing
        # image = image.numpy()[0].transpose((1,2,0)).copy()
        image = image.cpu().numpy()[0].transpose((1,2,0)).copy()
        h, w = self.crop_size
        new_img = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        # convert to channels first for inference
        new_img = new_img.transpose((2, 0, 1))
        new_img = np.expand_dims(new_img, axis=0)
        pred = model(torch.from_numpy(new_img).cuda())
        # resize to base size
        pred = F.interpolate(input=pred, size=(ori_height, ori_width), mode='bilinear', align_corners=False)
        # pred = pred.numpy()
        pred = pred.cpu()
        return pred.exp()
    
    
    def label_to_rgb(self, seg):
        h = seg.shape[0]
        w = seg.shape[1]
        seg_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for key, val in trainid2label.items():
            indices = seg == key
            seg_rgb[indices.squeeze()] = val.color 
        return seg_rgb
    
    
    def save_pred(self, image, pred, sv_path, name):
        pred = np.asarray(np.argmax(pred.cpu(), axis=1), dtype=np.uint8)
        pred = self.label_to_rgb(pred[0])
        image = image.cpu()
        image = image[0].permute(1,2,0).numpy()
        image = re_normalize(image)

        blend = cv2.addWeighted(image, 0.8, pred, 0.8, 0)
        pil_blend = Image.fromarray(blend).convert("RGB")
        pil_blend.save(os.path.join(sv_path, name[0]+'.png'))


# +
def label_mapping(seg: np.ndarray, label_map: dict):
    seg = seg.astype(np.int32)
    temp = np.copy(seg)
    for key, val in label_map.items():
        seg[temp == key] = val.trainId
    return seg


def display(display_list):
    plt.figure(figsize=(15, 5), dpi=150)
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
