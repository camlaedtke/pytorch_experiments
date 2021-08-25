import os
import sys
import time
import torch
import pathlib
import logging
import numpy as np
from torch import nn
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils.modelsummary import get_model_summary
from utils.train_utils import AverageMeter, get_confusion_matrix, adjust_learning_rate, create_logger


def train(config, dataloader, model, loss_fn, optimizer, epoch, scaler, writer_dict):
    model.train()
    
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    steps_per_epoch = len(dataloader) 
    steps_tot = epoch*steps_per_epoch
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    
    for step, batch in enumerate(dataloader):
        X, y, _, _ = batch
        X, y = X.cuda(), y.long().cuda()
        
        # Compute prediction and loss
        with torch.cuda.amp.autocast():
            pred = model(X)
            losses = loss_fn(pred, y)
        loss = losses.mean()
        
        # Backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()
        
        # update average loss
        ave_loss.update(loss.item())
        lr = adjust_learning_rate(
            optimizer, config['BASE_LR'], config['END_LR'], step+steps_tot, config['DECAY_STEPS'])
        optimizer.param_groups[0]['lr'] = lr
        
        msg = '\r Training   --- Iter:[{}/{}], Loss: {:.5f}, lr: {:.7f}, Time: {:.2f}'.format(
            step, steps_per_epoch, ave_loss.average(), optimizer.param_groups[0]['lr'], batch_time.average())
        #logging.info(msg)
        
    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1


def validate(config, dataloader, model, loss_fn, writer_dict):
    model.eval()
    
    ave_loss = AverageMeter()
    iter_steps = len(dataloader.dataset) // config['BATCH_SIZE'] 
    confusion_matrix = np.zeros((config['NUM_CLASSES'], config['NUM_CLASSES'], 1))
    
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            x, y, _, _ = batch
            size = y.size()
            X, y = X.cuda(), y.long().cuda()
            
            pred = model(X)
            losses = loss_fn(pred, y)
            loss = losses.mean()   
            
            if not isinstance(pred, (list, tuple)):
                pred = [pred]    
            for i, x in enumerate(pred):
                confusion_matrix[..., i] += get_confusion_matrix(
                    y, x, size, config['NUM_CLASSES'], config['IGNORE_LABEL'])
            ave_loss.update(loss.item())
            
    pos = confusion_matrix[..., 0].sum(1)
    res = confusion_matrix[..., 0].sum(0)
    tp = np.diag(confusion_matrix[..., 0])
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    for key, val in trainid2label.items():
        if key != config['IGNORE_LABEL'] and key != -1:
            writer.add_scalar('valid_mIoU_{}'.format(val.name), IoU_array[key], global_steps)    
    writer_dict['valid_global_steps'] = global_steps + 1
        
    return ave_loss.average(), mean_IoU, IoU_array


def testval(config, testloader, model, sv_dir='', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros((config['NUM_CLASSES'], config['NUM_CLASSES']))
    
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name, *border_padding = batch
            size = label.size()
            image, label = image.cuda(), label.long().cuda()
            pred = testloader.dataset.inference(model, image)

            confusion_matrix += get_confusion_matrix(
                label, pred, size, config['NUM_CLASSES'], config['IGNORE_LABEL'])

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))
                
            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                testloader.dataset.save_pred(image, pred, sv_path, name)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc


def testvideo(config, testloader, model, sv_dir='', sv_pred=False):
    model.eval()

    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, _, name, *border_padding = batch
            size = image.size()
            image = image.cuda()
            pred = testloader.dataset.inference(model, image)
                
            if sv_pred:
                sv_path = os.path.join(sv_dir, 'video_frames')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                testloader.dataset.save_pred(image, pred, sv_path, name)

    print("done!")
