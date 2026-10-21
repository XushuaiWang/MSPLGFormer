import logging
import os
import time
from copy import deepcopy

import torch

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def save_model(path, epoch, model, optimizer=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    checkpoint = {'epoch': epoch, 'state_dict': state_dict}
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    torch.save(checkpoint, path)


def load_model(model, model_path, optimizer=None, resume=False, change_output=False, lr=None, lr_step=None, lr_factor=None):
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = deepcopy(checkpoint['state_dict'])
    if change_output:
        for key in list(state_dict.keys()):
            if key.startswith('output_layer') or key.startswith('out'):
                state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    start_epoch = checkpoint.get('epoch', 0)
    print(f'Loaded model from {model_path}. Epoch: {start_epoch}')

    if optimizer is not None and resume and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if optimizer is not None:
        return model, optimizer, start_epoch
    return model


def readable_time(time_difference):
    hours = time_difference // 3600
    minutes = (time_difference // 60) % 60
    seconds = time_difference % 60
    return hours, minutes, seconds
