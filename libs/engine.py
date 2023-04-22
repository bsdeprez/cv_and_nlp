import math
import torch
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

def train_one_epoch(model, optimizer, data_loader, devive, epoch, writer=None):
    model.train()
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader)-1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_factor, total_iters=warmup_iters)

    for batch_idx, (images, targets) in tqdm(enumerate(data_loader), total=len(data_loader), leave=False, desc='Train batch'):
        # images = [image.to(device) for image in images]
        print(targets)
        print("-"*80)
