import sys
import math
import torch
from tqdm import tqdm

def train_one_epoch(model, optimizer, tokenizer, data_loader, epoch, device, writer=None):
    """
    The implementation of this class is based on:
    - https://github.com/florisdf/maibi_cv/blob/main/3_detection/lib/detection/engine.py#L33
    
    :param model: The model to train.
    :type model: transformers.PreTrainedModel
    :param tokenizer: A tokenizer to transform the labels into model-understandable tokens
    :type tokenizer: any
    :param data_loader: The data_loader containing the training data
    :type data_loader: torch.utils.data.DataLoader
    :param device: The device on which the model is stored. This should probably be CUDA
    :type device: torch.device
    :param writer: The summary writer to where the training metrics are written
    :type writer: torch.utils.tensorboard.SummaryWriter
    """
    # Set the model to train mode.
    model.train()
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader)-1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_factor, total_iters=warmup_iters)
    
    for i in tqdm(range(len(data_loader)), total=len(data_loader), leave=False, desc='Train batch'):
        batch = next(iter(data_loader))
        _, pixel_values, captions = batch
        # The model expects the input_ids to be tokenized.
        labels = tokenizer(captions, padding=True, return_tensors='pt').input_ids
        # The model expects the pixel values to have 4 dimensions, not 5.
        pixel_values = pixel_values.squeeze(0)
        loss = model(pixel_values=pixel_values.to(device), labels=labels.to(device)).loss
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training.")
            sys.exit(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Let the scheduler take a step.
        if lr_scheduler is not None:
            lr_scheduler.step()
        # Let the writer write away data.
        if writer is not None:
            log_id = epoch * len(data_loader) + i
            writer.add_scalar("Loss/train", loss_value, log_id)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], log_id)
