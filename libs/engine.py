import sys
import math
import torch
from pathlib import Path
from tqdm import tqdm
from data import RSTPReid
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel

def train_one_epoch(model, optimizer, tokenizer, data_loader, epoch, device=None, writer=None):
    """
    The implementation of this class is based on:
    - https://github.com/florisdf/maibi_cv/blob/main/3_detection/lib/detection/engine.py#L33
    """
    # Set the model to train mode.
    model.train()
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader)-1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_factor, total_iters=warmup_iters)
    
    for i in tqdm(range(len(data_loader)), total=len(data_loader), leave=False, desc='Train batch'):
        batch = next(iter(dataloader))
        _, pixel_values, captions = batch
        # The model expects the pixel values to have 4 dimensions, not 5.
        pixel_values = pixel_values.squeeze(0)
        # The model expects the input_ids to be tokenized.
        labels = tokenizer(captions, padding=True, return_tensors='pt').input_ids
        loss = model(pixel_values=pixel_values, labels=labels).loss
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
        



if __name__ == '__main__':
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    
    # Freeze some layers
    for param in model.decoder.parameters():
        param.requires_grad = False

    path = Path().resolve().parent / 'Data' / 'RSTPReid'
    assert path.exists()
    val_set = RSTPReid(path, 'val', image_processor)
    dataloader = DataLoader(val_set, batch_size=1, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_one_epoch(model, optimizer, tokenizer, dataloader, 1)