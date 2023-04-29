import sys
import math
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm

def train_one_epoch(model, optimizer, dataloader, epoch, tokenizer, device, writer=None):
    """ Train the model for one epoch.

    The implementation of this class is based on:
    - https://github.com/florisdf/maibi_cv/blob/main/3_detection/lib/detection/engine.py#L33
    
    :param model: The model to train.
    :type model: transformers.PreTrainedModel
    :param tokenizer: A tokenizer to transform the labels into model-understandable tokens
    :type tokenizer: any
    :param data_loader: The data_loader containing the training data
    :type data_loader: torch.utils.data.DataLoader
    :param writer: The summary writer to where the training metrics are written
    :type writer: torch.utils.tensorboard.SummaryWriter
    """
    model.train()
    lr_scheduler = None
    # Warmup
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iterations = min(1000, len(dataloader)-1)
        lr_scheduler = LinearLR(optimizer, start_factor=warmup_factor, total_iters=warmup_iterations)

    for batch_id, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Train batch"):
        pixel_values, captions = batch
        tokens = tokenizer(captions, return_tensors="pt", padding=True).input_ids
        loss = model(pixel_values=pixel_values.to(device), labels=tokens.to(device)).loss
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
            log_id = epoch * len(dataloader) + batch_id
            writer.add_scalar("Loss/train", loss_value, log_id)
            writer.add_scalar("Learning rate", optimizer.param_groups[0]["lr"], log_id)
