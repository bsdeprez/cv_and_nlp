import evaluate
import torch
import transformers
from tqdm import tqdm
from PIL import Image
from pathlib import Path


def evaluate_model(model, dataset, tokenizer, image_processor, device):
    metric = evaluate.load("rouge")
    model.eval()
    predictions, truths = [], []

    for example in tqdm(dataset):
        truths.append(example['caption'])
        predictions.append(predict(model, example['image_path'], tokenizer, image_processor, device))
    result = metric.compute(predictions=predictions, references=truths)
    return result


def predict(model, image_path, tokenizer, image_processor, device):
    """ Predict a caption for a given image.

    :param model: The VisionEncoderDecoder model used to predict a caption.
    :type model: transformers.PreTrainedModel
    :param image_path: The path of an image.
    :type image_path: Path
    :param tokenizer: The tokenizer which can decode the output of the model into human language.
    :type tokenizer: any
    :param image_processor: The image processor which can extract the input-ids from an image.
    :type image_processor: any
    :param device: The device on which the model runs.
    :type device: torch.device
    """
    with Image.open(image_path) as i_image:
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        pixel_values = image_processor([i_image], return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(device)
    model_kwargs = {"max_new_tokens": 20}
    output_ids = model.generate(pixel_values=pixel_values, **model_kwargs)
    predictions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return predictions[0].strip()
