import evaluate
import torch
import transformers
from tqdm import tqdm
from PIL import Image
from pathlib import Path


def evaluate_model(model, dataset, tokenizer, image_processor, device, model_kwargs={"max_new_tokens": 25, "num_beams": 4}):
    """
    Using huggingfaces built-in evaluation library, calculate the rouge metric for the model.

    :param model: The huggingface model to be evaluated.
    :type model: transformers.PreTrainedModel
    :param dataset: The dataset used to validate the model. 
    This set should be iid from the data generating process and not include any examples used during the training of the model. 
    Furthermore, it is best practice to also not include any values which will later be used in the test set.
    :type dataset: torch.utils.data.Dataset
    :param image_processor: The image processor which can extract the input-ids from an image.
    :type image_processor: any
    :param tokenizer: The model outputs tokens, which need to be translated to NLP. To do this, a tokenizer is required.
    :type tokenizer: any
    :param device: The device on which the model is stored.
    :type device: torch.device
    """
    metric = evaluate.load("rouge")
    model.eval()
    predictions, truths = [], []
    with torch.no_grad():
        for pixel_values, caption in tqdm(dataset):
            truths.append(caption)
            predictions.append(_predict(model, pixel_values.to(device), tokenizer, model_kwargs))
        result = metric.compute(predictions=predictions, references=truths)
    return result


def predict(model, image_path, tokenizer, image_processor, device, model_kwargs={"max_new_tokens": 25, "num_beams": 4}):
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
    model.eval()
    with Image.open(image_path) as i_image:
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        pixel_values = image_processor([i_image], return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(device)
    return _predict(model, pixel_values.to(device), tokenizer, model_kwargs)


def _predict(model, pixel_values, tokenizer, model_kwargs):
    output_ids = model.generate(pixel_values=pixel_values, **model_kwargs)
    predictions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return predictions[0].strip()