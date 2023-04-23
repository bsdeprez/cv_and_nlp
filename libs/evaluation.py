import evaluate
import torch
from tqdm import tqdm

def evaluate(model, validation_set, tokenizer, metric="rouge", device=torch.device('cpu')):
    """
    Using huggingfaces built-in evaluation library, calculate a metric for the model.

    :param model: The huggingface model to be evaluated.
    :type model: transformers.PreTrainedModel

    :param validation_set: The dataset used to validate the model. 
    This set should be iid from the data generating process and not include any examples used during the training of the model. 
    Furthermore, it is best practice to also not include any values which will later be used in the test set.
    :type validation_set: torch.utils.data.Dataset

    :param tokenizer: The model outputs tokens, which need to be translated to NLP. To do this, a tokenizer is required.
    :type tokenizer: any

    :param metric: The metric you want to use for evaluation. A list of available metrics can be found at https://huggingface.co/evaluate-metric
    :type metric: str

    :param device: The device on which the model is stored.
    :type device: torch.device
    """
    metric = evaluate.load(metric)
    model.eval()
    predictions, truths = _predict(model, validation_set, tokenizer, device)
    result = metric.compute(predictions=predictions, references=truths)
    print(result)


def _predict(model, dataset, tokenizer, device):
    """
    Use the model to create predictions for the given dataset.
    """
    predictions, truths = [], []
    for i in tqdm(range(len(dataset))):
        _, pixel_values, truth = dataset[i]
        pixel_values = pixel_values.to(device)
        output_ids = model.generate(pixel_values, max_length=128)
        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        predictions.append(preds[0])
        truths.append(truth)
    return predictions, truths

