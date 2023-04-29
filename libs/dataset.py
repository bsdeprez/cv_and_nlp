import json
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

class RSTPReid(Dataset):
    
    def __init__(self, path, split, tokenizer, feature_extractor, device, max_length=128):
        self._image_folder = path / 'imgs'
        annotations_file = path / 'data_captions.json'
        for f in [self._image_folder, annotations_file]:
            if not f.exists():
                raise FileNotFoundError(f"File not found: {f}")
        with open(annotations_file) as fd:
            self._df = json.load(fd)
        # Split in the requested data set.
        filter_object = filter(lambda x: x['split'] == split, self._df)
        self._df = list(filter_object)

        # Store the tokenizer and feature extractor
        self._feature_extractor = feature_extractor
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._device = device

    def __len__(self):
        return len(self._df)

    def __getitem__(self, index):
        example = self._df[index]
        # Prepare image
        image = Image.open(self._image_folder / example['img_path']).convert("RGB")
        pixel_values = self._feature_extractor(image, return_tensors='pt').pixel_values
        caption = example['captions'][0].replace('.', '. ').replace('  ', ' ').strip()
        labels = self._tokenizer(caption, truncation=True, padding="max_length", max_length = self._max_length).input_ids
        # Make sure that the PAD tokens are ignored by the loss function.
        labels = [label if label != self._tokenizer.pad_token_id else -100 for label in labels]
        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels)
        }
