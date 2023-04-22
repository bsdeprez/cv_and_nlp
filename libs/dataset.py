import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class RSTPReid(Dataset):

    def __init__(self, dataset_directory: Path, split, feature_extractor, tokenizer):
        # Set root directory for the images.
        self._image_folder = dataset_directory / 'imgs'
        if not self._image_folder.exists():
            raise FileNotFoundError(f"Could not find the images at {self._image_folder}.")
        if split not in ['train', 'test', 'val']:
            raise IndexError(f"The given split is not valid. Split must be 'train', 'test' or 'val'.")
        # read the json files with the descriptions.
        annotations_file = dataset_directory / 'data_captions.json'
        if not annotations_file.exists():
            raise FileNotFoundError(f"Could not find the annotations file: {annotations_file}")
        with open(annotations_file) as file:
            self._df = json.load(file)
        # Split in the requested data set.
        filter_object = filter(lambda x: x['split'] == split, self._df)
        self._df = list(filter_object)
        # Store the feature extractor and tokenizer.
        self._feature_extractor = feature_extractor
        self._tokenizer = tokenizer

    def __len__(self):
        return len(self._df)

    def __getitem__(self, index):
        example = self._df[index]
        captions = self._tokenize_caption(example['captions'])
        pixel_values = self._preprocess_image(example["img_path"])
        return pixel_values, captions[0]

    def _preprocess_image(self, image_name):
        image_file_name = self._image_folder / image_name
        i_image = Image.open(image_file_name)
        if i_image.mode != 'RGB':
            i_image = i_image.convert(mode="RGB")
        pixel_values = self._feature_extractor(images=[i_image], return_tensors='pt').pixel_values
        pixel_values = None
        i_image.close()
        return pixel_values
    
    def _tokenize_caption(self, captions):
        labels = self._tokenizer(captions, padding="max_length", max_target_length=25).input_ids
        return labels;
