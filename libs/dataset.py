import json
import torch
from PIL import Image
from torch.utils.data import Dataset

class RSTPReid(Dataset):

    def __init__(self, path, split, image_processor):
        # Verify 
        self._image_folder = path / 'imgs'
        if not self._image_folder.exists():
            raise FileNotFoundError(f"Could not find the images at {self._image_folder}.")
        if split not in ['train', 'test', 'val']:
            raise IndexError(f"The given split is not valid. Split must be 'train', 'test' or 'val'.")
        annotations_file = path / 'data_captions.json' 
        if not annotations_file.exists():
            raise FileNotFoundError(f"Could not find the annotations file: {annotations_file}")
        # Read the json files with the descriptions.
        with open(annotations_file) as file_descriptor:
            self._df = json.load(file_descriptor)
        # Split the dataset according to the given split.
        self._df = list(filter(lambda x: x['split'] == split, self._df))
        self._image_processor = image_processor

    def __len__(self):
        """
        Returns the number of entries in the dataset.
        """
        return len(self._df)
    

    def get_example(self, index):
        return {
            "image_path": self._image_folder / self._df[index]['img_path'],
            "caption": self._df[index]['captions'][0]
        }
    
    def __getitem__(self, index):
        image_path = self._image_folder / self._df[index]["img_path"]
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        pixel_values = self._image_processor(images=[i_image], return_tensors="pt").pixel_values
        i_image.close()
        return pixel_values, self._df[index]['captions'][0]

    @staticmethod
    def custom_collate(batch):
        all_pixel_values = []
        all_captions = []
        for pixel_values, caption in batch:
            all_pixel_values.append(pixel_values)
            all_captions.append(caption)
        return torch.stack(all_pixel_values).squeeze(), all_captions
