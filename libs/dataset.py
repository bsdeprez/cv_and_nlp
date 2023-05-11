import json
import torch
from PIL import Image
from torch.utils.data import Dataset

class RSTPReid(Dataset):
    """
    A torch.utils.data Dataset which represents the RSTPReid dataset.
    
    The implementation of this class is based on the following sources:
    - https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    - https://ankur3107.github.io/blogs/the-illustrated-image-captioning-using-transformers/
    """

    def __init__(self, path, split, image_processor):
        """ Creates the RSTPReid dataset
        
        :param path: The path to the RSTPReid folder.
        :type path: pathlib.Path
        :param split: The split on which the dataset should be created. This is either 'train', 'test' or 'val'.
        :type split: str
        :param image_processor: The model which extracts the required features from the image.
        :type image_processor: ViTImageProcessor
        """ 
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
        """ 
        Returns an example of an entry in the RSTPReid dataset before preprocessing.
        
        :param index: The index of the example.
        :type index: int
        """
        return {
            "image_path": self._image_folder / self._df[index]['img_path'],
            "caption": self._df[index]['captions'][0]
        }
    
    def __getitem__(self, index):
        """
        Iterator over the dataset.
        
        :param index: Index for accessing the flat image-caption pairs.
        :type index: int
        """
        image_path = self._image_folder / self._df[index]["img_path"]
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        pixel_values = self._image_processor(images=[i_image], return_tensors="pt").pixel_values
        i_image.close()
        return pixel_values, self._df[index]['captions'][0].replace(".", ". ").replace("  ", " ").strip()

    def remove_indices(self, indices):
        indices.sort(reverse=True)
        for index in indices:
            self._df.pop(index)
    
    @staticmethod
    def custom_collate(batch):
        """ 
        Transform a batch into a usable model input.
        """
        all_pixel_values = []
        all_captions = []
        for pixel_values, caption in batch:
            all_pixel_values.append(pixel_values)
            all_captions.append(caption)
        return torch.stack(all_pixel_values).squeeze(), all_captions
