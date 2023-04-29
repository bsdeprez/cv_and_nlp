from pathlib import Path
import json
from torch.utils.data import Dataset
from PIL import Image

class RSTPReid(Dataset):
    """
    A torch.utils.data Dataset which represents the RSTPReid dataset.
    
    The implementation of this class is based on the following sources:
    - https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    - https://ankur3107.github.io/blogs/the-illustrated-image-captioning-using-transformers/
    """
    
    def __init__(self, path, split, image_processor) -> None:
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
        # read the json files with the descriptions.
        annotations_file = path / 'data_captions.json'
        if not annotations_file.exists():
            raise FileNotFoundError(f"Could not find the annotations file: {annotations_file}")
        with open(annotations_file) as file:
            self._df = json.load(file)
        # Split in the requested data set.
        filter_object = filter(lambda x: x['split'] == split, self._df)
        self._df = list(filter_object)

        # Store the feature extractors
        self._image_processor = image_processor
        self.max_sequence_length=128  # Important for padding length

    def get_example(self, index):
        """ 
        Returns an example of an entry in the RSTPReid dataset before preprocessing.
        
        :param index: The index of the example.
        :type index: int
        """
        example = self._df[index]
        example['img_path'] = self._image_folder / example['img_path']
        return example

    def __len__(self):
        """
        Returns the number of entries in the dataset.
        """
        return len(self._df)

    def __getitem__(self, index):
        """
        Iterator over the dataset.
        
        :param index: Index for accessing the flat image-caption pairs.
        :type index: int
        """
        item = self._df[index]
        id = item['id']
        pixel_values = self._get_pixel_values(item['img_path'])
        return id, pixel_values, item['captions'][0]
    
    def _get_pixel_values(self, image_name):
        """
        Preprocess the image.
        This method starts by converting the image to RGB, and then uses a predefined feature extractor to 
        select the most interesting features.
        
        :rtype: torch.Tensor
        """
        image_path = self._image_folder / image_name
        i_image = Image.open(image_path)
        if i_image.mode != 'RGB':
            i_image = i_image.convert(mode="RGB")
        pixel_values = self._image_processor(images=[i_image], return_tensors='pt').pixel_values
        i_image.close()
        return pixel_values
