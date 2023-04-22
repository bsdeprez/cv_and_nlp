from pathlib import Path
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel

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
        example = self._df[index]
        example['img_path'] = self._image_folder / example['img_path']
        return example

    def __len__(self):
        return len(self._df)

    def __getitem__(self, index):
        """
        Iterator over the dataset.
        
        :param index: Index for accessing the flat image-caption pairs.
        """
        item = self._df[index]
        id = item['id']
        pixel_values = self._get_pixel_values(item['img_path'])
        return id, pixel_values, item['captions'][0]
    
    def _get_pixel_values(self, image_name):
        image_path = self._image_folder / image_name
        i_image = Image.open(image_path)
        if i_image.mode != 'RGB':
            i_image = i_image.convert(mode="RGB")
        pixel_values = self._image_processor(images=[i_image], return_tensors='pt').pixel_values
        i_image.close()
        return pixel_values



if __name__ == '__main__':
    path = Path().resolve().parent / 'Data' / 'RSTPReid'
    image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    val_set = RSTPReid(path, 'val', image_processor)
    """
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    path = Path().resolve().parent / 'Data' / 'RSTPReid'
    assert path.exists()
    val_set = RSTPReid(path, 'val', image_processor)
    dataloader = DataLoader(val_set, batch_size=1, shuffle=False)
    model.train()
    for i in range(1):
        batch = next(iter(dataloader))
        _, pixel_values, captions = batch
        pixel_values = pixel_values.squeeze(0)
        labels = tokenizer(captions, padding=True, return_tensors='pt').input_ids
        print(f"Pixel-values: {pixel_values.shape} ({type(pixel_values)})")
        print(f"Labels: {labels.shape} ({type(labels)})")
        loss = model(pixel_values=pixel_values, labels=labels).loss
        print(loss)
    """
