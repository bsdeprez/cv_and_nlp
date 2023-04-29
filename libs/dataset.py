import json
from PIL import Image
from torch.utils.data import Dataset

class RSTPReid(Dataset):

    def __init__(self, path, split, tokenizer, ):
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


    def __len__(self):
        """
        Returns the number of entries in the dataset.
        """
        return len(self._df)
    

    def __getitem__(self, index):
        return {
            "image_path": self._image_folder / self._df[index]['img_path'],
            "caption": self._df[index]['captions'][0]
        }

    @staticmethod
    def custom_collate(batch, text_tokenizer, image_processor):
        images = []
        captions = []
        for item in batch:
            i_image = Image.open(item['image_path'])
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")
            images.append(i_image)
            captions.append(item['caption'])
        
        # Extract the features from the image.
        pixel_values = image_processor(images=images, return_tensors='pt').pixel_values
        # Close the images
        for i_image in images:
            i_image.close()
        # Extract the tokens
        tokens = text_tokenizer(captions, padding=True, return_tensors='pt').input_ids
        return {
            "pixel_values": pixel_values.to(device),
            "labels": tokens.to(device)
        }
