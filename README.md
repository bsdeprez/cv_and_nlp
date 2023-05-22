# Tracking passenger movement using descriptive identifications
## Introduction
Most trains are equipped with Closed Circuit Television (CCTV) cameras, which provide valuable data to determine the occupancy of train cars. This information can facilitate the allocation of passengers to less crowded cars, while directing them away from congested ones. However, monitoring the movement of passengers from one car, and consequently one CCTV feed, to another poses a challenge. This project aims to explore the feasibility of recognizing individuals on a CCTV feed and generating a description of each person that is easily understandable to humans, such as "A person wearing a blue sweater with short, black hair". Subsequently, we aim to develop a Natural Language Processing (NLP) model that can distinguish between descriptions referring to the same person and those pertaining to different individuals. In this manner, we can track people in a way that is less invasive to their privacy, without retaining pictures or embeddings of individuals. Furthermore, it creates an explainable tracking mechanism.

## Data availability
### Test Set
We have acquired access to video footage of individuals moving inside a train car, obtained during a prior student project conducted on behalf of Televic Rail. This dataset contains both images and videos, similar to the example presented here. ![example_image](https://github.com/bsdeprez/cv_and_nlp/blob/main/Data/ExampleImageTestData.JPG)

This dataset will be used as test set, but it still needs to be annotated.

### Training sets
#### **CUHK-PEDES**
The CUHK-PEDES dataset is a caption-annotated pedestrian dataset. It contains 40206 images of 13003 persons. Images are collected from five existing person re-identification datasets, CUHK03, Market-1501, SSM, VIPER, and CUHK01 while each image is annotated with 2 text descriptions by crowd-sourcing workers. Sentences incorporate rich details about person appearances, actions, poses.

A description of the original paper and a way to contact the author can be found [here](http://xiaotong.me/static/projects/person-search-language/dataset.html).  
Alternatively, [this github](https://github.com/zifyloo/SSAN) references [this page](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description) as a place to download the data, but we also need to request the data from the author.
The CUHK-PEDES dataset is a caption-annotated pedestrian dataset. It contains 40206 images of 13003 persons. Images are collected from five existing person re-identification datasets, CUHK03, Market-1501, SSM, VIPER, and CUHK01 while each image is annotated with 2 text descriptions by crowd-sourcing workers. Sentences incorporate rich details about person appearances, actions, poses.

A description of the original paper and a way to contact the author can be found [here](http://xiaotong.me/static/projects/person-search-language/dataset.html).  
Alternatively, [this github](https://github.com/zifyloo/SSAN) references [this page](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description) as a place to download the data, but we also need to request the data from the author.

#### **RSTPReid**
The RSTPReid dataset contains 20505 images of 4101 persons from 15 cameras. Each person has 5 corresponding images taken by different cameras with complex both and outdoor scene transformations and backgrounds in various periods of time. Each image is annotated with 2 textual descriptions. ([source](https://github.com/NjtechCVLab/RSTPReid-Dataset))

The dataset can be downloaded from [google drive](https://drive.google.com/file/d/1HTeDZUVrZr6nL56ZlkYBNqjSWh3IGV2X/view).

#### **People clothing segmentation**
If we can label the type of clothes from a person, we could extract the colour directly from the image. This is more sensitive to flaws of a specific camera, but the third model might learn to deal with that?  
An example dataset can be found [here](https://www.kaggle.com/datasets/rajkumarl/people-clothing-segmentation)

### **Fashion Product Images**
We could also train the model on just clothing specific information (found [here](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)).

## Project Structure

### libs
This directory contains some handwritten python libraries which we used to load in the RSTPReid data and train the ViT-GPT2 model.

### Detr-resnet-Picture.py


### Evaluation of the full pipeline - Part 1.ipynb 
This notebook contains the first half of the evaluation on the full pipeline. Since our model consists of three separate parts, each connected to each other, we've parsed the testset model per model (the VSC didn't like it when we loaded all three models in memory at the same time). This notebook takes in the hand-annotated test data, lets DETR-ResNet50 snippet all recognized persons out of the images, and creates a snapshot of the results in the form of a csv-file. This csv-file is then read in again, and the ViT-GPT2 model creates captions of all the image-snippets. These transcriptions are then saved again in a csv-file, which will serve as the input for the third part of our pipeline. The notebooks are created in such a way that snapshotting the data does not influence the way our pipeline functions.

### Evaluation of the full pipeline - Part 2.ipynb 


### Image captioning.ipynb
This notebook contains the research done into the ViT-GPT2 model. It starts by investigating the RSTPReid data, which is followed by an initial evaluation of the zero-shot capabilities of the encoder-decoder pair. Then, the model is finetuned on the RSTPReid data, and the training loss and validation metrics are plotted per epoch.

### Initial exploration.ipynb


### Sentence similarity_modeltraining.ipynb
This notebook focusses on the sentence similarity part, based on the RSTPreid dataset. 
Different model architectures, as well as training approaches are discussed.
Results on the, self-created validation set, are presented. 
The finally chosen trained model is saved and afterwards used while testing in ‘Evaluation of the full pipeline - Part 2.ipynb ’

