import pandas as pd
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import matplotlib.pyplot as plt
from transformers import DetrFeatureExtractor
import torch
import cv2
import os
import glob

directory = '/Users/victorvanhullebusch/Desktop/CV & NLP project/Test folder'
images = glob.glob(os.path.join(directory, "*.JPG"))
emptylist = []


os.chdir('/Users/victorvanhullebusch/Desktop/CV & NLP project/outputs')
for i in images:
    try:
        image = Image.open(i)
        print(f"analyzing image {i}")
        feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        encoding = feature_extractor(image, return_tensors="pt")
        encoding.keys()
        outputs = model(**encoding)
        COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
                  [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
        probability = outputs.logits.softmax(-1)[0, :, :-1]
        threshold = probability.max(-1).values > 0.95

        target = torch.tensor(image.size[::-1]).unsqueeze(0)
        postprocessed_outputs = feature_extractor.post_process(outputs, target)
        coordinates_boxes = postprocessed_outputs[0]['boxes'][threshold]
        image = cv2.imread(i)
        boxnumber = len(coordinates_boxes)
        colors = COLORS * 100

        label = []
        for p, (xmin, ymin, xmax, ymax), c in zip(probability[threshold], coordinates_boxes.tolist(), colors):
            cl = p.argmax()
            text = model.config.id2label[cl.item()]
            label.append(text)


        for k in range(boxnumber):
            X, Y, W, H = coordinates_boxes[k].int()
            #X = X.int()
            #Y = Y.int()
            #W = W.int()
            #H = H.int()
            coordinates = image[Y:H, X:W]
            #os.chdir('/Users/victorvanhullebusch/Desktop/CV & NLP project/outputs')
            img = f'{i}_{k}_{label[k]}.JPG'
            cv2.imwrite(img, coordinates)

    except cv2.error:
        emptylist.append(i)
        continue

