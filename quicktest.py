import os

IMAGE_FOLDER = 'Data/cctv-images/images'
test_image = os.path.join(IMAGE_FOLDER, os.listdir(IMAGE_FOLDER)[94])
print(test_image)

weights = 'Data/Models/weights/yolov7.pt'

assert os.path.exists(test_image)
assert os.path.exists(weights)

command = f"python3 detect.py --source {test_image} --weights {weights}"


