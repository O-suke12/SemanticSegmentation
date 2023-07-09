import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from dataset import df_train
from dataset import Custom_dataset
import albumentations as A
import collections, functools, operator

image_dir = "dataset/semantic_drone_dataset/original_images"
mask_dir = "dataset/semantic_drone_dataset/label_images_semantic"
class_dir = "class_dict_seg.csv"

t_train = A.Compose([
    A.Resize(640, 864),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.GridDistortion(p=0.2),
    A.RandomBrightnessContrast((0,0.5),(0,0.5)),
    A.GaussNoise()
    ])
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
train_set = Custom_dataset(image_dir, mask_dir, df_train, mean, std, t_train)

freq_list = []
for i in range(len(train_set)):
    sample = train_set.__getitem__(i)[1].numpy()
    unique, counts = np.unique(sample, return_counts=True)
    freq_list.append(dict(zip(unique, counts)))

result = dict(functools.reduce(operator.add,map(collections.Counter, freq_list)))
myKeys = list(result.keys())
myKeys.sort()
result = {i: result[i] for i in myKeys}

fig = plt.figure()
color = np.array(range(23))
plt.imshow([color,color, color])
fig.savefig('evaluation/imcolor_dist.png')

fig = plt.figure()
plt.bar(result.keys(), result.values(), width=0.9)
plt.xticks(list(result.keys()), rotation ='horizontal')
fig.savefig('evaluation/image_dist.png')


