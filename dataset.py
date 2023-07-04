import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image


df_class = pd.read_csv("class_dict_seg.csv")
dict = {"Name":[]}
df = pd.DataFrame(dict)
df = pd.DataFrame(os.listdir("RGB_color_image_masks/RGB_color_image_masks"))
df_train, df_test = train_test_split(df, test_size=0.1, random_state=19)
df_train, df_valid = train_test_split(df_train, test_size=0.15, random_state=19)



class Custom_dataset(Dataset):
  def __init__(self, image_dir, mask_dir, df, mean, std, transform=True):
    self.df = df
    self.transform = transform
    self.mean = mean
    self.std = std
    self.image_dir = image_dir
    self.mask_dir = mask_dir
    
  def __len__(self):
    return len(self.df)

  def __getitem__(self,idx):
    image = cv2.imread(self.image_dir+"/"+self.df.iloc[idx,0][:-3]+"jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(self.mask_dir+"/"+self.df.iloc[idx,0], cv2.IMREAD_GRAYSCALE)

    aug = self.transform(image=image, mask=mask)
    image = aug["image"]
    image = Image.fromarray(image)
    mask = aug["mask"]
    t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
    image = t(image)
    mask = torch.from_numpy(mask).type(torch.long)
    return image, mask



class test_dataset(Dataset):

    def __init__(self, image_dir, mask_dir, df, transform=None):
        self.transform = transform
        self.df = df
        self.image_dir = image_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_dir+"/"+self.df.iloc[idx,0][:-3]+"jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_dir+"/"+self.df.iloc[idx,0], cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']

        if self.transform is None:
            img = Image.fromarray(img)

        mask = torch.from_numpy(mask).long()

        return img, mask

