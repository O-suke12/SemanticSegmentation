import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import albumentations as A

from dataset import df_train
from dataset import df_valid
from dataset import Custom_dataset
from model import model
from train import fit

def main():
    
    t_train = A.Compose([
    A.Resize(640, 864),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.GridDistortion(p=0.2),
    A.RandomBrightnessContrast((0,0.5),(0,0.5)),
    A.GaussNoise()
    ])

    t_valid = A.Compose([
        A.Resize(640, 864),
    ])

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    image_dir = "dataset/dataset/semantic_drone_dataset/original_images"
    mask_dir = "dataset/dataset/semantic_drone_dataset/label_images_semantic"

    train_set = Custom_dataset(image_dir, mask_dir, df_train, mean, std, t_train)
    valid_set = Custom_dataset(image_dir, mask_dir, df_valid, mean, std, t_valid)
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=2, shuffle=True, drop_last=True)
    
    
    max_lr = 1e-3
    epoch = 10
    weight_decay = 1e-4
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
                                                steps_per_epoch=len(train_loader))

    history = fit(epoch, model, train_loader, valid_loader, criterion, optimizer, sched)
    
    
if __name__ == "__main__":
    main()