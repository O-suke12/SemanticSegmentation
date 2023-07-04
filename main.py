import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import albumentations as A

from dataset import df_train
from dataset import df_valid
from dataset import Custom_dataset
from model import model
from train import fit
import matplotlib.pyplot as plt

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

    image_dir = "dataset/semantic_drone_dataset/original_images"
    mask_dir = "dataset/semantic_drone_dataset/label_images_semantic"

    train_set = Custom_dataset(image_dir, mask_dir, df_train, mean, std, t_train)
    valid_set = Custom_dataset(image_dir, mask_dir, df_valid, mean, std, t_valid)
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=2, shuffle=True, drop_last=True)
    
    
    max_lr = 1e-3
    epoch = 15
    weight_decay = 1e-4
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
                                                steps_per_epoch=len(train_loader))

    history = fit(epoch, model, train_loader, valid_loader, criterion, optimizer, sched)
    torch.save(model, 'Unet-Mobilenet.pt')
    plot_loss(history)
    plot_acc(history)
    
def plot_loss(history):
    fig = plt.figure()
    his_val = torch.tensor(history["val_loss"], device = 'cpu')
    his_train = torch.tensor(history["train_loss"], device = 'cpu')
    plt.plot(his_val, label='val', marker='o')
    plt.plot( his_train, label='train', marker='o')
    plt.title('Loss per epoch'); plt.ylabel('loss');
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    fig.savefig('evaluation/loss.png')
    
    
def plot_acc(history):
    fig = plt.figure()
    his_train = torch.tensor(history["train_acc"], device = 'cpu')
    his_val = torch.tensor(history["val_acc"], device = 'cpu')
    plt.plot(his_train, label='train_accuracy', marker='*')
    plt.plot(his_val, label='val_accuracy',  marker='*')
    plt.title('Accuracy per epoch'); plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    fig.savefig('evaluation/acc.png')
    
    
if __name__ == "__main__":
    main()
    