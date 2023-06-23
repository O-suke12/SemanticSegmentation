import torch
from torchvision import transforms as T
import cv2
import numpy as np
import os
from tqdm import tqdm
import itertools
import albumentations as A
import matplotlib.pyplot as plt
from model import model
from dataset import df_test
from dataset import test_dataset
from train import pixel_accuracy

def pick_best_worst(model, test_set, device, mean, std):
    best_id = []
    mean_id = []
    worst_id = []
    acc_list = []
    accuracy = {}
    with torch.no_grad():
        for i in tqdm(range(len(test_set))):
            image, mask = test_set[i]
            t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
            image = t(image)
            image = image.to(device); mask = mask.to(device)
            image = image.unsqueeze(0)
            mask = mask.unsqueeze(0)
            
            pred = model(image)
            acc = pixel_accuracy(pred, mask)
            accuracy.update({i: acc})
    
    acc_list = accuracy.values()
    fig = plt.figure()
    plt.hist(acc_list)
    
    fig.savefig('evaluation/ditribution.png')
    worst_id = sorted(accuracy.items(), key=lambda x:x[1])[:3]
    best_id = sorted(accuracy.items(), key=lambda x:x[1], reverse=True)[:3]
    worst_id = [pair[0] for pair in worst_id]
    best_id = [pair[0] for pair in best_id]
    average = sum(acc_list)/len(acc_list)
    accuracy = {idx: acc-average for idx, acc in accuracy.items()}
    mean_id = sorted(accuracy.items(), key=lambda x:abs(x[1]))[:3]
    mean_id = [pair[0] for pair in mean_id]
    return best_id, mean_id, worst_id

def save_images(Unet, test_set, device, id_list, mean, std, cond):
    with torch.no_grad():
        for i in id_list:
            origin, mask = test_set[i]
            t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
            image = t(origin)
            image = (image.unsqueeze(0)).to(device)
            mask = (mask.unsqueeze(0)).to(device)

            pred = Unet(image)
            acc = pixel_accuracy(pred, mask)
            fig = plt.figure()
            t = T.Compose([T.ToTensor()])
            origin = t(origin)
            plt.imshow(origin.permute(1,2,0))
            fig.savefig(f'evaluation/{cond}3/origin_{cond}{i}.png')
            mask = mask.cpu().squeeze(0)
            plt.imshow(mask)
            fig.savefig(f'evaluation/{cond}3/mask_{cond}{i}.png')
            pred = pred.cpu().squeeze(0)
            pred = torch.argmax(pred, dim=0)
            plt.imshow(pred)
            fig.suptitle('Pixel Accuracy {:.3f}'.format(acc), fontsize=20)
            fig.savefig(f'evaluation/{cond}3/pred_{cond}{i}.png')
    
   
    
def evaluation(Unet, test_set, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    
    Unet.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Unet.to(device)
    
    best_id = []
    mean_id = []
    worst_id = []
    best_id, mean_id, worst_id = pick_best_worst(Unet, test_set, device, mean, std)
    save_images(Unet, test_set, device, best_id, mean, std, cond="best")
    save_images(Unet, test_set, device, mean_id, mean, std, cond="mean")
    save_images(Unet, test_set, device, worst_id, mean, std, cond="worst")

def read_model():
    model_name = ""
    max_acc = 0.0
    for file in os.listdir():
        if file.endswith(".pt"):
            acc = os.path.splitext(file)[0]
            acc = float(acc[-5:])

            if max_acc < acc:
                max_acc = acc
                model_name = file
    return model_name
        

def main():
    image_dir = "dataset/dataset/semantic_drone_dataset/original_images"
    mask_dir = "dataset/dataset/semantic_drone_dataset/label_images_semantic"
    t_test = A.Resize(640, 864, interpolation=cv2.INTER_NEAREST)
    test_set = test_dataset(image_dir, mask_dir, df_test, transform=t_test)
    
    model = torch.load(read_model())
    evaluation(model, test_set)


if __name__ == "__main__":
    main()