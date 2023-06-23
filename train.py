import torch
import torchvision
from torchvision import transforms as T
import torch.nn.functional as F

import albumentations as A
import numpy as np
from tqdm import tqdm


def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

def get_lr(optimizer):
  for param_group in optimizer.param_groups:
      return param_group['lr']
    
def fit(epochs, model, train_loader, valid_loader, criterion, optimizer, scheduler):
    torch.cuda.empty_cache()
    train_losses =[]
    train_accs = []
    valid_losses = []
    valid_accs = []
    lrs = []
    min_loss = np.inf
    decrease = 1
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0

        for batch, mask in tqdm(train_loader):
            model.train()
            batch = batch.to(device)
            mask = mask.to(device)
            y_pred = model(batch)
            loss = criterion(y_pred, mask)
            train_loss += loss
            train_acc += pixel_accuracy(y_pred, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lrs.append(get_lr(optimizer))
            scheduler.step()

        model.eval()
        valid_loss = 0
        valid_acc = 0
        with torch.no_grad():
            for batch, mask in tqdm(valid_loader):
                batch = batch.to(device)
                mask = mask.to(device)
                y_pred = model(batch)
                valid_loss += criterion(y_pred, mask)
                valid_acc += pixel_accuracy(y_pred, mask)

        train_losses.append(train_loss/len(train_loader))
        train_accs.append(train_acc/len(train_loader))
        valid_losses.append(valid_loss/len(valid_loader))
        valid_accs.append(valid_acc/len(valid_loader))
        
        if min_loss > (valid_loss/len(valid_loader)):
            print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (valid_loss/len(valid_loader))))
            min_loss = (valid_loss/len(valid_loader))
            decrease += 1
            if decrease % 3 == 0:
                print('saving model...')
                torch.save(model, 'Unet-{:.3f}.pt'.format(valid_acc/len(valid_loader)))

        print(f"\nEpoch: {epoch+1} | Train_loss: {train_loss/len(train_loader):.5f} | Train_acc:       {(train_acc/len(train_loader)):.5f} | Valid_loss: {valid_loss/len(valid_loader):.5f} | Valid_acc: {(valid_acc/len(valid_loader)):.5f} \n")
    history = {'train_loss' : train_losses, 'val_loss': valid_losses,
               'train_acc' :train_acc, 'val_acc':valid_acc,
               'lrs': lrs}
    return history


