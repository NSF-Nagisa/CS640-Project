import argparse
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize

from model import *
from utils import setup_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='img.png')
    parser.add_argument('--model_path', type=str, default='vit-t-mae.pt')

    args = parser.parse_args()

    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))

    model = torch.load(args.model_path, map_location='cpu')

    ''' visualize the first 16 predicted images on val dataset'''
    model.eval()
    with torch.no_grad():
        samples = np.random.choice(1000, 8, replace=False)
        val_img = torch.stack([val_dataset[i][0] for i in samples])
        predicted_val_img, mask = model(val_img)
        predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
        img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
        img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
        torchvision.utils.save_image(img, args.img_path)

        
