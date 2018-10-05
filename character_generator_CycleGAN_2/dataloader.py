import glob, random, os

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

"""
OSError: cannot identify image file './data\\train/B\\watercolor278.jpg'

"""

class ImageDataset(Dataset):
    def __init__(self, dataroot, transform=None, unaligned=False, mode='train'):
        self.transform = transform
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(dataroot, 'A') + '/*.*'))
        print(self.files_A)
        self.files_B = sorted(glob.glob(os.path.join(dataroot, 'D') + '/*.*'))
        print(self.files_B)

    def __getitem__(self, idx):
        item_A = Image.open(self.files_A[idx % len(self.files_A)])
        
        if self.unaligned:
            item_B = Image.open(self.files_B[random.randint(0, len(self.files_B)-1)])

        else:
            item_B = Image.open(self.files_B[idx % len(self.files_B)])

        return {'A': self.transform(item_A), 'B':self.transform(item_B)}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

def get_loader(config):
    transform = transforms.Compose([
        transforms.Resize(int(config.image_size * 1.12), Image.BICUBIC),
        transforms.RandomCrop(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = ImageDataset(config.dataroot, transform=transform, unaligned=config.unaligned)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    print(len(dataloader))
    return dataloader

def get_test_loader(config):
    transform = transforms.Compose([
        transforms.Resize(int(config.image_size * 1.12), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = ImageFolder(config.dataroot, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    print(len(dataloader))
    return dataloader
