import glob, random, os

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import cv2

class ImageDataset(Dataset):
    def __init__(self, datarootA, datarootB, transform=None, unaligned=False):
        self.datarootA = datarootA
        self.datarootB = datarootB
        self.transform = transform
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(datarootA + '/*.*'))
        print("length of datasetA:", len(self.files_A))
        self.files_B = sorted(glob.glob(datarootB + '/*.*'))
        print("length of datasetB:", len(self.files_B))

        # self.preprocess_4channel_images()
        # input("Converting completed!")

    def preprocess_4channel_images(self):
        for idx in range(len(self.files_B)):
            img = cv2.imread(self.files_B[idx])
            cv2.imwrite(os.path.join(self.datarootB, "new{}.jpg".format(idx)), img)
            os.remove(self.files_B[idx])

        self.files_B = sorted(glob.glob(self.datarootB + '/*.*'))
        print("length of datasetB:", len(self.files_B))


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

    dataset = ImageDataset(config.datarootA, config.datarootB, transform=transform, unaligned=config.unaligned)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    print(len(dataloader))
    return dataloader

def get_test_loader(config):
    transform = transforms.Compose([
        transforms.Resize(int(config.image_size * 1.12), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = ImageFolder(config.datarootA, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    print(len(dataloader))
    return dataloader
