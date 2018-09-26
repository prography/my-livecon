import os, glob,shutil
from tqdm import tqdm
from PIL import Image
import torch.utils.data
import numpy as np
import torchvision.transforms as transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, image_size, data_type):
        self.root = root
        if not os.path.exists(self.root):
            raise Exception("[!] {} not exists.".format(root))

        self.name = os.path.basename(root)

        self.paths = glob.glob(os.path.join(self.root, '{}/*'.format(data_type)))
        if len(self.paths) == 0:
            raise Exception("No images are found in {}".format(self.root))

        self.shape = list(Image.open(self.paths[0]).size) + [3]

        self.train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert('RGB')
        return self.train_transform(image)

    def __len__(self):
        return len(self.paths)

def checkAB(root):
    pathlist = os.listdir(root)
    if "A" in pathlist and "B" in pathlist:
        len_A = len(os.listdir(os.path.join(root, "A")))
        len_B = len(os.listdir(os.path.join(root, "B")))
    else:
        raise Exception("No dataset in path!")

    if len_A != len_B:
        raise Exception("Sizes of A B dataset are not same!")

    train_root = os.path.join(root, "train")
    test_root = os.path.join(root, "test")

    type_list = [train_root,os.path.join(train_root,"A"),os.path.join(train_root,"B"),
                 test_root, os.path.join(test_root,"A"),os.path.join(test_root,"B")]
    for p in type_list:
        if not os.path.exists(p):
            os.system("mkdir %s" % p)
            print("Creating path",p)

    trainA, trainB = os.path.exists(os.path.join(train_root,"A")), os.path.exists(os.path.join(train_root,"B"))
    testA, testB = os.path.exists(os.path.join(test_root,"A")), os.path.exists(os.path.join(test_root,"B"))

    return trainA and trainB and testA and testB

def train_test_split(root, split_ratio):
    paths_A = glob.glob(os.path.join(root, 'A/*'))
    paths_B = glob.glob(os.path.join(root, 'B/*'))

    num_train = int(len(paths_A) * (1-split_ratio))

    np.random.shuffle(paths_A)
    np.random.shuffle(paths_B)

    print("Start splitting...")
    for i in tqdm(range(len(paths_A))):
        a, b = paths_A[i], paths_B[i]
        if i <= num_train:
            destA = "%s/train/A/%s" % (root, a.split("\\")[-1])
            destB = "%s/train/B/%s" % (root, b.split("\\")[-1])
            # print(a, destA)
            # print(b, destB)
            # os.system("cp %s %s/train/A/%s" % (a,root,a.split("/")[-1]))
            shutil.copy(a, destA)
            # os.system("cp %s %s/train/B/%s" % (b,root,b.split("/")[-1]))
            shutil.copy(b, destB)

        else:
            # os.system("cp %s %s/test/A/%s" % (a,root,a.split("/")[-1]))
            # os.system("cp %s %s/test/B/%s" % (b,root,b.split("/")[-1]))
            shutil.copy(a, "%s/test/A/%s" % (root, a.split("\\")[-1]))
            shutil.copy(b, "%s/test/B/%s" % (root, b.split("\\")[-1]))

    print("Finished splitting!")

def get_train_loader(root, batch_size, image_size, split_ratio, num_workers=1):
    train_root = os.path.join(root, "train")

    if not checkAB(root):
        raise Exception("Incorrect directory settings!")

    len_train_A = len(glob.glob(os.path.join(train_root, 'A/*')))
    len_train_B = len(glob.glob(os.path.join(train_root, 'B/*')))


    if len_train_A and len_train_B:
        pass
    else:
        train_test_split(root, split_ratio)

    trainA_dataset, trainB_dataset = \
        Dataset(train_root, image_size, "A"), \
        Dataset(train_root, image_size, "B")

    trainA_loader = torch.utils.data.DataLoader(dataset=trainA_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers)
    trainB_loader = torch.utils.data.DataLoader(dataset=trainB_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers)

    trainA_loader.shape = trainA_dataset.shape
    trainB_loader.shape = trainB_dataset.shape


    train_loaders = [trainA_loader, trainB_loader]
    return train_loaders

def get_test_loader(root, batch_size, image_size, split_ratio, num_workers=1):
    test_root = os.path.join(root, 'test')

    if not checkAB(root):
        raise Exception("Incorrect directory settings!")

    len_test_A = len(glob.glob(os.path.join(test_root, 'A/*')))
    len_test_B = len(glob.glob(os.path.join(test_root, 'B/*')))


    if len_test_A and len_test_B:
        pass
    else:
        train_test_split(root, split_ratio)

    testA_dataset, testB_dataset = \
        Dataset(test_root, image_size, "A"), \
        Dataset(test_root, image_size, "B")

    testA_loader = torch.utils.data.DataLoader(dataset=testA_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers)
    testB_loader = torch.utils.data.DataLoader(dataset=testB_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers)

    testA_loader.shape = testA_dataset.shape
    testB_loader.shape = testB_dataset.shape


    test_loaders = [testA_loader, testB_loader]
    return test_loaders