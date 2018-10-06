import os, glob, shutil
import numpy as np
from tqdm import tqdm
import collections, h5py

from models.encoder import vgg13

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

from PIL import Image

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_test_split(root, split_ratio):
    paths_A = glob.glob(os.path.join(root, 'A/*'))
    paths_B = glob.glob(os.path.join(root, 'B/*'))

    num_train = int(len(paths_A) * (1-split_ratio))

    np.random.shuffle(paths_A)
    np.random.shuffle(paths_B)

    os.makedirs("%s/train/A/A" % root)
    os.makedirs("%s/train/B/B" % root)
    os.makedirs("%s/test/A/A" % root)
    os.makedirs("%s/test/B/B" % root)

    print("Start splitting...")
    for i in tqdm(range(len(paths_A))):
        a, b = paths_A[i], paths_B[i]
        if i <= num_train:
            destA = "%s/train/A/A/%s" % (root, a.split("/")[-1])
            destB = "%s/train/B/B/%s" % (root, b.split("/")[-1])
            # print(a, destA)
            # print(b, destB)
            # os.system("cp %s %s/train/A/%s" % (a,root,a.split("/")[-1]))
            shutil.copy(a, destA)
            # os.system("cp %s %s/train/B/%s" % (b,root,b.split("/")[-1]))
            shutil.copy(b, destB)

        else:
            # os.system("cp %s %s/test/A/%s" % (a,root,a.split("/")[-1]))
            # os.system("cp %s %s/test/B/%s" % (b,root,b.split("/")[-1]))
            shutil.copy(a, "%s/test/A/A/%s" % (root, a.split("/")[-1]))
            shutil.copy(b, "%s/test/B/B/%s" % (root, b.split("/")[-1]))

    print("Finished splitting!")

def image_loader(image_name):
    imsize = 224
    loader = transforms.Compose([transforms.Resize((imsize, imsize)), transforms.ToTensor()])

    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image.to(device)

def prepare_vgg_dataset(root):
    NUM_CLASSES = 2622

    # load pretrained weight
    model = vgg13()
    model.features = torch.nn.Sequential(collections.OrderedDict(zip(
        ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
         'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2',
         'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
         'pool5'], model.features)))
    model.classifier = torch.nn.Sequential(collections.OrderedDict(
        zip(['fc6', 'relu6', 'drop6', 'fc7', 'relu7', 'drop7', 'fc8', 'prob'], model.classifier)))
    state_dict = h5py.File("D:/Deep_learning/Data/멘토_LiveCon/encoder_weights/vggface.h5", 'r')
    torch.load("D:/Deep_learning/Data/멘토_LiveCon/encoder_weights/VGG_FACE.caffemodel.pth")
    model.load_state_dict(
        {l: torch.from_numpy(np.array(v)).view_as(p) for k, v in state_dict.items() for l, p in model.named_parameters()
         if k in l})
    model.eval()

    print(model)

    # prepare images
    img_list = [root + "/" + img for img in os.listdir(root)]
    print(len(img_list))

    # make directories
    for dir_idx in range(NUM_CLASSES):
        target_dir = "D:/Deep_learning/Data/멘토_LiveCon/dataset_cropped/train/A/origin" + str(dir_idx)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        if dir_idx + 1 == 1000:
            print("1000 directories completed!")

    print("[*] Make directories completed!")

    # do validating and copy into corresponding directory
    for idx, img_path in enumerate(img_list):
        img = image_loader(img_path)
        probs = model(img).cpu().detach().numpy()
        cls = np.argmax(probs, axis=1)

        dest = "D:/Deep_learning/Data/멘토_LiveCon/dataset_cropped/train/A/origin" + str(cls.item())
        shutil.copy(img_path, dest)
        print("Copy %s to %s!\n" % (img_path, dest))

    print("[*] Completed!")

if __name__ == "__main__":
    train_test_split('dataset', 0.1)
