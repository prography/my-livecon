import argparse, os
from PIL import Image
import numpy as np

import torch
from torch.optim import Adam
import torchvision.transforms as T

from model import GlassNet
from vis_tool import Visualizer

def get_config():
    parser = argparse.ArgumentParser()
    # Data configuration
    parser.add_argument('--dataroot', type=str, default="dataset/MeGlass_120x120", help='path to dataset')
    parser.add_argument('--label_path', type=str, default="dataset/meta.txt", help='path to metadata file')
    parser.add_argument('--image_size', type=int, default=120, help='image size')


    # Training configuration
    parser.add_argument('--num_epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate of optimizer')

    # Steps configuration
    parser.add_argument('--log_interval', type=int, default=2000, help='log print interval (step)')
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='model saving interval (epoch)')
    parser.add_argument('--val_interval', type=int, default=1, help='accuracy computing interval (epoch)')

    # Path configuration
    parser.add_argument('--checkpoint_folder', type=str, default='checkpoints', help='path to save checkpoint files')

    config = parser.parse_args()
    return config

def train():
    # get configurations
    config = get_config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # make directory
    if not os.path.exists(config.checkpoint_folder):
        os.makedirs(config.checkpoint_folder)

    # build network
    net = GlassNet(config.image_size, 3, 16, 2)
    net = net.to(device)

    # open metadata file
    with open(config.label_path, 'r') as f:
        labels = f.readlines()

    train_idx = int(len(labels) * 0.7)
    train_labels = labels[:train_idx]
    val_labels = labels[train_idx:]

    # define image transformation
    transform = T.Compose([
        T.Resize((config.image_size, config.image_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.5,), std=(0.5,))
    ])

    # define optimizer and loss function
    optimizer = Adam(net.parameters(), config.lr)
    criterion = torch.nn.BCELoss().to(device)

    # define visualizer
    vis = Visualizer()

    iter = 0
    loss_store = []
    print("[*]Learning Started!")
    for epoch in range(config.num_epochs):
        for idx, data in enumerate(train_labels):
            filename, cls = data.split()
            image = Image.open(os.path.join(config.dataroot, filename))
            image_tensor = transform(image).unsqueeze(0)
            cls_tensor = torch.LongTensor(1, 1).fill_(int(cls))
            cls_onehot = torch.FloatTensor(1, 2)

            cls_onehot.zero_()
            cls_onehot.scatter_(1, cls_tensor, 1)

            image_tensor = image_tensor.to(device)
            cls_onehot = cls_onehot.to(device)

            out = net(image_tensor)
            loss = criterion(out, cls_onehot)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_store.append(loss.item())

            if (iter+1) % config.log_interval == 0:
                # do logging
                print("[%d/%d] [%d/%d] loss:%.5f" %
                      (epoch+1, config.num_epochs,
                       idx+1, len(train_labels), np.mean(loss_store)))

                vis.plot("Loss plot", np.mean(loss_store))
                loss_store.clear()

            iter += 1

        if (epoch+1) % config.checkpoint_interval == 0:
            # save checkpoint file
            ckpt_path = os.path.join(config.checkpoint_folder, "epoch{}.pth".format(epoch+1))
            torch.save(net.state_dict(), ckpt_path)
            print("[*]Checkpoint saved!")

        if (epoch+1) % config.val_interval == 0:
            # do evaluating
            accuracy = 0.

            with torch.no_grad():
                for data in val_labels:
                    filename, cls = data.split()
                    image = Image.open(os.path.join(config.dataroot, filename))
                    image_tensor = transform(image).unsqueeze(0)
                    cls_tensor = torch.LongTensor(1, 1).fill_(int(cls))
                    cls_onehot = torch.FloatTensor(1, 2)

                    cls_onehot.zero_()
                    cls_onehot.scatter_(1, cls_tensor, 1)

                    image_tensor = image_tensor.to(device)
                    cls_onehot = cls_onehot.to(device)

                    out = net(image_tensor)

                    predicted = torch.argmax(out, dim=1)
                    actual = torch.argmax(cls_onehot, dim=1)

                    if predicted == actual:
                        accuracy += float(1) / len(val_labels)
                print("[*]Validating Accuracy: %.3f%%" % (accuracy*100))

if __name__ == '__main__':
    train()