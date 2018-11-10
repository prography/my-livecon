from __future__ import print_function
import random
import os

import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True

import torchvision.transforms as T

from trainer import Trainer
from tester import Tester
from config import get_config
from dataloader import get_loader
# from prepare_dataset import split_train_test

def main(config):
    # make directories
    if config.sample_folder is None:
        config.sample_folder = 'samples_%s' % config.datasetB
    os.system('mkdir {0}'.format(config.sample_folder))
    print("Make sample folder!")

    if config.ckpt_folder is None:
        config.ckpt_folder = 'checkpoints_%s' % config.datasetB
    os.system('mkdir {0}'.format(config.ckpt_folder))
    print("Make checkpoints folder!")

    if config.testFolder is None:
        config.testFolder = "test_%s" % config.datasetB
    os.system('mkdir {0}'.format(config.testFolder))
    print("Make test results folder!")

    config.manual_seed = random.randint(1, 10000)
    print("Random Seed: ", config.manual_seed)

    random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)
    torch.cuda.manual_seed_all(config.manual_seed)

    # for faster training
    cudnn.benchmark = True

    # setup dataloader

    # for google cartoon dataset, apply other transformation
    google_cartoon_tns = T.Compose([
        T.Resize(330),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    print("[*] Preparing dataloader ...")
    train_loader_A, train_set_A = get_loader(config.datasetA, config.datarootA,
                                       config.original_image_size, config.image_size,
                                       config.batch_size, config.workers, split='train')

    val_loader_A, val_set_A = get_loader(config.datasetA, config.valDatarootA,
                                       config.original_image_size, config.image_size,
                                       config.val_batch_size, config.workers, split='test')
                                         # transform_fn=google_cartoon_tns)

    train_loader_B, train_set_B = get_loader(config.datasetB, config.datarootB,
                                            config.original_image_size, config.image_size,
                                            config.batch_size, config.workers, split='train')

    val_loader_B, val_set_B = get_loader(config.datasetB, config.valDatarootB,
                                         config.original_image_size, config.image_size,
                                         config.val_batch_size, config.workers, split='test')
                                         # transform_fn=google_cartoon_tns)

    print("[*] Prepare dataloader completed!!!")

    if config.mode == "train":
        trainer = Trainer(config, train_loader_A, train_set_A, val_loader_A, val_set_A,
                          train_loader_B, train_set_B, val_loader_B, val_set_B)
        trainer.train()
    elif config.mode == "test":
        tester = Tester(config, val_loader_A, val_set_A, val_loader_B, val_set_B)
        tester.test()


if __name__ == "__main__":
    config = get_config()
    main(config)