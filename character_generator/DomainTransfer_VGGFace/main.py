from __future__ import print_function
import random
import os

import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True

from trainer import Trainer
from config import get_config
from dataloader import get_loader
# from prepare_dataset import split_train_test

def main(config):
    # make directories
    if config.sample_folder is None:
        config.sample_folder = 'samples'
    os.system('mkdir {0}'.format(config.sample_folder))
    print("Make sample folder!")

    if config.ckpt_folder is None:
        config.ckpt_folder = 'checkpoints'
    os.system('mkdir {0}'.format(config.ckpt_folder))
    print("Make checkpoints folder!")

    config.manual_seed = random.randint(1, 10000)
    print("Random Seed: ", config.manual_seed)

    random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)
    torch.cuda.manual_seed_all(config.manual_seed)

    # for faster training
    cudnn.benchmark = True

    # setup dataloader
    print("[*] Preparing dataloader ...")
    train_loader_A, train_set_A = get_loader(config.datasetA, config.datarootA,
                                       config.original_image_size, config.image_size,
                                       config.batch_size, config.workers, split='train')

    val_loader_A, val_set_A = get_loader(config.datasetA, config.valDatarootA,
                                       config.original_image_size, config.image_size,
                                       config.val_batch_size, config.workers, split='test')

    train_loader_B, train_set_B = get_loader(config.datasetB, config.datarootB,
                                            config.original_image_size, config.image_size,
                                            config.batch_size, config.workers, split='train')

    val_loader_B, val_set_B = get_loader(config.datasetB, config.valDatarootB,
                                         config.original_image_size, config.image_size,
                                         config.val_batch_size, config.workers, split='test')

    print("[*] Prepare dataloader completed!!!")

    trainer = Trainer(config, train_loader_A, train_set_A, val_loader_A, val_set_A,
                      train_loader_B, train_set_B, val_loader_B, val_set_B)
    trainer.train()

if __name__ == "__main__":
    config = get_config()
    main(config)