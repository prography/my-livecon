from __future__ import print_function
import random
import torch
import torch.backends.cudnn as cudnn
import os
from dataloader import get_loader
from config import get_config
from trainer import Trainer

def main(config):
    # sample outfolder
    if not os.path.exists('./samples'):
        os.makedirs('./samples')
    if config.sample_folder is None:
        config.sample_folder = 'samples/%s'%(config.dataset)
    if not os.path.exists(config.sample_folder):
        os.makedirs(config.sample_folder)
        print("Directory",config.sample_folder,"has created")
    else:
        print("Directory",config.sample_folder,"already exsits")

    # checkpoints outfolder
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    if config.ckpt_folder is None:
        config.ckpt_folder = 'checkpoints/%s' % (config.dataset)
    if not os.path.exists(config.ckpt_folder):
        os.makedirs(config.ckpt_folder)
        print("Directory",config.ckpt_folder,"has created")
    else:
        print("Directory",config.ckpt_folder,"already exsits")

    config.manual_seed = random.randint(1, 10000)
    print("Random Seed: ", config.manual_seed)
    random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)

    if config.cuda:
        torch.cuda.manual_seed_all(config.manual_seed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not config.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    data_lodaer = get_loader(config.dataroot, config.batch_size, config.image_size, config.split_ratio,
                            num_workers=int(config.workers))

    trainer = Trainer(config, data_lodaer)
    trainer.train()

if __name__ == "__main__":
    config = get_config()
    main(config)