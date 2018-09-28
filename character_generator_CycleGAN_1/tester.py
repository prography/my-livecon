import torch
import torchvision.utils as vutils

import os

from models import Generator, Discriminator
from dataloader import get_test_loader
from config import get_config

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# denormalization
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

class Tester(object):
    def __init__(self, config):
        self.model_path = config.model_path
        self.config = config

        self.ngpu = config.ngpu
        self.ngf = config.ngf
        self.input_nc = config.input_nc
        self.output_nc = config.output_nc

        self.test_result_folder = config.test_result_folder

        self.load_models()

        self.a_loader, _ = get_test_loader(config.dataroot, config.batch_size, config.image_size, config.split_ratio,
                                                        num_workers=int(config.workers))

    def load_models(self):
        if not os.path.exists(self.model_path):
            print("Model path doesn't exist!")

        self.netG_AB = Generator(self.ngpu, self.ngf, self.input_nc, self.output_nc)
        self.netG_AB.load_state_dict(torch.load(self.model_path, map_location=lambda storage, loc: storage))
        self.netG_AB = self.netG_AB.to(device)
        self.netG_AB.eval()

    def test(self, testnum):
        if not os.path.exists(self.test_result_folder):
            os.makedirs(self.test_result_folder)

        A_loader = iter(self.a_loader)

        for idx in range(testnum):
            try:
                realA = A_loader.next()
            except StopIteration:
                A_loader = iter(self.a_loader)
                realA = A_loader.next()

            realA = realA.to(device)
            fakeB = self.netG_AB(realA)

            vutils.save_image(denorm(fakeB.data), os.path.join(self.test_result_folder, "test_result_%d.png" % idx))
            vutils.save_image(denorm(realA.data), os.path.join(self.test_result_folder, "test_input_%d.png" % idx))

        print("Test and save complete!!")


if __name__ == "__main__":
    config = get_config()
    tester = Tester(config)
    tester.test(testnum=5)

