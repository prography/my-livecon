import torch
import torchvision.utils as vutils

import os

from model import Generator, Discriminator
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

        self.input_nc = config.n_in
        self.output_nc = config.n_out

        self.test_result_folder = config.test_result_folder

        self.load_models()

        self.dataloader = get_test_loader(config)

    def load_models(self):
        if not os.path.exists(self.model_path):
            print("Model path doesn't exist!")

        print("[*] Loading model...!")
        self.netG_AB = Generator(self.input_nc, self.output_nc)
        self.netG_AB.load_state_dict(torch.load(self.model_path, map_location=lambda storage, loc: storage))
        self.netG_AB = self.netG_AB.to(device)
        self.netG_AB.eval()

    def test(self, testnum):
        if not os.path.exists(self.test_result_folder):
            os.makedirs(self.test_result_folder)

        for idx, (realA, cls) in enumerate(self.dataloader):
            save_cnt = 0
            if cls == 0 and save_cnt < testnum:
                realA = realA.to(device)
                fakeB = self.netG_AB(realA)

                vutils.save_image(denorm(fakeB.data), os.path.join(self.test_result_folder, "test_result_%d.png" % idx))
                vutils.save_image(denorm(realA.data), os.path.join(self.test_result_folder, "test_input_%d.png" % idx))

                save_cnt += 1
                print("Saving %d image..!" % save_cnt)

            elif save_cnt >= testnum:
                break

        print("Test and save complete!!")


if __name__ == "__main__":
    config = get_config()
    tester = Tester(config)
    tester.test(testnum=5)

