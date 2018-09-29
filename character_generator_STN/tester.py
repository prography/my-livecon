import os
import torch
from torchvision import transforms
import torchvision.utils as vutils

from model.nst import TransferNet
from utils import *
from dataloader import get_test_loader
from config import get_config

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Tester(object):
    def __init__(self, config):
        self.model_path = config.model_path
        self.test_result_path = config.test_result_folder

        if not os.path.exists(self.test_result_path):
            os.makedirs(self.test_result_path)

        self.load_net()
        self.test_loader = get_test_loader(config)

    def load_net(self):
        model = TransferNet()
        print("[*] Loading model...!")
        model.load_state_dict(torch.load(self.model_path, map_location=lambda storage, loc: storage))
        self.model = model.to(device)
        self.model.eval()

    def test(self, test_num):

        test_iter = iter(self.test_loader)
        save_cnt = 0

        while(save_cnt < test_num):
            img, cls = next(test_iter)
            # print(next(test_iter))
            if cls == 0:
                output = self.model(img)
                output = output.div(255.0).clamp(0, 1)

                img = img.div(255.0).clamp(0, 1)

                vutils.save_image(img, "%s/input_%d.jpg" % (self.test_result_path, save_cnt))
                vutils.save_image(output, "%s/output_%d.jpg" % (self.test_result_path, save_cnt))
                print("saving output image %d completed!" % save_cnt)

                save_cnt += 1



        # image = load_image(self.image_path)
        # image = self.transform(image)
        # image = image.unsqueeze(0).to(device)
        #
        # output = self.model(image)
        # output = output.div(255.0).clamp(0, 1)
        # file = "%s/output%d.jpg" % (self.out_folder, self.image_num)
        # vutils.save_image(output, file)
        # print("saving output image completed!")


if __name__ == '__main__':
    config = get_config()
    tester = Tester(config)
    tester.test(5)


