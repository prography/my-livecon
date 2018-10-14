import torch
import torch.nn as nn
import torchvision.utils as vutils

from models.dtn import Generator
from models.encoder import vgg13

import collections, h5py, os
import numpy as np

class Tester(object):
    def __init__(self, config, valDataloaderA, valDatasetA, valDataloaderB, valDatasetB):
        self.config = config
        self.testEncoder = config.netE
        self.testGenerator = config.testGenerator
        self.testFolder = config.testFolder

        self.inCh = config.in_ch
        self.outCh = config.out_ch
        self.ngf = config.ngf

        self.valDataloaderA = valDataloaderA
        self.valDatasetA = valDatasetA
        self.valDataloaderB = valDataloaderB
        self.valDatasetB = valDatasetB

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.load_models()

    def load_models(self):
        print("[*] Loading Encoder from {}...".format(self.testEncoder))
        self.netE = vgg13()
        self.netE.features = torch.nn.Sequential(
            collections.OrderedDict(
                zip(
                    ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2',
                     'pool2',
                     'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1', 'relu4_1',
                     'conv4_2',
                     'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
                     'relu5_3',
                     'pool5'], self.netE.features))
        )
        self.netE.classifier = torch.nn.Sequential(collections.OrderedDict(
            zip(['fc6', 'relu6', 'drop6', 'fc7', 'relu7', 'drop7', 'fc8', 'prob'], self.netE.classifier)))


        state_dict = h5py.File(self.testEncoder, 'r')
        self.netE.load_state_dict(
            {l: torch.from_numpy(np.array(v)).view_as(p) for k, v in state_dict.items() for l, p in
             self.netE.named_parameters() if k in l})

        self.netE = self.netE.to(self.device)
        print("[*] Load Encoder completed!")

        print("[*] Loading Generator from {}...".format(self.testGenerator))
        self.netG = Generator(self.outCh, self.ngf)
        self.netG.load_state_dict(torch.load(self.testGenerator, map_location=lambda storage, loc: storage))
        self.netG = self.netG.to(self.device)
        print("[*] Load Generator completed!")


    def test(self):
        # switch models to evaluation mode
        self.netE.eval()
        self.netG.eval()

        # save validating images
        valIterA = iter(self.valDataloaderA)
        valIterB = iter(self.valDataloaderB)

        valItemA, _ = next(valIterA)
        valItemB, _ = next(valIterB)

        valItemA = valItemA.to(self.device)
        valItemB = valItemB.to(self.device)

        vutils.save_image(valItemA, os.path.join(self.testFolder, "test_realSource.png"),
                          nrow=6, normalize=True)
        vutils.save_image(valItemB, os.path.join(self.testFolder, "test_realTarget.png"),
                          nrow=6, normalize=True)

        print("Generating test results...")
        with torch.no_grad():
            valOutput = torch.FloatTensor(valItemA.size(0),
                                          self.inCh,
                                          valItemA.size(2),
                                          valItemA.size(3)).fill_(0)

            for idx in range(valItemA.size(0)):
                singleImg = valItemA[idx, :, :, :].unsqueeze(0)

                _, hVal = self.netE(singleImg)
                xHatVal = self.netG(hVal)
                valOutput[idx, :, :, :].unsqueeze_(0).copy_(xHatVal.data)

            resultPath = os.path.join(self.testFolder, "results.png")
            vutils.save_image(valOutput, resultPath, nrow=6, normalize=True)
            print("[*] Save test result images to {}!".format(resultPath))