import torch
import torch.nn as nn
import torch.nn.functional as F

def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, dropout=False):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
    else:
        block.add_module('%s_tconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout:
        block.add_module('%s_dropout' % name, nn.Dropout2d(0.5, inplace=True))
    return block

# input: 224
class Discriminator(nn.Module):
    def __init__(self, in_ch, out_ch, ndf=64):
        super(Discriminator, self).__init__()

        # input is 224
        layer_idx = 1
        name = 'layer%d' % layer_idx
        layer1 = nn.Sequential()
        layer1.add_module(name, nn.Conv2d(in_ch, ndf, 4, 4, 1, bias=False))

        # input is 56
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer2 = blockUNet(ndf, ndf*2, name, transposed=False, bn=True, relu=False, dropout=False)

        # input is 28
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer3 = blockUNet(ndf*2, ndf*4, name, transposed=False, bn=True, relu=False, dropout=False)

        # input: 14
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer4 = blockUNet(ndf*4, ndf*8, name, transposed=False, bn=True, relu=False, dropout=False)

        # input: 7
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer5 = nn.Sequential()
        layer5.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
        layer5.add_module('%s_conv' % name, nn.Conv2d(ndf*8, ndf*4, 4, 1, 0, bias=False))

        # input:4
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer6 = nn.Sequential()
        layer6.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
        layer6.add_module('%s_conv' % name, nn.Conv2d(ndf*4, ndf*2, 4, 1, 0, bias=False))

        # input is 1
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer7 = nn.Sequential()
        layer7.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
        layer7.add_module('%s_conv' % name, nn.Conv2d(ndf*2, out_ch, 1, 1, 0, bias=False))
        # final output is 1

        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.layer5 = layer5
        self.layer6 = layer6
        self.layer7 = layer7

    def forward(self, x):
        # 1, 3, 224, 224
        out1 = self.layer1(x)
        # 1, 256, 112, 112
        out2 = self.layer2(out1)
        # 1, 512, 56, 56
        out3 = self.layer3(out2)
        # 1, 1024, 28, 28
        out4 = self.layer4(out3)
        # 1, 2048, 14, 14
        out5 = self.layer5(out4)
        # 1, 4096, 7, 7
        out6 = self.layer6(out5)
        # 1, 2048, 4, 4
        out7 = self.layer7(out6)
        out = out7.squeeze(3).squeeze(2)
        # 1, 3
        return out

# generate fake images
# input: 1
class Generator(nn.Module):
    def __init__(self, out_ch, ngf=64):
        super(Generator, self).__init__()

        # input is 7
        layer_idx = 1
        name = 'dlayer%d' % layer_idx
        dlayer1 = nn.Sequential()
        dlayer1.add_module('%s_conv' % name, nn.ConvTranspose2d(ngf*2, ngf*16, 3, 1, 1, bias=False))

        # input is 7
        layer_idx += 1
        name = 'dlayer%d' % layer_idx
        dlayer2 = blockUNet(ngf*16, ngf*8, name, transposed=True, bn=True, relu=False, dropout=False)

        # input is 14
        layer_idx += 1
        name = 'dlayer%d' % layer_idx
        dlayer3 = blockUNet(ngf*8, ngf*4, name, transposed=True, bn=True, relu=False, dropout=False)

        # input 28
        layer_idx += 1
        name = 'dlayer%d' % layer_idx
        dlayer4 = blockUNet(ngf*4, ngf*2, name, transposed=True, bn=True, relu=False, dropout=False)

        # input 56
        layer_idx += 1
        name = 'dlayer%d' % layer_idx
        dlayer5 = blockUNet(ngf*2, ngf, name, transposed=True, bn=True, relu=False, dropout=False)

        # input 112
        layer_idx += 1
        name = 'dlayer%d' % layer_idx
        dlayer6 = nn.Sequential()
        dlayer6.add_module('%s_relu' % name, nn.LeakyReLU(0.2, inplace=True))
        dlayer6.add_module('%s_tconv' % name, nn.ConvTranspose2d(ngf, out_ch, 4, 2, 1, bias=False))
        dlayer6.add_module('%s_tanh' % name, nn.Tanh())

        self.dlayer1 = dlayer1
        self.dlayer2 = dlayer2
        self.dlayer3 = dlayer3
        self.dlayer4 = dlayer4
        self.dlayer5 = dlayer5
        self.dlayer6 = dlayer6

    def forward(self, x):
        # 1, 512, 7, 7
        dout1 = self.dlayer1(x)

        # print(dout1.shape)

        # 1, 2048, 14, 14
        dout2 = self.dlayer2(dout1)
        # 1, 1024, 28, 28
        dout3 = self.dlayer3(dout2)
        # 1, 512, 56, 56
        dout4 = self.dlayer4(dout3)
        # 1, 256, 112, 112
        dout5 = self.dlayer5(dout4)
        # 1, 3, 224, 224
        dout6 = self.dlayer6(dout5)
        return dout6