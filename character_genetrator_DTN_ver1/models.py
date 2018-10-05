import torch.nn as nn

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

# input: batch of images
# discriminate whether input image is real or fake
class Discriminator(nn.Module):
    def __init__(self, in_ch, out_ch, ndf=64):
        super(Discriminator, self).__init__()

        # input is 32
        layer_idx = 1
        name = 'layer%d' % layer_idx
        layer1 = nn.Sequential()
        layer1.add_module(name, nn.Conv2d(in_ch, ndf, 4, 2, 1, bias=False))

        # input is 16
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer2 = blockUNet(ndf, ndf*2, name, transposed=False, bn=True, relu=False, dropout=False)

        # input is 8
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer3 = blockUNet(ndf*2, ndf*4, name, transposed=False, bn=True, relu=False, dropout=False)

        # input is 4
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer4 = nn.Sequential()
        layer4.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
        layer4.add_module('%s_conv' % name, nn.Conv2d(ndf*4, ndf*2, 4, 1, 0, bias=False))
        # input is 1

        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer5 = nn.Sequential()
        layer5.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
        layer5.add_module('%s_conv' % name, nn.Conv2d(ndf*2, out_ch, 1, 1, 0, bias=False))
        # final output is 1

        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.layer5 = layer5

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        return out5.squeeze(3).squeeze(2) # final output: [batch, out_ch]

# input: batch of images
# out5: batch, classes, 1, 1
# out4: batch, ngf*2, 1, 1
class Encoder(nn.Module):
    def __init__(self, in_ch, ngf, num_classes):
        super(Encoder, self).__init__()

        # input is 32
        layer_idx = 1
        name = 'layer%d' % layer_idx
        layer1 = nn.Sequential()
        layer1.add_module(name, nn.Conv2d(in_ch, ngf, 4, 2, 1, bias=False))

        # input is 16
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer2 = blockUNet(ngf, ngf*2, name, transposed=False, bn=True, relu=False, dropout=False)

        # input is 8
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer3 = blockUNet(ngf*2, ngf*4, name, transposed=False, bn=True, relu=False, dropout=False)

        # input is 4
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer4 = nn.Sequential()
        layer4.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
        layer4.add_module('%s_conv' % name, nn.Conv2d(ngf*4, ngf*2, 4, 1, 0, bias=False))

        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer5 = nn.Sequential()
        layer5.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
        layer5.add_module('%s_conv' % name, nn.Conv2d(ngf*2, num_classes, 1, 1, 0, bias=False))

        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.layer5 = layer5

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        return out5, out4

# generate fake images
# input:
class Generator(nn.Module):
    def __init__(self, out_ch, ngf=64):
        super(Generator, self).__init__()

        # input is 1
        layer_idx = 4
        name = 'dlayer%d' % layer_idx

        dlayer4 = nn.Sequential()
        dlayer4.add_module('%s_conv' % name, nn.ConvTranspose2d(ngf*2, ngf*4, 4, 1, 0, bias=False))

        # input is 4
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer3 = blockUNet(ngf*4, ngf*2, name, transposed=True, bn=True, relu=True, dropout=False)

        # input is 8
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer2 = blockUNet(ngf*2, ngf, name, transposed=True, bn=True, relu=True, dropout=False)

        # input is 16
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer1 = nn.Sequential()
        dlayer1.add_module('%s_relu' % name, nn.ReLU(inplace=True))
        dlayer1.add_module('%s_tconv' % name, nn.ConvTranspose2d(ngf, out_ch, 4, 2, 1, bias=False))
        dlayer1.add_module('%s_tanh' % name, nn.Tanh())

        self.dlayer4 = dlayer4
        self.dlayer3 = dlayer3
        self.dlayer2 = dlayer2
        self.dlayer1 = dlayer1

    def forward(self, dout5):
        dout4 = self.dlayer4(dout5)
        dout3 = self.dlayer3(dout4)
        dout2 = self.dlayer2(dout3)
        dout1 = self.dlayer1(dout2)
        return dout1