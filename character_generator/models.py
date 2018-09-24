import torch
import torch.nn as nn
import torch.nn.parallel

# resnet block
class ResidualBlock(nn.Module):
    def __init__(self, channel, kernel, stride, padding):
        super(ResidualBlock, self).__init__()
        self.padding = padding
        self.main = nn.Sequential(
            nn.ReflectionPad2d(self.padding),
            nn.Conv2d(channel, channel, kernel, stride, 0),
            nn.InstanceNorm2d(channel),
            nn.ReLU(True),

            nn.ReflectionPad2d(self.padding),
            nn.Conv2d(channel, channel, kernel, stride, 0),
            nn.InstanceNorm2d(channel))

    def forward(self, input):
        x = self.main(input)

        return input + x


class Generator(nn.Module):
    def __init__(self, ngpu, ngf, input_nc, output_nc, nblocks=9):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.ngf = ngf
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.nblocks = nblocks

        # Encoding layers
        self.conv = nn.Sequential(
            # input: nc, 256, 256
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7, 1, 0, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            # state output: ngf, 256, 256

            nn.Conv2d(ngf, ngf * 2, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(ngf *2 ),
            nn.ReLU(True),
            # state output: ngf, 128, 128

            nn.Conv2d(ngf*2, ngf*4, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(ngf*4),
            nn.ReLU(True),
            # state output: ngf*4, 64, 64
        )

        # Resnet layers
        self.resnet = []
        for i in range(nblocks):
            self.resnet.append(ResidualBlock(ngf*4, 3, 1, 1))
        self.resnet = nn.Sequential(*self.resnet)
        # state output: ngf*4, 64, 64

        # Decoding layers
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1, bias=False),
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(True),
            # state output: ngf*2, 128, 128

            nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            # state output: ngf, 256, 256

            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7, 1, 0),
            nn.Tanh()
            # final output: nc, 256, 256
        )

    def main(self, input):
        x = self.conv(input)
        x = self.resnet(x)
        x = self.deconv(x)

        return x

    def forward(self, input):
        # on multi-gpu env
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # on single-gpu env
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf, input_nc):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.main = nn.Sequential(
            # input: nc, 256, 256
            nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state output: ndf, 128, 128

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state output: ndf*2, 64, 64

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state output: ndf*4, 32, 32

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state output: ndf*8, 16, 16

            nn.Conv2d(ndf * 8, ndf * 8, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state output: ndf*8, 15, 15

            nn.Conv2d(ndf * 8, 1, 4, 1, 1, bias=False),
            # final output: 1, 14, 14
        )

    def forward(self, input):
        # on multi-gpu env
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # on single-gpu env
        else:
            output = self.main(input)

        return output



