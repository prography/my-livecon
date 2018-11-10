import itertools
import numpy as np

import time, os
import torch
import torch.nn as nn
import torch.optim as optim

from model import Generator, Discriminator
import utils
from vis_tool import Visualizer

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# denormalization image
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

class Trainer(object):
    def __init__(self, config, dataloader):
        self.config = config
        self.dataloader = dataloader

        self.starting_epoch = config.starting_epoch
        self.n_epochs = config.num_epochs
        self.lr = config.lr
        self.decay_epoch = config.decay_epoch

        self.log_interval = config.log_interval
        self.sample_interval = config.sample_interval
        self.ckpt_interval = config.ckpt_interval

        self.n_in = config.n_in
        self.n_out = config.n_out

        self.sample_folder = config.sample_folder
        self.ckpt_folder = config.ckpt_folder

        self.build_net()

    def build_net(self):
        netG_A2B = Generator(self.n_in, self.n_out)
        netG_B2A = Generator(self.n_out, self.n_in)
        netD_A = Discriminator(self.n_in)
        netD_B = Discriminator(self.n_out)

        netG_A2B.apply(utils.weights_init_normal)
        netG_B2A.apply(utils.weights_init_normal)
        netD_A.apply(utils.weights_init_normal)
        netD_B.apply(utils.weights_init_normal)

        if self.config.netG_A2B != "":
            netG_A2B.load_state_dict(torch.load(self.config.netG_A2B))
            print("[*] Load model from %s!" % self.config.netG_A2B)

        if self.config.netG_B2A != "":
            netG_B2A.load_state_dict(torch.load(self.config.netG_B2A))
            print("[*] Load model from %s!" % self.config.netG_B2A)

        if self.config.netD_A != "":
            netD_A.load_state_dict(torch.load(self.config.netD_A))
            print("[*] Load model from %s!" % self.config.netD_A)

        if self.config.netD_B != "":
            netD_B.load_state_dict(torch.load(self.config.netD_B))
            print("[*] Load model from %s!" % self.config.netD_B)


        self.netG_A2B = netG_A2B.to(device)
        self.netG_B2A = netG_B2A.to(device)
        self.netD_A = netD_A.to(device)
        self.netD_B = netD_B.to(device)

    # def sample_images(self, epoch, images):
    #     for image_name, image in images.items():
    #         vutils.save_image(denorm(image), "%s/epoch%d_%s.png" % (self.sample_folder, epoch, image_name))

    def train(self):
        vis = Visualizer()

        # setup losses
        criterion_GAN = nn.MSELoss()
        criterion_cycle = nn.L1Loss()
        criterion_identity = nn.L1Loss()

        # setup optimizers
        opt_G = optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
                          lr=self.lr, betas=(0.5, 0.999))
        opt_D_A = optim.Adam(self.netD_A.parameters(), lr=self.lr, betas=(0.5, 0.999))
        opt_D_B = optim.Adam(self.netD_B.parameters(), lr=self.lr, betas=(0.5, 0.999))

        lambda_lr = utils.LambdaLR(self.n_epochs, self.starting_epoch, self.decay_epoch).step

        # setup learning rate scheduler
        lr_scheduler_G = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lambda_lr)
        lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(opt_D_A, lr_lambda=lambda_lr)
        lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(opt_D_B, lr_lambda=lambda_lr)

        # setup buffers
        fake_A_buffer = utils.ReplayBuffer()
        fake_B_buffer = utils.ReplayBuffer()

        print("Learning started!!!")
        start_time = time.time()
        for epoch in range(self.starting_epoch, self.n_epochs):
            avg_loss_G = []
            avg_loss_D = []
            avg_loss_G_GAN = []
            avg_loss_G_cycle = []
            avg_loss_G_identity = []

            for step, data in enumerate(self.dataloader):
                self.netG_A2B.train()
                self.netG_B2A.train()
                real_A = data['A'].to(device)
                real_B = data['B'].to(device)

                # skip if image has 1 channel
                if real_A.size(1) == 1 or real_B.size(1) == 1:
                    continue

                step_batch = real_A.size(0)

                target_real = torch.ones(step_batch, requires_grad=False).to(device)
                target_fake = torch.zeros(step_batch, requires_grad=False).to(device)

                # =============================================#
                #             Train Generator                  #
                # =============================================#
                for p in self.netD_A.parameters():
                    p.requires_grad = False
                for p in self.netD_B.parameters():
                    p.requires_grad = False

                opt_G.zero_grad()

                # netG_A2B(B) should be equal to B if real B is fed
                same_B = self.netG_A2B(real_B)
                loss_identity_B = criterion_identity(same_B, real_B) * 5.0

                # netG_B2A(A) should be equal to A if real A is fed
                same_A = self.netG_B2A(real_A)
                loss_identity_A = criterion_identity(same_A, real_A) * 5.0

                # compute GAN loss
                fake_B = self.netG_A2B(real_A)
                outD_from_fake = self.netD_B(fake_B)
                loss_GAN_A2B = criterion_GAN(outD_from_fake, target_real)

                fake_A = self.netG_B2A(real_B)
                outD_from_fake = self.netD_A(fake_A)
                loss_GAN_B2A = criterion_GAN(outD_from_fake, target_real)

                # compute Cycle loss
                recovered_A = self.netG_B2A(fake_B)
                loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

                recovered_B = self.netG_A2B(fake_A)
                loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

                # compute Total loss
                loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
                loss_G.backward()
                opt_G.step()

                # =============================================#
                #             Train Discriminator - A          #
                # =============================================#
                for p in self.netD_A.parameters():
                    p.requires_grad = True
                self.netD_A.train()

                opt_D_A.zero_grad()

                # compute real loss
                outD_from_real = self.netD_A(real_A)
                loss_D_from_real = criterion_GAN(outD_from_real, target_real)

                # compute from fake
                fake_A = fake_A_buffer.push_and_pop(fake_A)
                outD_from_fake = self.netD_A(fake_A.detach())
                loss_D_from_fake = criterion_GAN(outD_from_fake, target_fake)

                # compute total loss
                loss_D_A = (loss_D_from_real + loss_D_from_fake) * 0.5
                loss_D_A.backward()
                opt_D_A.step()

                # =============================================#
                #             Train Discriminator - B          #
                # =============================================#
                for p in self.netD_B.parameters():
                    p.requires_grad = True
                self.netD_B.train()

                opt_D_B.zero_grad()

                # compute real loss
                outD_from_real = self.netD_B(real_B)
                loss_D_from_real = criterion_GAN(outD_from_real, target_real)

                # compute fake loss
                fake_B = fake_B_buffer.push_and_pop(fake_B)
                outD_from_fake = self.netD_B(fake_B.detach())
                loss_D_from_fake = criterion_GAN(outD_from_fake, target_fake)

                # compute total loss
                loss_D_B = (loss_D_from_real + loss_D_from_fake) * 0.5
                loss_D_B.backward()
                opt_D_B.step()

                avg_loss_D.append((loss_D_A.item() + loss_D_B.item()) * 0.5)
                avg_loss_G_identity.append((loss_identity_A.item() + loss_identity_B.item()) * 0.5)
                avg_loss_G_cycle.append((loss_cycle_ABA.item() + loss_cycle_BAB.item()) * 0.5)
                avg_loss_G_GAN.append((loss_GAN_A2B.item() + loss_GAN_B2A.item()) * 0.5)
                avg_loss_G.append(loss_G.item())

                if (step+1) % self.log_interval == 0:
                    end_time = time.time()


                    print("[%d/%d] [%d/%d] time:%f loss_G:%.3f loss_D:%.3f"
                          % (epoch+1, self.n_epochs, step+1, len(self.dataloader), end_time-start_time,
                             np.mean(avg_loss_G), np.mean(avg_loss_D)))

                    vis.plot("ver2 loss_G", np.mean(avg_loss_G))
                    vis.plot("ver2 loss_D", np.mean(avg_loss_D))
                    vis.plot("ver2 loss_G_GAN", np.mean(avg_loss_G_GAN))
                    vis.plot("ver2 loss_G_Cycle", np.mean(avg_loss_G_cycle))
                    vis.plot("ver2 loss_G_identity", np.mean(avg_loss_G_identity))

                    avg_loss_G.clear()
                    avg_loss_D.clear()
                    avg_loss_G_GAN.clear()
                    avg_loss_G_cycle.clear()
                    avg_loss_G_identity.clear()

                if (step+1) % self.sample_interval == 0:
                    # images = {"real_A": real_A, "real_B": real_B, "fake_A": fake_A, "fake_B": fake_B}
                    # self.sample_images(epoch, images)
                    # print("Sample images saved!")
                    images = [real_A, fake_B, real_B, fake_A]
                    labels = ['real A', 'fake B', 'real B', 'fake A']
                    outpath = os.path.join(self.sample_folder, "sample_epoch{}.png".format(epoch))
                    utils.save_image(images, labels, outpath)

                if (step+1) % self.ckpt_interval == 0:
                    torch.save(self.netG_A2B.state_dict(), os.path.join(self.ckpt_folder, "netG_A2B_epoch{}.pth".format(epoch)))
                    torch.save(self.netG_B2A.state_dict(), os.path.join(self.ckpt_folder, "netG_B2A_epoch{}.pth".format(epoch)))
                    torch.save(self.netD_A.state_dict(), os.path.join(self.ckpt_folder, "netD_A_epoch{}.pth".format(epoch)))
                    torch.save(self.netD_B.state_dict(), os.path.join(self.ckpt_folder, "netD_B_epoch{}.pth".format(epoch)))
                    print("[*] Checkpoints saved!")

            lr_scheduler_G.step()
            lr_scheduler_D_A.step()
            lr_scheduler_D_B.step()

        print("Learning finished!!!")
        torch.save(self.netG_A2B.state_dict(), "%s/final_netG_A2B.pth" % self.ckpt_folder)
        torch.save(self.netG_B2A.state_dict(), "%s/final_netG_B2A.pth" % self.ckpt_folder)
        torch.save(self.netD_A.state_dict(), "%s/final_netD_A.pth" % self.ckpt_folder)
        torch.save(self.netD_B.state_dict(), "%s/final_netD_B.pth" % self.ckpt_folder)
        vis.plot("ver2 training completed!", 1)