import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable

from models.dtn import Generator, Discriminator
from models.encoder import vgg13
from vis_tool import Visualizer
import misc

import numpy as np
import collections, h5py

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trainer(object):
    def __init__(self, config,
                 train_loader_A, train_set_A, val_loader_A, val_set_A,
                 train_loader_B, train_set_B, val_loader_B, val_set_B):

        self.config = config

        self.train_loader_A = train_loader_A
        self.train_set_A = train_set_A
        self.val_loader_A = val_loader_A
        self.val_set_A = val_set_A

        self.train_loader_B = train_loader_B
        self.train_set_B = train_set_B
        self.val_loader_B = val_loader_B
        self.val_set_B = val_set_B

        print("[*] source: %s target: %s" % (self.train_set_A, self.train_set_B))

        self.batch_size = config.batch_size
        self.val_batch_size = config.val_batch_size
        self.image_size = config.image_size

        self.ngf = config.ngf
        self.ndf = config.ndf
        self.in_ch = config.in_ch
        self.out_ch = config.out_ch
        self.num_classes = config.num_classes

        self.rmsprop = config.rmsprop
        self.num_epochs = config.num_epochs
        self.lrG = config.lrG
        self.lrD = config.lrD

        self.beta_TID = config.betaTID
        self.alpha_CONST = config.alphaCONST
        self.crossentropy = config.crossentropy
        self.wd = config.wd
        self.beta1 = config.beta1

        self.log_interval = config.log_interval
        self.sample_interval = config.sample_interval
        self.ckpt_interval = config.ckpt_interval

        self.sample_folder = config.sample_folder
        self.ckpt_folder = config.ckpt_folder

        self.build_networks()

    def build_networks(self):
        print("[*] Build Generator model...")

        self.netG = Generator(self.out_ch, self.ngf).to(device)
        self.netG.apply(misc.weights_init)

        if self.config.netG != '':
            print("[*] Loading trained Generator's weights...")
            self.netG.load_state_dict(torch.load(self.config.netG))

        print("[*] Build Generator model completed!")

        print("[*] Build Encoder model...")

        # load pre-trained VGG DeepFace model as encoder
        self.netE = vgg13()

        self.netE.features = torch.nn.Sequential(
            collections.OrderedDict(
                zip(
            ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
             'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2',
             'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
             'pool5'], self.netE.features))
            )
        self.netE.classifier = torch.nn.Sequential(collections.OrderedDict(
            zip(['fc6', 'relu6', 'drop6', 'fc7', 'relu7', 'drop7', 'fc8', 'prob'], self.netE.classifier)))

        if self.config.netE != "":
            print("[*] Loading trained Encoder's weight...")
            state_dict = h5py.File(self.config.netE, 'r')
            self.netE.load_state_dict(
                {l: torch.from_numpy(np.array(v)).view_as(p) for k, v in state_dict.items() for l, p in
                 self.netE.named_parameters() if k in l})

            self.netE = self.netE.to(device)
            print("[*] Build Encoder model completed!")
        else:
            print("[*] Error! Provide trained encoder's weight path!")
            return

        print("[*] Build Discriminator model...")
        self.netD = Discriminator(self.in_ch, self.out_ch, self.ndf).to(device)
        self.netD.apply(misc.weights_init)

        if self.config.netD != '':
            print("[*] Loading trained Discriminator's weights...")
            self.netD.load_state_dict(torch.load(self.config.netD))

        print("[*] Build Discriminator model completed!")

        print("netG:", self.netG)
        print("netE:", self.netE)
        print("netD:", self.netD)

        self.netG.train()
        self.netE.eval() # freeze encoder
        self.netD.train()

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def train(self):
        # setup visdom
        vis = Visualizer()

        # setup criterions
        criterion_CE = nn.CrossEntropyLoss().to(device)
        criterion_CAE = nn.MSELoss().to(device)

        # setup optimizers
        if self.rmsprop:
            optimizer_D = optim.SGD(self.netD.parameters(), lr=self.lrD)
            optimizer_G = optim.RMSprop(self.netG.parameters(), lr=self.lrG)
        else:
            optimizer_D = optim.SGD(self.netD.parameters(), lr=self.lrD)
            optimizer_G = optim.Adam(self.netG.parameters(), lr=self.lrG, betas=(self.beta1, 0.999), weight_decay=0.0)

        # setup input, output tensors
        source = torch.FloatTensor(self.batch_size, self.in_ch, self.image_size, self.image_size)
        target = torch.FloatTensor(self.batch_size, self.out_ch, self.image_size, self.image_size)

        val_source = torch.FloatTensor(self.val_batch_size, self.in_ch, self.image_size, self.image_size)
        val_target = torch.FloatTensor(self.val_batch_size, self.out_ch, self.image_size, self.image_size)

        source, target = source.to(device), target.to(device)
        val_source, val_target = val_source.to(device), val_target.to(device)

        label_d = torch.LongTensor(self.batch_size).to(device)
        label_c = torch.LongTensor(self.batch_size).to(device)

        fake_source_label = 0
        fake_target_label = 1
        real_target_label = 2

        # save validating images
        val_source_iter = iter(self.val_loader_A)
        val_target_iter = iter(self.val_loader_B)

        val_source_item, _ = next(val_source_iter)
        val_target_item, _ = next(val_target_iter)

        val_source_item = val_source_item.to(device)
        val_target_item = val_target_item.to(device)

        val_source.resize_as_(val_source_item).copy_(val_source_item)
        val_target.resize_as_(val_target_item).copy_(val_target_item)

        vutils.save_image(val_source, "%s/samples_real_source.png" % self.sample_folder,
                          nrow=6, normalize=True)
        vutils.save_image(val_target, "%s/samples_real_target.png" % self.sample_folder,
                          nrow=6, normalize=True)

        # freezing encoder
        for p in self.netE.parameters():
            p.requires_grad = False

        gan_iters = 0

        lossD_list = []
        lossG_list = []

        print("Learning started...!")

        for epoch in range(self.num_epochs):
            source_iter = iter(self.train_loader_A)
            target_iter = iter(self.train_loader_B)

            source_idx, target_idx = 0, 0

            while source_idx<len(source_iter) and target_idx<len(target_iter):
                # SVHN --> MNIST
                if self.config.datasetB == 'mnist':
                    target_item_1c, _ = next(target_iter)
                    source_item, source_label = next(source_iter)
                    source_label -= 1

                    target_item = torch.FloatTensor(target_item_1c.size(0),
                                                   self.out_ch,
                                                   target_item_1c.size(2),
                                                   target_item_1c.size(3))

                    target_item[:, 0, :, :].unsqueeze_(1).copy_(target_item_1c)
                    target_item[:, 1, :, :].unsqueeze_(1).copy_(target_item_1c)
                    target_item[:, 2, :, :].unsqueeze_(1).copy_(target_item_1c)

                # MNIST --> SVHN
                elif self.config.datasetA == 'mnist':
                    target_item, _ = next(target_iter)
                    source_item_1c, source_label = next(source_iter)
                    source_item = torch.FloatTensor(source_item_1c.size(0),
                                                    self.in_ch,
                                                    source_item_1c.size(2),
                                                    source_item_1c.size(3))

                    source_item[:, 0, :, :].unsqueeze_(1).copy_(source_item_1c)
                    source_item[:, 1, :, :].unsqueeze_(1).copy_(source_item_1c)
                    source_item[:, 2, :, :].unsqueeze_(1).copy_(source_item_1c)

                # our case
                else:
                    target_item, _ = next(target_iter)
                    source_item, source_label = next(source_iter)

                trg_mini_batch = target_item.size(0)
                src_mini_batch = source_item.size(0)
                source_idx += 1
                target_idx += 1

                if trg_mini_batch != src_mini_batch:
                    print("[*] Dataset finished.. load again!")
                    continue

                target_item = target_item.to(device)
                source_item = source_item.to(device)

                target.data.resize_as_(target_item).copy_(target_item)
                source.data.resize_as_(source_item).copy_(source_item)

                #==========================================#
                #        1. train the Discriminator        #
                #==========================================#
                for p in self.netD.parameters():
                    p.requires_grad = True

                self.netD.zero_grad()

                # compute error with real target
                # real target label = 2
                label_d.data.resize_(trg_mini_batch).fill_(real_target_label) # 1
                outD_real_target = self.netD(target) # 1, 3

                errD_real_target = criterion_CE(outD_real_target, label_d)
                errD_real_target.backward()

                # compute error with fake target
                _, h_target = self.netE(target)

                x_hat_target = self.netG(h_target)
                fake_target = x_hat_target.detach()

                # fake target label = 1
                label_d.data.resize_(trg_mini_batch).fill_(fake_target_label)
                outD_fake_target = self.netD(fake_target)
                errD_fake_target = criterion_CE(outD_fake_target, label_d)
                errD_fake_target.backward()

                # compute error with fake source
                _, h_source = self.netE(source)
                x_hat_source = self.netG(h_source)
                fake_source = x_hat_source.detach()

                # fake source label = 2
                label_d.data.resize_(src_mini_batch).fill_(fake_source_label)
                outD_fake_source = self.netD(fake_source)
                errD_fake_source = criterion_CE(outD_fake_source, label_d)
                errD_fake_source.backward()

                lossD = errD_real_target + errD_fake_target + errD_fake_source
                lossD_list.append(lossD.item())
                optimizer_D.step()

                # ==========================================#
                #            2. train the Generator         #
                # ==========================================#
                for p in self.netD.parameters():
                    p.requires_grad = False

                self.netG.zero_grad()

                if self.crossentropy:
                    label_c.data.resize_(self.batch_size).copy_(source_label)

                    # encoding fake source
                    pred_x_hat_source, _ = self.netE(x_hat_source)
                    L_const = criterion_CE(pred_x_hat_source.squeeze(3).squeeze(2), label_c)

                else:
                    _, output_h_source = self.netE(x_hat_source)
                    L_const = criterion_CAE(Variable(output_h_source, requires_grad=True), h_source.detach())

                L_const = L_const * self.alpha_CONST

                if self.alpha_CONST != 0:
                    L_const.backward(retain_graph=True)

                L_tid = criterion_CAE(x_hat_target, target)
                L_tid = self.beta_TID * L_tid

                if self.beta_TID != 0:
                    L_tid.backward(retain_graph=True)

                # compute loss with real target label, fake source input
                label_d.data.fill_(real_target_label)

                outD_fake_source = self.netD(x_hat_source)
                errG_fake_source = criterion_CE(Variable(outD_fake_source, requires_grad=True), label_d)
                errG_fake_source.backward()

                outD_fake_target = self.netD(x_hat_target)
                errG_fake_target = criterion_CE(Variable(outD_fake_target, requires_grad=True), label_d)
                errG_fake_target.backward()

                lossG = errG_fake_target + errG_fake_source
                lossG_list.append(lossG.item())

                optimizer_G.step()
                gan_iters += 1

                # do logging
                if gan_iters % self.log_interval == 0:
                    print("[%d/%d] [%d/%d] [%d/%d] G loss:%.3f D loss:%.3f"
                          % (epoch+1, self.num_epochs, source_idx+1, len(self.train_loader_A),
                             target_idx+1, len(self.train_loader_B),
                             np.mean(lossG_list), np.mean(lossD_list)))

                    vis.plot("[DTN_vggFace] G loss with lr=%.4f" % self.lrG, np.mean(lossG_list))
                    vis.plot("[DTN_vggFace] D loss with lr=%.4f" % self.lrD, np.mean(lossD_list))

                    lossG_list.clear()
                    lossD_list.clear()

                # generating images
                if gan_iters % self.sample_interval == 0:
                    val_batch_output = torch.FloatTensor(val_source.size(0),
                                                         self.in_ch,
                                                         val_source.size(2),
                                                         val_source.size(3)).fill_(0)
                    for idx in range(val_source.size(0)):
                        if self.config.datasetA == 'mnist':
                            single_img = torch.FloatTensor(1, self.in_ch, val_source.size(2), val_source.size(3))
                            single_img[0, 0, :, :].unsqueeze_(1).copy_(val_source[idx, :, :, :])
                            single_img[0, 1, :, :].unsqueeze_(1).copy_(val_source[idx, :, :, :])
                            single_img[0, 2, :, :].unsqueeze_(1).copy_(val_source[idx, :, :, :])
                        else:
                            single_img = val_source[idx, :, :, :].unsqueeze(0)

                        _, h_val = self.netE(single_img)
                        x_hat_val = self.netG(h_val)
                        val_batch_output[idx, :, :, :].unsqueeze_(0).copy_(x_hat_val.data)

                    vutils.save_image(val_batch_output, '%s/generated_epoch%04d_iter%08d.png'
                                        % (self.sample_folder, epoch+1, gan_iters), nrow=6, normalize=True)
                    print("[*] Saving sample images completed!")

                    vis.img("[vggFace] Generated image", self.denorm(val_batch_output))

                # do checkpointing
                if gan_iters % self.ckpt_interval == 0:
                    torch.save(self.netG.state_dict(), "%s/netG_epoch%04d_iter%08d.pth"
                               % (self.ckpt_folder, epoch+1, gan_iters))
                    torch.save(self.netD.state_dict(), "%s/netD_epoch%04d_iter%08d.pth"
                               % (self.ckpt_folder, epoch + 1, gan_iters))
                    print("[*] Saving checkpoints completed!")

        print("Learning finished!")
        torch.save(self.netG.state_dict(), "%s/final_netG.pth" % self.ckpt_folder)
        torch.save(self.netD.state_dict(), "%s/final_netD.pth" % self.ckpt_folder)
        print("[*] Saving final checkpoints completed!")

        vis.plot("Domain Transfer Network training finished!", 1)