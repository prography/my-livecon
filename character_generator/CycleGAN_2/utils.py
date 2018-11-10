import torch.nn as nn
import torch
import random
import matplotlib.pyplot as plt

# initialize network layers' weight and bias
def weights_init_normal(module):
    classname = module.__class__.__name__
    # if input parameter module is Convolutional layer
    if classname.find('Conv') != -1:
        nn.init.normal(module.weight.data, 0.0, 0.02)

    # if input parameter module is Batch Normalization layer
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal(module.weight.data, 1.0, 0.02)
        nn.init.constant(module.bias.data, 0.0)

# decay learning rate
class LambdaLR():
    def __init__(self, num_epochs, offset, decay_start_epoch):
        assert ((num_epochs - decay_start_epoch) > 0), \
            "Decay must start before the training session ends!"

        self.num_epochs = num_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.num_epochs - self.decay_start_epoch)

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), \
            'Empty buffer or trying to create a black hole. Be careful.'

        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)

            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)

            # if buffer is full
            else:
                # randomly choose data to replace
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)

        return torch.cat(to_return)

# images: list of images
# labels: list of labels
def save_image(images, labels, outpath):
    fig = plt.figure()

    for i, d in enumerate(images):
        d = d.squeeze()
        im = d.data.cpu().numpy()
        im = (im.transpose(1, 2, 0) + 1) / 2

        f = fig.add_subplot(2, 3, i + 1)
        f.imshow(im)
        f.set_title(labels[i])
        f.set_xticks([])
        f.set_yticks([])

    plt.savefig(outpath)