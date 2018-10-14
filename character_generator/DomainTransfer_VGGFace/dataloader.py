import torchvision.transforms as T
from torchvision.datasets.svhn import SVHN
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_loader(dataset_name, dataroot, original_image_size, image_size, batch_size=64, workers=4, split='train', transform_fn=None):

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    transforms_list = []

    if transform_fn is None and (split == 'train' or split == 'extra'):
        transforms_list.extend([T.Resize((original_image_size, original_image_size)),
                                  T.CenterCrop(image_size),
                                  T.ToTensor(),
                                  T.Normalize(mean, std)
                                ])
        transform_fn = T.Compose(transforms_list)

    elif transform_fn is None and split == 'test':
        transforms_list.extend([T.Resize((image_size, image_size)),
                                  T.ToTensor(),
                                  T.Normalize(mean, std)
                                ])
        transform_fn = T.Compose(transforms_list)

    if dataset_name == 'svhn':
        if split == 'train': split = 'extra'
        dataset = SVHN(root=dataroot,
                       download=True,
                       split=split,
                       transform=transform_fn)

    elif dataset_name == 'mnist':
        flag_trn = split == 'train'
        dataset = MNIST(root=dataroot,
                        download=True,
                        train=flag_trn,
                        transform=transform_fn)
        
    else:
        dataset = ImageFolder(root=dataroot,
                              transform=transform_fn)

    assert dataset
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=int(workers))

    return dataloader, dataset