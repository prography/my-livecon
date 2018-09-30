import torchvision.transforms as T
from torchvision.datasets.svhn import SVHN
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader

def get_loader(dataset_name, dataroot, original_image_size, image_size, batch_size=64, workers=4, split='train', transform_fn=None):

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    if transform_fn is None and (split == 'train' or split == 'extra'):
        transform_fn = T.Compose([T.Resize(original_image_size),
                                  T.RandomCrop(image_size),
                                  T.ToTensor(),
                                  T.Normalize(mean, std)
                                ])

    elif transform_fn is None and split == 'test':
        transform_fn = T.Compose([T.Resize(image_size),
                                  T.ToTensor(),
                                  T.Normalize(mean, std)
                                ])

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

    assert dataset
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=int(workers))

    return dataloader, dataset