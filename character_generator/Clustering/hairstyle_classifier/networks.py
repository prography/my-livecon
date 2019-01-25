from hairstyle_classifier.pspnet import PSPNet

def get_network(name):
    name = name.lower()
    if name == 'pspnet_squeezenet':
        return PSPNet(num_class=1, base_network='squeezenet')
    elif name == 'pspnet_resnet101':
        return PSPNet(num_class=1, base_network='resnet101')

    raise ValueError
