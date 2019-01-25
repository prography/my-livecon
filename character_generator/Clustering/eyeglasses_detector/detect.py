import torch
import torchvision.transforms as T

from eyeglasses_detector.model import GlassNet

from PIL import Image

# frame: 얼굴 이미지 전체 (Crop X)
def detect_eyeglasses(image_path):
    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define hyper-parameters
    image_size = 120
    weight_path = 'eyeglasses_detector/checkpoints/epoch10.pth'

    # load trained network
    net = GlassNet(image_size, 3, 16, 2)
    net.load_state_dict(torch.load(weight_path, map_location=lambda storage, loc: storage))
    net = net.to(device)

    # load image
    frame = Image.open(image_path)

    # define image transform
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.5,), std=(0.5,))
    ])

    # convert image to tensor
    image_tensor = transform(frame).unsqueeze(0)

    # forwarding
    with torch.no_grad():
        out = net(image_tensor)
        print(out)
        pred = torch.argmax(out, dim=1)

    return pred