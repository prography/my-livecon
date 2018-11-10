from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as T

import cv2

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def main():
    transform = T.Compose([
        T.Resize(330),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset = ImageFolder(r"D:\Deep_learning\Data\멘토_LiveCon\dataset_google_cartoon",
                          transform=transform)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    for img, cls in dataloader:
        print("Image shape:", img.shape)
        print("Class:", cls)

        imgNumpy = denorm(img).detach().numpy()

        sampleImg = imgNumpy[0].transpose(1, 2, 0)
        r, g, b = cv2.split(sampleImg)
        sampleImg = cv2.merge([b, g, r])

        cv2.imshow("google cartoon image", sampleImg)
        cv2.waitKey(0)

if __name__ == "__main__":
    main()