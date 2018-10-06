import argparse

parser = argparse.ArgumentParser()

# Dataset configurations
parser.add_argument('--rootpath', type=str, default='dataset',  help='root directory for dataset')

parser.add_argument('--datasetA', default='celebA',  help='dataset name for A domain')
parser.add_argument('--datarootA', default='dataset/dataset_cropped/train/A/origin', help='root-path for datasetA')
parser.add_argument('--valDatarootA', default='dataset/dataset_cropped/train/A/origin', help='root-path for val. datasetA')

parser.add_argument('--datasetB', default='ghibli',  help='dataset name for B domain')
parser.add_argument('--datarootB', default='dataset/dataset_cropped/train/D', help='root-path for datasetB')
parser.add_argument('--valDatarootB', default='dataset/dataset_cropped/train/D', help='root-path for val. datasetB')

# Dataloader configurations
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--val_batch_size', type=int, default=36, help='validation batch size')
parser.add_argument('--original_image_size', type=int, default=36, help='the height / width of the original input image')
parser.add_argument('--image_size', type=int, default=32, help='the height / width of the cropped input image to network')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)

# Model configurations
parser.add_argument('--num_classes', type=int, default=1, help='# of classes in source domain')
parser.add_argument('--ngf', type=int, default=256)
parser.add_argument('--ndf', type=int, default=256)
parser.add_argument('--in_ch', type=int, default=3)
parser.add_argument('--out_ch', type=int, default=3)

parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netE_pretrained1', default="encoder_weights/vggface.h5", help="path to netE trained on source domain")
parser.add_argument('--netE_pretrained2', default="encoder_weights/VGG_FACE.caffemodel.pth", help="path to netE trained on source domain")

# Training configurations
parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--betaTID', type=float, default=15, help='beta weight')
parser.add_argument('--alphaCONST', type=float, default=15, help='alpha weight')
parser.add_argument('--crossentropy', action='store_true', help='Whether to use crossentropy loss in computing L_CONST(default: L1Loss')
parser.add_argument('--wd', type=float, default=0.0000, help='weight decay for Descriminator, default=0.0')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--rmsprop', action='store_true', help='Whether to use rmsprop (default is adam)')

# Interval configurations
parser.add_argument('--log_interval', type=int, default=500, help='interval for console display')
parser.add_argument('--sample_interval', type=int, default=500, help='interval for evauating(generating) val. images')
parser.add_argument('--ckpt_interval', type=int, default=500, help='interval for checkpointing')

# Directory configurations
parser.add_argument('--sample_folder', default=None, help='folder to output images')
parser.add_argument('--ckpt_folder', default=None, help='folder to output checkpoints')

def get_config():
    config = parser.parse_args()
    return config