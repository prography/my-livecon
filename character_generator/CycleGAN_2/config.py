import argparse

parser = argparse.ArgumentParser()

# data processing hyper-parameters
parser.add_argument('--datarootA', type=str, default="dataset_cropped", help='root directory of the dataset A(source)')
parser.add_argument('--datarootB', type=str, default="dataset_cropped", help='root directory of the dataset B(target)')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--num_workers', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--unaligned', type=bool, default=True, help='whether dataset A-B matches')
parser.add_argument('--image_size', type=int, default=256, help='size of the data crop (squared assumed)')

# training hyper-parameters
parser.add_argument('--starting_epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs of training. default=50')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate. default=0.00002')
parser.add_argument('--decay_epoch', type=int, default=40, help='epoch to start linearly decaying the learning rate to 0. default=40')
parser.add_argument('--n_in', type=int, default=3, help='number of channels of input data')
parser.add_argument('--n_out', type=int, default=3, help='number of channels of output data')
parser.add_argument('--netG_A2B', type=str, default="", help="to continue training")
parser.add_argument('--netG_B2A', type=str, default="", help="to continue training")
parser.add_argument('--netD_A', type=str, default="", help="to continue training")
parser.add_argument('--netD_B', type=str, default="", help="to continue training")

# step-size configurations
parser.add_argument('--log_interval', type=int, default=20, help="step interval to print log message")
parser.add_argument('--sample_interval', type=int, default=100, help="step interval to save sample images. default=100")
parser.add_argument('--ckpt_interval', type=int, default=500, help="step interval to save checkpoints. default=500")

# directory configurations
parser.add_argument('--sample_folder', type=str, default=None)
parser.add_argument('--ckpt_folder', type=str, default=None)

# test configuration
parser.add_argument('--model_path', type=str, default="checkpoints/GoogleCartoon/netG_A2B_epoch16.pth")
parser.add_argument('--test_result_folder', type=str, default="test/cropped_ver2_10")


config = parser.parse_args()

def get_config():
    return config