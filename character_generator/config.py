import argparse

parser = argparse.ArgumentParser()

# dataset configuration
parser.add_argument('--dataset', default='face2ghibli', type=str, help='name of your dataset')
parser.add_argument('--dataroot', default='dataset/dataset', help='path to dataset.')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--image_size', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--split_ratio', type=float, default=0.1, help='ratio of test dataset over total dataset')

# model configuration
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--nb', type=int, default=9, help='number of resnet blocks')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)

# training configuration
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--cycle_lambda', type=float, default=10.0, help='cycle loss lambda')

parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.5')
parser.add_argument('--decay_epoch', type=int, default=100, help='learning rate decay start epoch num')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')

# interval configuration
parser.add_argument('--log_step', type=int, default=100, help='step of printing log message')
parser.add_argument('--sample_step', type=int, default=500, help='step of saving sample images')
parser.add_argument('--checkpoint_step', type=int, default=1800, help='step of saving checkpoints')

# directory configuration
parser.add_argument('--model_path', default='', help="path to saved models (to continue training)")
parser.add_argument('--sample_folder', default=None, help='folder to training sample images')
parser.add_argument('--ckpt_folder', default=None, help='folder to model checkpoints')
parser.add_argument('--test_result_folder', default='test', help='folder to test output images')

def get_config():
    return parser.parse_args()