import argparse
from misc import *
from myutils.metrics import *
import torchvision.utils as vutils
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=False, default='pix2pix',  help='')
    parser.add_argument('--dataroot', required=False, default='', help='path to trn dataset')
    parser.add_argument('--valDataroot', required=False, default='', help='path to val dataset')
    parser.add_argument('--modelPath', type=str, default='./models/', help='pretrained VGG16 path')
    parser.add_argument('--mode', type=str, default='B2A', help='B2A: facade, A2B: edges2shoes')
    parser.add_argument('--manualSeed', type=int, default=101, help='B2A: facade, A2B: edges2shoes')
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--valBatchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--originalSize', type=int, default=286, help='the height / width of the original input image')
    parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the cropped input image to network')
    parser.add_argument('--inputChannelSize', type=int, default=3, help='size of the input channels')
    parser.add_argument('--outputChannelSize', type=int, default=3, help='size of the output channels')
    parser.add_argument('--sizePatchGAN', type=int, default=62)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=1, help='number of epochs to train for')
    parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--annealStart', type=int, default=0, help='annealing learning rate start to')
    parser.add_argument('--annealEvery', type=int, default=400, help='epoch to reaching at learning rate of 0')
    parser.add_argument('--lambdaGAN', type=float, default=0.35, help='lambdaGAN')
    parser.add_argument('--lambdaIMG', type=float, default=1, help='lambdaIMG')
    parser.add_argument('--poolSize', type=int, default=50, help='Buffer size for storing previously generated samples from G')
    parser.add_argument('--wd', type=float, default=0.0000, help='weight decay in D')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--exp', default='sample', help='folder to output images and model checkpoints')
    parser.add_argument('--evalIter', type=int, default=10, help='interval for evauating(generating) images from valDataroot')
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    return parser.parse_args()

if __name__=='__main__':
    opt = get_args()
    opt.dataroot = 'facades/train512'
    # get dataloader
    dataloader = getLoader(opt, 
                           mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), 
                           split='train', shuffle=True)
    
    baseRoot = 'C:\\Users\\IIPL\\Desktop\\data\\NYU'
    for data in tqdm(dataloader):
        input, target, trans, ato, imgname = data
        # print(ato)
        
        vutils.save_image(input, f'{baseRoot}/hazy/{imgname[0]}.png', normalize=False, scale_each=False)
        vutils.save_image(ato, f'{baseRoot}/airlight_GT/{imgname[0]}.png', normalize=False, scale_each=False)