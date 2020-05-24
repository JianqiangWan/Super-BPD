import argparse
import os
import torch
import torch.nn as nn
from model import VGG16
from vis_flux import vis_flux
from datasets import FluxSegmentationDataset
from torch.autograd import Variable
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader

DATASET = 'PascalContext'
TEST_VIS_DIR = './test_pred_flux/'
SNAPSHOT_DIR = './snapshots/'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Super-BPD Network")
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help="Dataset for training.")
    parser.add_argument("--test-vis-dir", type=str, default=TEST_VIS_DIR,
                        help="Directory for saving vis results during testing.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    return parser.parse_args()

args = get_arguments()

def main():

    if not os.path.exists(args.test_vis_dir + args.dataset):
        os.makedirs(args.test_vis_dir + args.dataset)

    model = VGG16()

    model.load_state_dict(torch.load(args.snapshot_dir + args.dataset + '_400000.pth'))

    model.eval()
    model.cuda()
    
    dataloader = DataLoader(FluxSegmentationDataset(dataset=args.dataset, mode='test'), batch_size=1, shuffle=False, num_workers=4)

    for i_iter, batch_data in enumerate(dataloader):

        Input_image, vis_image, gt_mask, gt_flux, weight_matrix, dataset_lendth, image_name = batch_data

        print(i_iter, dataset_lendth)

        pred_flux = model(Input_image.cuda())

        vis_flux(vis_image, pred_flux, gt_flux, gt_mask, image_name, args.test_vis_dir + args.dataset + '/')

        pred_flux = pred_flux.data.cpu().numpy()[0, ...]
        sio.savemat(args.test_vis_dir + args.dataset + '/' + image_name[0] + '.mat', {'flux': pred_flux})


if __name__ == '__main__':
    main()





