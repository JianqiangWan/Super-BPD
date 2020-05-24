import torch
import torch.nn as nn
from model import VGG16
from vis_flux import vis_flux
from datasets import FluxSegmentationDataset
from torch.autograd import Variable
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader

DATASET = 'PascalContext'

def main():

    model = VGG16()

    model.load_state_dict(torch.load('saved/' + DATASET + '_400000.pth'))

    model.eval()
    model.cuda()
    
    dataloader = DataLoader(FluxSegmentationDataset(dataset=DATASET, mode='test'), batch_size=1, shuffle=False, num_workers=4)


    for i_iter, batch_data in enumerate(dataloader):

        Input_image, vis_image, gt_mask, gt_flux, weight_matrix, dataset_lendth, image_name = batch_data

        pred_flux = model(Input_image.cuda())

        vis_flux(vis_image, pred_flux, gt_flux, gt_mask, image_name, 'test_pred_flux/' + DATASET + '/')

        pred_flux = pred_flux.data.cpu().numpy()[0, ...]
        sio.savemat('test_pred_flux/' + DATASET + '/' + image_name[0] + '.mat', {'flux': pred_flux})


if __name__ == '__main__':
    main()





