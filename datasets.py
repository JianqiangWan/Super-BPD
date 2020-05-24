import cv2
import numpy as np
import os.path as osp
import scipy.io as sio
from torch.utils.data import Dataset

IMAGE_MEAN = np.array([103.939, 116.779, 123.675], dtype=np.float32)

class FluxSegmentationDataset(Dataset):

    def __init__(self, dataset='PascalContext', mode='train'):
        
        self.dataset = dataset
        self.mode = mode

        file_dir = 'datasets/' + self.dataset + '/' + self.mode + '.txt'

        self.random_flip = False
        
        if self.dataset == 'PascalContext' and mode == 'train':
            self.random_flip = True

        with open(file_dir, 'r') as f:
            self.image_names = f.read().splitlines()

        self.dataset_length = len(self.image_names)
    
    def __len__(self):

        return self.dataset_length

    def __getitem__(self, index):

        random_int = np.random.randint(0,2)

        image_name = self.image_names[index]

        image_path = osp.join('datasets', self.dataset, 'images', image_name + '.jpg')
        
        image = cv2.imread(image_path, 1)
        
        if self.random_flip:
            if random_int:
                image = cv2.flip(image, 1)
        
        vis_image = image.copy()

        height, width = image.shape[:2]
        image = image.astype(np.float32)
        image -= IMAGE_MEAN
        image = image.transpose(2, 0, 1)

        if self.dataset == 'PascalContext':
            label_path = osp.join('datasets', self.dataset, 'labels', image_name + '.mat')
            label = sio.loadmat(label_path)['LabelMap']
        
        elif self.dataset == 'BSDS500':
            label_path = osp.join('datasets', self.dataset, 'labels', image_name + '.png')
            label = cv2.imread(label_path, 0)

        if self.random_flip:
            if random_int:
                label = cv2.flip(label, 1)

        label += 1

        gt_mask = label.astype(np.float32)

        categories = np.unique(label)

        if 0 in categories:
            raise RuntimeError('invalid category')

        label = cv2.copyMakeBorder(label, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

        weight_matrix = np.zeros((height+2, width+2), dtype=np.float32)
        direction_field = np.zeros((2, height+2, width+2), dtype=np.float32)

        for category in categories:
            img = (label == category).astype(np.uint8)
            weight_matrix[img > 0] = 1. / np.sqrt(img.sum())

            _, labels = cv2.distanceTransformWithLabels(img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)

            index = np.copy(labels)
            index[img > 0] = 0
            place =  np.argwhere(index > 0)

            nearCord = place[labels-1,:]
            x = nearCord[:, :, 0]
            y = nearCord[:, :, 1]
            nearPixel = np.zeros((2, height+2, width+2))
            nearPixel[0,:,:] = x
            nearPixel[1,:,:] = y
            grid = np.indices(img.shape)
            grid = grid.astype(float)
            diff = grid - nearPixel    

            direction_field[:, img > 0] = diff[:, img > 0]     

        weight_matrix = weight_matrix[1:-1, 1:-1]
        direction_field = direction_field[:, 1:-1, 1:-1]
        
        if self.dataset == 'BSDS500':
            image_name = image_name.split('/')[-1]

        return image, vis_image, gt_mask, direction_field, weight_matrix, self.dataset_length, image_name




