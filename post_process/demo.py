import torch
import bpd_cuda
import math
import scipy.io as sio
import cv2
import numpy as np

def label2color(label):

    label = label.astype(np.uint16)
    
    height, width = label.shape
    color3u = np.zeros((height, width, 3), dtype=np.uint8)
    unique_labels = np.unique(label)

    if unique_labels[-1] >= 2**24:       
        raise RuntimeError('Error: label overflow!')

    for i in range(len(unique_labels)):
    
        binary = '{:024b}'.format(unique_labels[i])
        # r g b 3*8 24
        r = int(binary[::3][::-1], 2)
        g = int(binary[1::3][::-1], 2)
        b = int(binary[2::3][::-1], 2)

        color3u[label == unique_labels[i]] = np.array([r, g, b])

    return color3u

flux = sio.loadmat('./2009_004607.mat')['flux']
flux = torch.from_numpy(flux).cuda()

angles = torch.atan2(flux[1,...], flux[0,...])
angles[angles < 0] += 2*math.pi

height, width = angles.shape

# unit: degree
# theta_a, theta_l, theta_s, S_o, 45, 116, 68, 5
results = bpd_cuda.forward(angles, height, width, 45, 116, 68, 5)
root_points, super_BPDs_before_dilation, super_BPDs_after_dilation, super_BPDs = results

root_points = root_points.cpu().numpy()
super_BPDs_before_dilation = super_BPDs_before_dilation.cpu().numpy()
super_BPDs_after_dilation = super_BPDs_after_dilation.cpu().numpy()
super_BPDs = super_BPDs.cpu().numpy()

cv2.imwrite('root.png', 255*(root_points > 0))
cv2.imwrite('super_BPDs.png', label2color(super_BPDs))
cv2.imwrite('super_BPDs_before_dilation.png', label2color(super_BPDs_before_dilation))
cv2.imwrite('super_BPDs_after_dilation.png', label2color(super_BPDs_after_dilation))

