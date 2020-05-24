# Super-BPD for Fast Image Segmentation (CVPR 2020)

## Introduction

We propose direction-based super-BPD, an alternative to superpixel, for fast generic image segmentation, achieving state-of-the-art real-time result.

## Dataset
* Download the [BSDS500  & PascalContext Dataset](https://1drv.ms/u/s!AtAJxn0z15QehBRUdGCzWZq9AN59), and unzip it in the `Super-BPD/data` folder.

## Testing

* Download the pre-trained [PascalContext model](http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth) and put it in the `saved` folder.

* Test the model and results will be saved in the `test_pred_flux/PascalContext` folder.

## Training Examples
Train the PascalContext model.

```
python train.py 
```

