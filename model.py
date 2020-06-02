import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.backbone_layer1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True))

        self.backbone_layer2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True))

        self.backbone_layer3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True))

        self.backbone_layer4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True))

        self.backbone_layer5 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True))
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        
        self.d2conv_ReLU = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True))
        self.d4conv_ReLU = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=4, dilation=4), nn.ReLU(inplace=True))
        self.d8conv_ReLU = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=8, dilation=8), nn.ReLU(inplace=True))
        self.d16conv_ReLU = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=16, dilation=16), nn.ReLU(inplace=True))
        
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1), nn.ReLU(inplace=True))

        self.predict_layer = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(512, 512, kernel_size=1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(512, 2, kernel_size=1))
        
    def forward(self, x):

        input_size = x.size()[2:]

        stage1 = self.backbone_layer1(x)
        stage1_maxpool = self.maxpool(stage1)

        stage2 = self.backbone_layer2(stage1_maxpool)
        stage2_maxpool = self.maxpool(stage2)

        stage3 = self.backbone_layer3(stage2_maxpool)
        stage3_maxpool = self.maxpool(stage3)
        tmp_size = stage3.size()[2:]

        stage4 = self.backbone_layer4(stage3_maxpool)
        stage4_maxpool = self.maxpool(stage4)

        stage5 = self.backbone_layer5(stage4_maxpool)

        d2conv_ReLU = self.d2conv_ReLU(stage5)
        d4conv_ReLU = self.d4conv_ReLU(stage5)
        d8conv_ReLU = self.d8conv_ReLU(stage5)
        d16conv_ReLU = self.d16conv_ReLU(stage5)

        dilated_conv_concat = torch.cat((d2conv_ReLU, d4conv_ReLU, d8conv_ReLU, d16conv_ReLU), 1)

        sconv1 = self.conv1(dilated_conv_concat)
        sconv1 = F.interpolate(sconv1, size=tmp_size, mode='bilinear', align_corners=True)

        sconv2 = self.conv2(stage5)
        sconv2 = F.interpolate(sconv2, size=tmp_size, mode='bilinear', align_corners=True)

        sconv3 = self.conv3(stage4)
        sconv3 = F.interpolate(sconv3, size=tmp_size, mode='bilinear', align_corners=True)

        sconv4 = self.conv4(stage3)
        sconv4 = F.interpolate(sconv4, size=tmp_size, mode='bilinear', align_corners=True)

        sconcat = torch.cat((sconv1, sconv2, sconv3, sconv4), 1)

        pred_flux = self.predict_layer(sconcat)
        pred_flux = F.interpolate(pred_flux, size=input_size, mode='bilinear', align_corners=True)

        return pred_flux






        