import torch
from torch import nn
import torch.nn.functional as F

import models.resnet as models


class ResUnet(nn.Module):
    def __init__(self, layers=34, classes=2, BatchNorm=nn.BatchNorm2d, pretrained=True, cfg=None):
        super(ResUnet, self).__init__()
        assert classes > 1
        models.BatchNorm = BatchNorm
        if layers == 18:
            resnet = models.resnet18(pretrained=True, deep_base=False)
            block = models.BasicBlock
            layers = [2, 2, 2, 2]
            size_list = [64, 128, 256, 512]
        elif layers == 34:
            resnet = models.resnet34(pretrained=True, deep_base=False)
            block = models.BasicBlock
            layers = [3, 4, 6, 3]
            size_list = [64, 128, 256, 512]
        elif layers == 50:
            resnet = models.resnet50(pretrained=True, deep_base=False)
            block = models.Bottleneck
            layers = [3, 4, 6, 3]
            size_list = [256, 512, 1024, 2048]
        elif layers == 101:
            resnet = models.resnet101(pretrained=True, deep_base=False)
            block = models.Bottleneck
            layers = [3, 4, 23, 3]
            size_list = [256, 512, 1024, 2048]
        else:
            raise NotImplementedError
        
        block.expansion = 1 # added by xiang on Nov 28, 2023
        
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        # self.categories = cfg.categories
        # Decoder
        # self.up4 = nn.Sequential(nn.ConvTranspose2d(512,256,kernel_size=2,stride=2),BatchNorm(256),nn.ReLU())
        self.up4 = nn.Sequential(nn.Conv2d(size_list[-1], size_list[-2], kernel_size=3, stride=1, padding=1), BatchNorm(size_list[-2]), nn.ReLU())
        resnet.inplanes = size_list[-2] + size_list[-2]
        self.delayer4 = resnet._make_layer(block, size_list[-2], layers[-1])

        self.up3 = nn.Sequential(nn.Conv2d(size_list[-2], size_list[-3], kernel_size=3, stride=1, padding=1), BatchNorm(size_list[-3]), nn.ReLU())
        resnet.inplanes = size_list[-3] + size_list[-3]
        self.delayer3 = resnet._make_layer(block, size_list[-3], layers[-2])

        self.up2 = nn.Sequential(nn.Conv2d(size_list[-3], 96, kernel_size=3, stride=1, padding=1), BatchNorm(96), nn.ReLU())
        resnet.inplanes = 96 + size_list[-4]
        self.delayer2 = resnet._make_layer(block, 96, layers[-3])

        self.cls = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=3, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, cfg.classes, kernel_size=1)
        )
        
        self.img_size = (256,256)
        self.use_2d_classifier = True
        
        if self.use_2d_classifier:
            self.fc = nn.Sequential(
                nn.Linear(512 * self.img_size[0]//32 * self.img_size[1]//32, 2048), # (C* H/8 * W/8)
                nn.ReLU(),
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Linear(512, cfg.categories)
            )
        
        self.cls_mat = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=3, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, cfg.mat, kernel_size=1)
        )
            
        if self.training:
            self.aux = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                     BatchNorm(256), nn.ReLU(inplace=True),
                                     nn.Conv2d(256, classes, kernel_size=1))

    def forward(self, x):
        """
        x: BCHWV
        output: BCHWV
        """
        # 2D feature extract
        b,c,h,w,v = x.size()
        
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # BVCHW
        x = x.view(b*v,c,h,w)
        
        x1 = self.layer0(x)  # 1/4
        x2 = self.layer1(x1)  # 1/4
        x3 = self.layer2(x2)  # 1/8
        x4 = self.layer3(x3)  # 1/16
        x5 = self.layer4(x4)  # 1/32
        
        if self.use_2d_classifier:
            x6 = torch.max(x5.view(b, v, x5.shape[-3], x5.shape[-2], x5.shape[-1]), 1)[0].view(b,-1)
            categories_2d = F.softmax(self.fc(x6), dim=1)

        p4 = self.up4(F.interpolate(x5, x4.shape[-2:], mode='bilinear', align_corners=True))
        p4 = torch.cat([p4, x4], dim=1)
        p4 = self.delayer4(p4)
        p3 = self.up3(F.interpolate(p4, x3.shape[-2:], mode='bilinear', align_corners=True))
        p3 = torch.cat([p3, x3], dim=1)
        p3 = self.delayer3(p3)
        p2 = self.up2(F.interpolate(p3, x2.shape[-2:], mode='bilinear', align_corners=True))
        p2 = torch.cat([p2, x2], dim=1)
        p2 = self.delayer2(p2)
        
        x = self.cls(p2)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        x = x.view(b,v,-1,h,w).permute(0,2,3,4,1)
        
        res_mat = self.cls_mat(p2)
        res_mat = F.interpolate(res_mat, size=(h, w), mode='bilinear', align_corners=True)
        res_mat = res_mat.view(b,v,-1,h,w).permute(0,2,3,4,1)
        if self.use_2d_classifier:
            return x, res_mat, categories_2d
        else:
            return x, res_mat
        
