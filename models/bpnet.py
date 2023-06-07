#!/usr/bin/env python
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from models.unet_2d import ResUnet as model2D
from models.unet_3d import mink_unet as model3D
import MinkowskiEngine as ME
from models.bpm import Linking

import MinkowskiEngine as ME

import torch.nn.init as init
import pudb

def state_dict_remove_moudle(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:]  # remove 'module.' of dataparallel
        name = k.replace('module.', '')
        new_state_dict[name] = v
    return new_state_dict


def constructor3d(**kwargs):
    model = model3D(**kwargs)
    # model = model.cuda()
    return model


def constructor2d(**kwargs):
    model = model2D(**kwargs)
    # model = model.cuda()
    return model


class BPNet(nn.Module):

    def __init__(self, cfg=None):
        super(BPNet, self).__init__()
        self.viewNum = cfg.viewNum
        # 2D
        self.categories = cfg.categories
        net2d = constructor2d(layers=cfg.layers_2d, classes=cfg.classes, cfg=cfg)
        self.layer0_2d = net2d.layer0
        self.layer1_2d = net2d.layer1
        self.layer2_2d = net2d.layer2
        self.layer3_2d = net2d.layer3
        self.layer4_2d = net2d.layer4
        self.up4_2d = net2d.up4
        self.delayer4_2d = net2d.delayer4
        self.up3_2d = net2d.up3
        self.delayer3_2d = net2d.delayer3
        self.up2_2d = net2d.up2
        self.delayer2_2d = net2d.delayer2
        self.cls_2d = net2d.cls

        # 3D
        net3d = constructor3d(in_channels=3, out_channels=cfg.classes, D=3, arch=cfg.arch_3d)
        self.layer0_3d = nn.Sequential(net3d.conv0p1s1, net3d.bn0, net3d.relu)
        self.layer1_3d = nn.Sequential(net3d.conv1p1s2, net3d.bn1, net3d.relu, net3d.block1)
        self.layer2_3d = nn.Sequential(net3d.conv2p2s2, net3d.bn2, net3d.relu, net3d.block2)
        self.layer3_3d = nn.Sequential(net3d.conv3p4s2, net3d.bn3, net3d.relu, net3d.block3)
        self.layer4_3d = nn.Sequential(net3d.conv4p8s2, net3d.bn4, net3d.relu, net3d.block4)
        self.layer5_3d = nn.Sequential(net3d.convtr4p16s2, net3d.bntr4, net3d.relu)
        self.layer6_3d = nn.Sequential(net3d.block5, net3d.convtr5p8s2, net3d.bntr5, net3d.relu)
        self.layer7_3d = nn.Sequential(net3d.block6, net3d.convtr6p4s2, net3d.bntr6, net3d.relu)
        self.layer8_3d = nn.Sequential(net3d.block7, net3d.convtr7p2s2, net3d.bntr7, net3d.relu)
        self.layer9_3d = net3d.block8
        self.cls_3d = net3d.final
        
        ### some configuration
        self.img_size = (256,256)
        self.use_2d_classifier = cfg.use_2d_classifier
        self.use_3d_classifier = cfg.use_3d_classifier
        
        assert not (self.use_2d_classifier & self.use_3d_classifier) # in general, we only choose one shape classifier
        
        self.cls_mat = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, cfg.mat, kernel_size=1)
        )
        
        self.cls_3dmat = ME.MinkowskiConvolution(
            net3d.PLANES[7],
#             cfg.classes,
            cfg.mat,
            kernel_size=1,
            has_bias=True,
            dimension=3)
            
        if self.use_2d_classifier:
            self.pool2d = True
            if self.pool2d:
                in_channel = 512
            else: ## pool on local features, should not use
                in_channel = 512 * self.img_size[0]//32 * self.img_size[1]//32

            self.fc = nn.Sequential(
                nn.BatchNorm1d(in_channel),
                nn.Linear(in_channel, 512), # nn.Linear(, 2048), # (C* H/8 * W/8)
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(512),
                nn.Linear(512, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, self.categories)
            )
            
            self.fc.apply(self.init_weights)
            
        if self.use_3d_classifier:
            
            self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
            self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()

            self.fc3d = nn.Sequential(
                self.get_mlp_block(net3d.PLANES[3] * 2, 512),
                ME.MinkowskiDropout(),
                self.get_mlp_block(512, 512),
                ME.MinkowskiLinear(512, self.categories, bias=True),
            )

        # classification
        # Linker
        self.linker_p2 = Linking(96, net3d.PLANES[6], viewNum=self.viewNum)
        self.linker_p3 = Linking(128, net3d.PLANES[5], viewNum=self.viewNum)
        self.linker_p4 = Linking(256, net3d.PLANES[4], viewNum=self.viewNum)
        self.linker_p5 = Linking(512, net3d.PLANES[3], viewNum=self.viewNum)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

  
    def get_mlp_block(self, in_channel, out_channel):
        return nn.Sequential(
            ME.MinkowskiLinear(in_channel, out_channel, bias=False),
            ME.MinkowskiBatchNorm(out_channel),
            ME.MinkowskiReLU(), # MinkowskiLeakyReLU
        )

    def forward(self, sparse_3d, images, links):
        """
        images: BCHWV
        output: BCHWV
        """
        # 2D feature extract
        x_size = images.size()
        b, c, h, w, v = x_size[0], x_size[1], x_size[2], x_size[3], x_size[4]
        data_2d = images.permute(4,0, 1, 2, 3).contiguous()  # VBCHW
        
        data_2d = data_2d.view(x_size[0] * x_size[4], x_size[1], x_size[2], x_size[3])
        x = self.layer0_2d(data_2d)  # 1/4
        x2 = self.layer1_2d(x)  # 1/4
        x3 = self.layer2_2d(x2)  # 1/8
        x4 = self.layer3_2d(x3)  # 1/16
        x5 = self.layer4_2d(x4)  # 1/32, 
        
        if self.use_2d_classifier:
            # (VB, 512, 8, 8) - > (VB, 512) - >  (V, B, 512) -> (B, 512) -> (B C)
            if self.pool2d:
                x6 = torch.max(F.max_pool2d(x5, (h//32, w//32)).squeeze().view(v, b, -1), 0)[0].view(b,-1)
            else:
                x6 = torch.max(x5.view(v, b, x5.shape[-3], x5.shape[-2], x5.shape[-1]), 0)[0].view(b,-1)
            categories_2d = self.fc(x6)


        # 3D feature extract
        out_p1 = self.layer0_3d(sparse_3d)
        out_b1p2 = self.layer1_3d(out_p1)
        out_b2p4 = self.layer2_3d(out_b1p2)
        out_b3p8 = self.layer3_3d(out_b2p4)
        out_b4p16 = self.layer4_3d(out_b3p8)  # corresponding to FPN p5
        
        if self.use_3d_classifier:
            gb1 = self.global_max_pool(out_b4p16)
            gb2 = self.global_avg_pool(out_b4p16)
            categories_3d = self.fc3d(ME.cat(gb1, gb2)).F
#             print('out_b4p16', out_b4p16.shape, categories_3d.shape)
        
        # Linking @ p5
        B_V, C, H, W = x5.shape
        links_current_level = links.clone()
        links_current_level[:, 1:3, :] = ((H - 1.) / (h - 1.) * links_current_level[:, 1:3, :].float()).int()
        fused_3d_p5, fused_2d_p5 = self.linker_p5(x5, out_b4p16, links_current_level, init_3d_data=sparse_3d)

        p4 = self.up4_2d(F.interpolate(fused_2d_p5, x4.shape[-2:], mode='bilinear', align_corners=True))
        p4 = torch.cat([p4, x4], dim=1)
        p4 = self.delayer4_2d(p4)
        feat_3d = self.layer5_3d(fused_3d_p5)  # corresponding to FPN p4

        # Linking @ p4
        B_V, C, H, W = p4.shape
        links_current_level = links.clone()
        links_current_level[:, 1:3, :] = ((H - 1.) / (h - 1.) * links_current_level[:, 1:3, :].float()).int()
        fused_3d_p4, fused_2d_p4 = self.linker_p4(p4, feat_3d, links_current_level, init_3d_data=sparse_3d)

        p3 = self.up3_2d(F.interpolate(fused_2d_p4, x3.shape[-2:], mode='bilinear', align_corners=True))
        p3 = torch.cat([p3, x3], dim=1)
        p3 = self.delayer3_2d(p3)
        feat_3d = self.layer6_3d(ME.cat(fused_3d_p4, out_b3p8))  # corresponding to FPN p3

        # Linking @ p3
        B_V, C, H, W = p3.shape
        links_current_level = links.clone()
        links_current_level[:, 1:3, :] = ((H - 1.) / (h - 1.) * links_current_level[:, 1:3, :].float()).int()
        fused_3d_p3, fused_2d_p3 = self.linker_p3(p3, feat_3d, links_current_level, init_3d_data=sparse_3d)

        p2 = self.up2_2d(F.interpolate(fused_2d_p3, x2.shape[-2:], mode='bilinear', align_corners=True))
        p2 = torch.cat([p2, x2], dim=1)
        p2 = self.delayer2_2d(p2)
        feat_3d = self.layer7_3d(ME.cat(fused_3d_p3, out_b2p4))  # corresponding to FPN p2

        # Linking @ p2
        B_V, C, H, W = p2.shape
        links_current_level = links.clone()
        links_current_level[:, 1:3, :] = ((H - 1.) / (h - 1.) * links_current_level[:, 1:3, :].float()).int()
        fused_3d_p2, fused_2d_p2 = self.linker_p2(p2, feat_3d, links_current_level, init_3d_data=sparse_3d)

        feat_3d = self.layer8_3d(ME.cat(fused_3d_p2, out_b1p2))

        # Res
        res_2d = self.cls_2d(fused_2d_p2)
        res_2d = F.interpolate(res_2d, size=(h, w), mode='bilinear', align_corners=True)
        B_V, C, H, W = res_2d.shape
        res_2d = res_2d.view(v, b, C, H, W).permute(1,2,3,4,0) # BCHWV
    
        res_mat = self.cls_mat(fused_2d_p2)
        res_mat = F.interpolate(res_mat, size=(h, w), mode='bilinear', align_corners=True)
        V_B, C, H, W = res_mat.shape
        res_mat = res_mat.view(v, b, C, H, W).permute(1,2,3,4,0) # BCHWV
        
        res_3d_feat = self.layer9_3d(ME.cat(feat_3d, out_p1))
        res_3d = self.cls_3d(res_3d_feat)
        res_3dmat = self.cls_3dmat(res_3d_feat)
#         res_3dmat = self.cls_3dmat(res_3d)
        
        if self.use_2d_classifier:
            return res_3d.F, res_2d, res_mat, res_3dmat.F, categories_2d
        elif self.use_3d_classifier:
            return res_3d.F, res_2d, res_mat, res_3dmat.F, categories_3d
        else:
            res_3d.F, res_2d, res_mat, res_3dmat.F, None
        