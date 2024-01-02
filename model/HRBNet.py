import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import time
import cv2

import model.resnet as models
import model.vgg as vgg_models
import model.hrnet as hrnet_models
import util.util as util
from matplotlib import pyplot as plt
import os



def Weighted_GAP(supp_feat, mask):
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    mask = F.interpolate(mask, size=(feat_h,feat_w), mode='bilinear', align_corners=True)
    supp_feat = supp_feat * mask
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat

class Pyramid_Joint_module(nn.Module):
    def __init__(self):
        super(Pyramid_Joint_module, self).__init__()
        self.layer1_0 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )
        self.layer1_1 = nn.Sequential(
            nn.Conv2d(96, 48, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )
        self.layer1_2 = nn.Sequential(
            nn.Conv2d(192,48, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )
        self.layer1_3 = nn.Sequential(
            nn.Conv2d(384, 48, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )
        self.layer2 = nn.Conv2d(192, 192, kernel_size=1, padding=0, bias=False)
        self.layer3 = nn.Conv2d(192, 192, kernel_size=1, padding=0, bias=False)
        # self.layer4 = nn.Conv2d(192, 192, kernel_size=1, padding=0, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, out, feature_size):
        f_0 = self.layer1_0(out[0])
        f_1 = self.layer1_1(out[1])
        f_2 = self.layer1_2(out[2])
        f_3 = self.layer1_3(out[3])
        f_1 = F.interpolate(f_1, size=feature_size, mode='bilinear', align_corners=True)
        f_2 = F.interpolate(f_2, size=feature_size, mode='bilinear', align_corners=True)
        f_3= F.interpolate(f_3, size=feature_size, mode='bilinear', align_corners=True)
        f = torch.cat([f_0,f_1,f_2,f_3],1)
        f_x = self.layer2(f)
        b,c,h,w = f_x.size()
        # print(f_x.size())
        avg_x = self.avg_pool(f_x)
        b,c,h_a,w_a = avg_x.size()
        avg_x = avg_x.view(b,c,h_a*w_a).permute(0, 2, 1)
        t_x = self.layer3(f).view(b,c,h*w)
        context = torch.matmul(avg_x,t_x)
        context = self.softmax(context)
        context = context.view(b,1,h,w)
        sp = self.sigmoid(context)
        # f = self.layer4(f)
        f = f*sp
        return f

class HRBNet(nn.Module):
    def __init__(self, layers=48, classes=2, zoom_factor=8, \
        criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, \
        pretrained=True, sync_bn=True, shot=1, ppm_scales=[119, 60, 30, 15,8], isvgg=False):
        super(HRBNet, self).__init__()
        assert layers in [18, 32, 48]
        print(ppm_scales)
        assert classes > 1
        from torch.nn import BatchNorm2d as BatchNorm        
        self.zoom_factor = zoom_factor
        self.criterion = criterion
        self.shot = shot
        self.ppm_scales = ppm_scales
        self.vgg = isvgg

        models.BatchNorm = BatchNorm
        print('INFO: using hrnet{}'.format(layers))
        reduce_dim = 192
        if layers == 18:
            hrnet = hrnet_models.hrnet18(pretrained=pretrained)
            fea_dimlsit = [18, 36, 72, 144]
            fea_dim = 18+36+72+144
        elif layers == 32:
            hrnet = hrnet_models.hrnet32(pretrained=pretrained)
            fea_dimlsit = [32, 64, 128, 256]
            fea_dim = 32+64+128+256
        else:
            hrnet = hrnet_models.hrnet48(pretrained=pretrained)
            fea_dimlsit = [48, 96, 192, 384]
            fea_dim = 48+96+192+384
        self.layer0 = nn.Sequential(hrnet.conv1, hrnet.bn1, hrnet.relu, hrnet.conv2, hrnet.bn2, hrnet.relu)
        self.stage1_cfg = hrnet.stage1_cfg
        self.layer1 = hrnet.layer1

        self.stage2_cfg = hrnet.stage2_cfg
        self.transition1 = hrnet.transition1
        self.layer2 = hrnet.stage2

        self.stage3_cfg = hrnet.stage3_cfg
        self.transition2 = hrnet.transition2
        self.layer3 = hrnet.stage3

        self.stage4_cfg = hrnet.stage4_cfg
        self.transition3 = hrnet.transition3
        self.layer4 = hrnet.stage4

        self.s_pyramid = Pyramid_Joint_module()
        self.q_pyramid = Pyramid_Joint_module()

        self.pyramid_bins = ppm_scales
        self.avgpool_list = []
        for bin in self.pyramid_bins:
            if bin > 1:
                self.avgpool_list.append(
                    nn.AdaptiveAvgPool2d(bin)
                )

        factor = 1
        mask_add_num = 1+1
        self.init_merge = []
        self.init_mergeco = []
        self.supp_init_merge = []
        self.supp_init_mergeco = []
        self.beta_conv = []
        self.inner_cls = []

        for bin in self.pyramid_bins:
            self.supp_init_merge.append(nn.Sequential(
                nn.Conv2d(reduce_dim * 2, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))
            self.supp_init_mergeco.append(nn.Sequential(
                nn.Conv2d(reduce_dim * 2, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(reduce_dim * 2, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))
            self.init_mergeco.append(nn.Sequential(
                nn.Conv2d(reduce_dim * 2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(reduce_dim, classes, kernel_size=1)
            ))
        self.init_merge = nn.ModuleList(self.init_merge)
        self.init_mergeco = nn.ModuleList(self.init_mergeco)
        self.supp_init_merge = nn.ModuleList(self.supp_init_merge)
        self.supp_init_mergeco = nn.ModuleList(self.supp_init_mergeco)
        self.beta_conv = nn.ModuleList(self.beta_conv)
        self.inner_cls = nn.ModuleList(self.inner_cls)

        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim * len(self.pyramid_bins), reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # self.GAP = nn.AdaptiveAvgPool2d(1)

        self.alpha_conv = []
        for idx in range(len(self.pyramid_bins) - 1):
            self.alpha_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim*2, reduce_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU()
            ))
        self.alpha_conv = nn.ModuleList(self.alpha_conv)
     

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )

        self.q_q = nn.Conv2d(reduce_dim, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.q_v = nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, stride=1, padding=0, bias=False)

        self.s_q = nn.Conv2d(reduce_dim, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.s_v = nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, stride=1, padding=0, bias=False)

        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, s_x=torch.FloatTensor(1,1,3,473,473).cuda(), s_y=torch.FloatTensor(1,1,473,473).cuda(), y=None):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        #   Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_1 = self.layer1(query_feat_0)#256,119

            x_list = []
            for i in range(self.stage2_cfg['NUM_BRANCHES']):
                if self.transition1[i] is not None:
                    x_list.append(self.transition1[i](query_feat_1))
                else:
                    x_list.append(query_feat_1)
            query_feat_2 = self.layer2(x_list)#48,119;96,60

            x_list = []
            for i in range(self.stage3_cfg['NUM_BRANCHES']):
                if self.transition2[i] is not None:
                    if i < self.stage2_cfg['NUM_BRANCHES']:
                        x_list.append(self.transition2[i](query_feat_2[i]))
                    else:
                        x_list.append(self.transition2[i](query_feat_2[-1]))
                else:
                    x_list.append(query_feat_2[i])
            query_feat_3 = self.layer3(x_list)#48,119;96,60;192,30

            x_list = []
            for i in range(self.stage4_cfg['NUM_BRANCHES']):
                if self.transition3[i] is not None:
                    if i < self.stage3_cfg['NUM_BRANCHES']:
                        x_list.append(self.transition3[i](query_feat_3[i]))
                    else:
                        x_list.append(self.transition3[i](query_feat_3[-1]))
                else:
                    x_list.append(query_feat_3[i])

            query_feat_4 = self.layer4(x_list)#48,119;96,60;192,30;384,15


            # ======fix
            x0_h, x0_w = query_feat_4[1].size(2), query_feat_4[1].size(3)
            x1 = F.interpolate(query_feat_4[0], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
            x2 = F.interpolate(query_feat_4[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
            x3 = F.interpolate(query_feat_4[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
            query_feat_f60 = torch.cat([query_feat_4[1], x1, x2, x3], 1)
            # ======high
            x0_h, x0_w = query_feat_4[0].size(2), query_feat_4[0].size(3)
            x1 = F.interpolate(query_feat_4[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
            x2 = F.interpolate(query_feat_4[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
            x3 = F.interpolate(query_feat_4[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
            query_feat_f = torch.cat([query_feat_4[0], x1, x2, x3], 1)
            query_feat = self.q_pyramid(query_feat_4,[x0_h,x0_w])

            query_feat_x = self.q_v(query_feat)
            batch, channel, height, width = query_feat_x.size()
            query_feat_x = query_feat_x.view(batch, channel, height * width)  # N,C,HW
            context_mask = self.q_q(query_feat)
            # print(context_mask.size())
            context_mask = context_mask.view(batch, 1, height * width)  # N,1,HW
            context_mask = self.softmax(context_mask)
            context = torch.matmul(query_feat_x, context_mask.transpose(1, 2))
            context = context.unsqueeze(-1)
            q_ch_att = self.sigmoid(context)

        #   Support Feature     
        supp_feat_list = []
        supp_featgap_list = []
        final_supp_list = []

        final_supp_list60 = []
        mask_list = []
        ch_att_list = []
        for i in range(self.shot):
            mask = (s_y[:,i,:,:] == 1).float().unsqueeze(1)
            mask_list.append(mask)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:,i,:,:,:])
                supp_feat_1 = self.layer1(supp_feat_0)

                x_list = []
                for i in range(self.stage2_cfg['NUM_BRANCHES']):
                    if self.transition1[i] is not None:
                        x_list.append(self.transition1[i](supp_feat_1))
                    else:
                        x_list.append(supp_feat_1)
                supp_feat_2 = self.layer2(x_list)

                x_list = []
                for i in range(self.stage3_cfg['NUM_BRANCHES']):
                    if self.transition2[i] is not None:
                        if i < self.stage2_cfg['NUM_BRANCHES']:
                            x_list.append(self.transition2[i](supp_feat_2[i]))
                        else:
                            x_list.append(self.transition2[i](supp_feat_2[-1]))
                    else:
                        x_list.append(supp_feat_2[i])
                supp_feat_3 = self.layer3(x_list)

                x_list = []
                for i in range(self.stage4_cfg['NUM_BRANCHES']):
                    if self.transition3[i] is not None:
                        if i < self.stage3_cfg['NUM_BRANCHES']:
                            x_list.append(self.transition3[i](supp_feat_3[i]))
                        else:
                            x_list.append(self.transition3[i](supp_feat_3[-1]))
                    else:
                        x_list.append(supp_feat_3[i])
                supp_feat_4 = self.layer4(x_list)

                # ======fix
                x0_h, x0_w = supp_feat_4[1].size(2), supp_feat_4[1].size(3)
                x1 = F.interpolate(supp_feat_4[0], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
                x2 = F.interpolate(supp_feat_4[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
                x3 = F.interpolate(supp_feat_4[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
                supp_feat_f60 = torch.cat([supp_feat_4[1], x1, x2, x3], 1)
                final_supp_list60.append(supp_feat_f60)  # 60
                #========high
                x0_h, x0_w = supp_feat_4[0].size(2), supp_feat_4[0].size(3)
                x1 = F.interpolate(supp_feat_4[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
                x2 = F.interpolate(supp_feat_4[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
                x3 = F.interpolate(supp_feat_4[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
                supp_feat_f = torch.cat([supp_feat_4[0], x1, x2, x3], 1)
                final_supp_list.append(supp_feat_f)
                supp_feat = self.s_pyramid(supp_feat_4, [x0_h, x0_w])
                supp_feat_list.append(supp_feat)

                supp_feat_x = self.s_v(supp_feat)
                supp_feat_x = supp_feat_x.view(batch, channel, height * width)  # N,C,HW
                scontext_mask = self.s_q(supp_feat)
                scontext_mask = scontext_mask.view(batch, 1, height * width)  # N,1,HW
                scontext_mask = self.softmax(scontext_mask)
                scontext = torch.matmul(supp_feat_x, scontext_mask.transpose(1, 2))
                scontext = scontext.unsqueeze(-1)
                s_ch_att = self.sigmoid(scontext)
                ch_att = s_ch_att * q_ch_att
                ch_att_list.append(ch_att)
                supp_feat_gap = Weighted_GAP(supp_feat, mask)
                supp_featgap_list.append(supp_feat_gap)

        cosine_eps = 1e-7
        corr_query_mask_list = []
        corr_query_mask = util.compute_prior_mask(final_supp_list60, mask_list, corr_query_mask_list,
                                                    query_feat_f60, cosine_eps)
        corr_query_mask_listbdc = []
        corr_query_maskbdc = util.compute_prior_mask_bdc(final_supp_list60, mask_list, corr_query_mask_listbdc,
                                                           query_feat_f60, cosine_eps)

        if self.shot > 1:

            supp_feat_gap = supp_featgap_list[0]
            ch_att = ch_att_list[0]
            for i in range(1, len(supp_feat_list)):
                supp_feat_gap += supp_featgap_list[i]
                ch_att += ch_att_list[i]
            supp_feat_gap /= len(supp_feat_list)
            ch_att /= len(ch_att_list)

        supp_feat_co = supp_feat * ch_att
        query_feat_co = query_feat * ch_att
        if self.training:
            supp_out_list = []
            supp_pyramid_feat_list = []

            for idx, tmp_bin in enumerate(self.pyramid_bins):
                if tmp_bin <= 1.0:
                    bin = int(supp_feat.shape[2] * tmp_bin)
                    supp_feat_map_bin = nn.AdaptiveAvgPool2d(bin)(supp_feat)
                    supp_featco_bin = nn.AdaptiveAvgPool2d(bin)(supp_feat_co)
                else:
                    bin = tmp_bin
                    supp_feat_map_bin = self.avgpool_list[idx](supp_feat)
                    supp_featco_bin = self.avgpool_list[idx](supp_feat_co)

                supp_featgap_bin = supp_feat_gap.expand(-1, -1, bin, bin)

                merge_supp_feat_bin = torch.cat([supp_feat_map_bin,supp_featco_bin], 1)
                merge_supp_feat_bin = self.supp_init_merge[idx](merge_supp_feat_bin)
                merge_supp_feat_bin = torch.cat([merge_supp_feat_bin, supp_featgap_bin], 1)
                merge_supp_feat_bin = self.supp_init_mergeco[idx](merge_supp_feat_bin)

                if idx >= 1:
                    pre_supp_feat_bin = supp_pyramid_feat_list[idx - 1].clone()
                    pre_supp_feat_bin = F.interpolate(pre_supp_feat_bin, size=(bin, bin), mode='bilinear',
                                                      align_corners=True)
                    rec_supp_feat_bin = torch.cat([merge_supp_feat_bin, pre_supp_feat_bin], 1)
                    merge_supp_feat_bin = self.alpha_conv[idx - 1](rec_supp_feat_bin) + merge_supp_feat_bin

                merge_supp_feat_bin = self.beta_conv[idx](merge_supp_feat_bin) + merge_supp_feat_bin
                inner_supp_out_bin = self.inner_cls[idx](merge_supp_feat_bin)
                merge_supp_feat_bin = F.interpolate(merge_supp_feat_bin,
                                                    size=(supp_feat.size(2), supp_feat.size(3)), mode='bilinear',
                                                    align_corners=True)
                supp_pyramid_feat_list.append(merge_supp_feat_bin)
                supp_out_list.append(inner_supp_out_bin)

            supp_feat_map_init = torch.cat(supp_pyramid_feat_list, 1)
            supp_feat_map_init = self.res1(supp_feat_map_init)
            supp_feat_map_init = self.res2(supp_feat_map_init) + supp_feat_map_init
            supp_out = self.cls(supp_feat_map_init)

        out_list = []
        pyramid_feat_list = []
        for idx, tmp_bin in enumerate(self.pyramid_bins):
            if tmp_bin <= 1.0:
                bin = int(query_feat.shape[2] * tmp_bin)
                query_feat_bin = nn.AdaptiveAvgPool2d(bin)(query_feat)
                query_feat_co_bin = nn.AdaptiveAvgPool2d(bin)(query_feat_co)
            else:
                bin = tmp_bin
                query_feat_bin = self.avgpool_list[idx](query_feat)
                query_feat_co_bin = self.avgpool_list[idx](query_feat_co)

            supp_featgap_bin = supp_feat_gap.expand(-1, -1, bin, bin)
            corr_mask_binbdc = F.interpolate(corr_query_maskbdc, size=(bin, bin), mode='bilinear', align_corners=True)
            corr_mask_bin = F.interpolate(corr_query_mask, size=(bin, bin), mode='bilinear', align_corners=True)
            corr_mask_bin_sum = torch.cat([corr_mask_binbdc,corr_mask_bin],1)

            merge_feat_bin = torch.cat([query_feat_bin, query_feat_co_bin], 1)
            merge_feat_bin = self.init_merge[idx](merge_feat_bin)

            merge_feat_bin = torch.cat([merge_feat_bin,supp_featgap_bin,corr_mask_bin_sum], 1)
            merge_feat_bin = self.init_mergeco[idx](merge_feat_bin)

            if idx >= 1:
                pre_feat_bin = pyramid_feat_list[idx - 1].clone()
                pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True)
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                merge_feat_bin = self.alpha_conv[idx - 1](rec_feat_bin) + merge_feat_bin

            merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin
            inner_out_bin = self.inner_cls[idx](merge_feat_bin)
            merge_feat_bin = F.interpolate(merge_feat_bin, size=(query_feat.size(2), query_feat.size(3)),
                                           mode='bilinear', align_corners=True)
            pyramid_feat_list.append(merge_feat_bin)
            out_list.append(inner_out_bin)

        query_feat = torch.cat(pyramid_feat_list, 1)
        query_feat = self.res1(query_feat)
        query_feat = self.res2(query_feat) + query_feat
        out = self.cls(query_feat)
        

        #   Output Part
        if self.zoom_factor != 1:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            supp_out = F.interpolate(supp_out, size=(h, w), mode='bilinear', align_corners=True)
            main_loss = self.criterion(out, y.long())+ self.criterion(supp_out, s_y.squeeze(dim=1).long())
            aux_loss = torch.zeros_like(main_loss).cuda()
            supp_aux_loss = torch.zeros_like(main_loss).cuda()

            for idx_k in range(len(out_list)):    
                inner_out = out_list[idx_k]
                inner_out = F.interpolate(inner_out, size=(h, w), mode='bilinear', align_corners=True)
                aux_loss = aux_loss + self.criterion(inner_out, y.long())
                supp_inner_out = supp_out_list[idx_k]
                supp_inner_out = F.interpolate(supp_inner_out, size=(h, w), mode='bilinear', align_corners=True)
                supp_aux_loss = supp_aux_loss + self.criterion(supp_inner_out, s_y.squeeze(dim=1).long())
            aux_loss = aux_loss / len(out_list)+ supp_aux_loss / len(supp_out_list)
            return out.max(1)[1], main_loss, aux_loss
        else:
            return out





