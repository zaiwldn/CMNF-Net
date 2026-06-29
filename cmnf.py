import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from transformers import SegformerModel
import math

class GeoDM(nn.Module):
    def __init__(self, in_features, out_features, num_Hs=2, q_dim=16, norm_p=2, SCALE_FACTOR_fc=0.1, enable_bias=False):
        super(GeoDM, self).__init__()
        self.norm_p = norm_p
        self.num_Hs = num_Hs
        self.SCALE_FACTOR_fc = SCALE_FACTOR_fc
        self.In_Qs_1 = torch.nn.Parameter(torch.rand(num_Hs, in_features, q_dim//2))
        self.Out_Qs_1 = torch.nn.Parameter(torch.rand(num_Hs, out_features, q_dim//2))
        self.shared_coeff_fc = torch.nn.Parameter(
            SCALE_FACTOR_fc * torch.tensor([(-1)**h_id for h_id in range(self.num_Hs)]).unsqueeze(1).unsqueeze(2),
            requires_grad=False
        )
        self.bias = nn.Parameter(torch.zeros(out_features)) if enable_bias else None
        nn.init.kaiming_uniform_(self.In_Qs_1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Out_Qs_1, a=math.sqrt(5))

    def pathIntegrals(self):
        dist_io = torch.cdist(self.In_Qs_1, self.Out_Qs_1, p=self.norm_p)
        return torch.sum(self.shared_coeff_fc * dist_io, dim=0)

    def forward(self, x):
        weight = self.pathIntegrals()
        return F.linear(x, weight.T, self.bias)


class DKSM(nn.Module):
    def __init__(self, in_channels, out_channels, q_dim=32, num_Hs=4):
        super().__init__()
        self.conv_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.GeoDM = GeoDM(
            in_features=out_channels,
            out_features=out_channels,
            q_dim=q_dim,
            num_Hs=num_Hs,
            norm_p=2
        )
        self.gate = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x_proj = self.conv_proj(x)
        b, c, h, w = x_proj.shape
        x_flat = x_proj.flatten(2).transpose(1, 2)  # [b, h*w, c]
        x_kernel = self.GeoDM(x_flat).transpose(1, 2).view(b, c, h, w)  # [b, c, h, w]
        x_fused = torch.cat([x_proj, x_kernel], dim=1)
        x_fused = self.gate(x_fused)
        return self.norm(self.act(x_fused))


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.add_module('conv', nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        ))
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class ConvRelu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.add_module('conv', nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        ))
        self.add_module('relu', nn.ReLU())
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class CBAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = Conv(c, c, 3, 1, 1)
        self.conv2 = nn.Sequential(ConvRelu(c, c, 1, 1, 0), Conv(c, c, 1, 1, 0))
        self.conv3 = nn.Sequential(ConvRelu(2, 16, 3, 1, 1), Conv(16, 1, 3, 1, 1))

    def forward(self, x):
        x = self.conv1(x)
        c = torch.sigmoid(self.conv2(F.adaptive_avg_pool2d(x, output_size=(1, 1))) + self.conv2(F.adaptive_max_pool2d(x, output_size=(1, 1))))
        x = x * c
        s = torch.sigmoid(self.conv3(torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1)))
        x = x * s
        return x

class Encoder(nn.Module):
    def __init__(self, ver):
        super().__init__()
        self.ver = ver

        # ResNet-101 backbone
        if ver == 'rn101':
            backbone = tv.models.resnet101(pretrained=True)
            self.conv1 = backbone.conv1
            self.bn1 = backbone.bn1
            self.relu = backbone.relu
            self.maxpool = backbone.maxpool
            self.layer1 = backbone.layer1
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4
            self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # MiT-b1 backbone
        if ver == 'mitb1':
            self.backbone = SegformerModel.from_pretrained('nvidia/mit-b1')
            self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # MiT-b2 backbone
        if ver == 'mitb2':
            self.backbone = SegformerModel.from_pretrained("mit-b2", local_files_only=True)
            self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img):
        # ResNet-101 backbone
        if self.ver == 'rn101':
            x = (img - self.mean) / self.std
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            s4 = x
            x = self.layer2(x)
            s8 = x
            x = self.layer3(x)
            s16 = x
            x = self.layer4(x)
            s32 = x
            return {'s4': s4, 's8': s8, 's16': s16, 's32': s32}

        # MiT-b1 backbone
        if self.ver == 'mitb1':
            x = (img - self.mean) / self.std
            x = self.backbone(x, output_hidden_states=True).hidden_states
            s4 = x[0]
            s8 = x[1]
            s16 = x[2]
            s32 = x[3]
            return {'s4': s4, 's8': s8, 's16': s16, 's32': s32}

        # MiT-b2 backbone
        if self.ver == 'mitb2':
            x = (img - self.mean) / self.std
            x = self.backbone(x, output_hidden_states=True).hidden_states
            s4 = x[0]
            s8 = x[1]
            s16 = x[2]
            s32 = x[3]
            return {'s4': s4, 's8': s8, 's16': s16, 's32': s32}

class Decoder(nn.Module):
    def __init__(self, backbone_channels):
        super().__init__()
        self.channel_adjust = nn.ModuleDict({
            key: DKSM(
                in_channels=ch * 2,
                out_channels=256,
                q_dim=32,
                num_Hs=4
            ) for key, ch in backbone_channels.items()
        })

        self.fuse_s32 = nn.Sequential(
            ConvRelu(256, 256, 3, padding=1),
            DKSM(256, 256),
            CBAM(256)
        )
        self.fuse_s16 = nn.Sequential(
            ConvRelu(256 + 256, 256, 3, padding=1),
            DKSM(256, 256),
            CBAM(256)
        )
        self.fuse_s8 = nn.Sequential(
            ConvRelu(256 + 256, 256, 3, padding=1),
            DKSM(256, 256),
            CBAM(256)
        )
        self.fuse_s4 = nn.Sequential(
            ConvRelu(256 + 256, 256, 3, padding=1),
            DKSM(256, 256),
            CBAM(256)
        )

        self.final_conv = nn.Sequential(
            ConvRelu(256, 128, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Conv(128, 2, 3, padding=1)
        )

    def forward(self, app_feats, mo_feats):
        adjusted = {}
        for key in ['s4', 's8', 's16', 's32']:
            combined = torch.cat([app_feats[key], mo_feats[key]], dim=1)
            adjusted[key] = self.channel_adjust[key](combined)

        x = self.fuse_s32(adjusted['s32'])
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        x = torch.cat([x, adjusted['s16']], dim=1)
        x = self.fuse_s16(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        x = torch.cat([x, adjusted['s8']], dim=1)
        x = self.fuse_s8(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        x = torch.cat([x, adjusted['s4']], dim=1)
        x = self.fuse_s4(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        return self.final_conv(x)


class HFF(nn.Module):
    def __init__(self, in_channels_list, fusion_channels=256):
        super().__init__()
        self.s4_channels, self.s8_channels, self.s16_channels, self.s32_channels = in_channels_list
        self.fusion_channels = fusion_channels
        self.lateral_convs = nn.ModuleDict({
            's4': nn.Sequential(
                nn.Conv2d(self.s4_channels, fusion_channels, kernel_size=1),
                nn.BatchNorm2d(fusion_channels),
                nn.ReLU(inplace=True)
            ),
            's8': nn.Sequential(
                nn.Conv2d(self.s8_channels, fusion_channels, kernel_size=1),
                nn.BatchNorm2d(fusion_channels),
                nn.ReLU(inplace=True)
            ),
            's16': nn.Sequential(
                nn.Conv2d(self.s16_channels, fusion_channels, kernel_size=1),
                nn.BatchNorm2d(fusion_channels),
                nn.ReLU(inplace=True)
            ),
            's32': nn.Sequential(
                nn.Conv2d(self.s32_channels, fusion_channels, kernel_size=1),
                nn.BatchNorm2d(fusion_channels),
                nn.ReLU(inplace=True)
            )
        })
        self.up_modules = nn.ModuleDict({
            's32_to_s16': nn.Sequential(
                nn.ConvTranspose2d(fusion_channels, fusion_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(fusion_channels),
                nn.ReLU(inplace=True)
            ),
            's16_to_s8': nn.Sequential(
                nn.ConvTranspose2d(fusion_channels, fusion_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(fusion_channels),
                nn.ReLU(inplace=True)
            ),
            's8_to_s4': nn.Sequential(
                nn.ConvTranspose2d(fusion_channels, fusion_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(fusion_channels),
                nn.ReLU(inplace=True)
            )
        })
        self.fusion_modules = nn.ModuleDict({
            's16_fusion': nn.Sequential(
                CBAM(fusion_channels * 2),
                nn.Conv2d(fusion_channels * 2, fusion_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(fusion_channels),
                nn.ReLU(inplace=True)
            ),
            's8_fusion': nn.Sequential(
                CBAM(fusion_channels * 2),
                nn.Conv2d(fusion_channels * 2, fusion_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(fusion_channels),
                nn.ReLU(inplace=True)
            ),
            's4_fusion': nn.Sequential(
                CBAM(fusion_channels * 2),
                nn.Conv2d(fusion_channels * 2, fusion_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(fusion_channels),
                nn.ReLU(inplace=True)
            ),
            's32_fusion': nn.Sequential(
                CBAM(fusion_channels),
                nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(fusion_channels),
                nn.ReLU(inplace=True)
            )
        })
        self.channel_mapping = nn.ModuleDict({
            's4': nn.Conv2d(fusion_channels, self.s4_channels, kernel_size=1),
            's8': nn.Conv2d(fusion_channels, self.s8_channels, kernel_size=1),
            's16': nn.Conv2d(fusion_channels, self.s16_channels, kernel_size=1),
            's32': nn.Conv2d(fusion_channels, self.s32_channels, kernel_size=1)
        })

    def forward(self, feature_maps):
        laterals = {}
        for key in ['s4', 's8', 's16', 's32']:
            laterals[key] = self.lateral_convs[key](feature_maps[key])

        fused_s32 = self.fusion_modules['s32_fusion'](laterals['s32'])
        up_s32_to_s16 = self.up_modules['s32_to_s16'](fused_s32)
        if up_s32_to_s16.shape[-1] != laterals['s16'].shape[-1] or up_s32_to_s16.shape[-2] != laterals['s16'].shape[-2]:
            up_s32_to_s16 = F.interpolate(up_s32_to_s16, size=laterals['s16'].shape[2:],
                                          mode='bilinear', align_corners=True)
        concat_s16 = torch.cat([up_s32_to_s16, laterals['s16']], dim=1)
        fused_s16 = self.fusion_modules['s16_fusion'](concat_s16)
        up_s16_to_s8 = self.up_modules['s16_to_s8'](fused_s16)
        if up_s16_to_s8.shape[-1] != laterals['s8'].shape[-1] or up_s16_to_s8.shape[-2] != laterals['s8'].shape[-2]:
            up_s16_to_s8 = F.interpolate(up_s16_to_s8, size=laterals['s8'].shape[2:],
                                         mode='bilinear', align_corners=True)
        concat_s8 = torch.cat([up_s16_to_s8, laterals['s8']], dim=1)
        fused_s8 = self.fusion_modules['s8_fusion'](concat_s8)
        up_s8_to_s4 = self.up_modules['s8_to_s4'](fused_s8)
        if up_s8_to_s4.shape[-1] != laterals['s4'].shape[-1] or up_s8_to_s4.shape[-2] != laterals['s4'].shape[-2]:
            up_s8_to_s4 = F.interpolate(up_s8_to_s4, size=laterals['s4'].shape[2:],
                                        mode='bilinear', align_corners=True)
        concat_s4 = torch.cat([up_s8_to_s4, laterals['s4']], dim=1)
        fused_s4 = self.fusion_modules['s4_fusion'](concat_s4)

        final_features = {}
        final_features['s4'] = self.channel_mapping['s4'](fused_s4)
        final_features['s8'] = self.channel_mapping['s8'](fused_s8)
        final_features['s16'] = self.channel_mapping['s16'](fused_s16)
        final_features['s32'] = self.channel_mapping['s32'](fused_s32)
        return final_features

class VOS(nn.Module):
    def __init__(self, ver):
        super().__init__()
        self.app_encoder = Encoder(ver)
        self.mo_encoder = Encoder(ver)
        if ver == 'rn101':
            backbone_channels = {'s4': 256, 's8': 512, 's16': 1024, 's32': 2048}
        elif ver == 'mitb1':
            backbone_channels = {'s4': 64, 's8': 128, 's16': 320, 's32': 512}
        elif ver == 'mitb2':
            backbone_channels = {'s4': 64, 's8': 128, 's16': 320, 's32': 512}
        else:
            raise ValueError(f"Unsupported backbone version: {ver}")
        self.decoder = Decoder(backbone_channels)

class CMNF(nn.Module):
    def __init__(self, ver, aos):
        super().__init__()
        self.vos = VOS(ver)
        self.aos = aos
        self.dropout = nn.Dropout(0.1)

        if ver == 'rn101':
            channels = {'s16': 1024, 's32': 2048}
        elif ver == 'mitb2':
            channels = {'s16': 320, 's32': 512}
        else:
            raise ValueError(f"Unsupported backbone: {ver}")
        if ver == 'rn101':
            in_channels_list = [256, 512, 1024, 2048]
        elif ver == 'mitb2':
            in_channels_list = [64, 128, 320, 512]
        self.app_hff_fusion = HFF(in_channels_list, fusion_channels=256)
        self.mo_hff_fusion = HFF(in_channels_list, fusion_channels=256)

    def forward(self, imgs, flows):
        B, L, _, H1, W1 = imgs.size()
        _, _, _, H2, W2 = flows.size()

        s = 512
        imgs = F.interpolate(imgs.view(B * L, -1, H1, W1), size=(s, s), mode='bicubic').view(B, L, -1, s, s)
        flows = F.interpolate(flows.view(B * L, -1, H2, W2), size=(s, s), mode='bicubic').view(B, L, -1, s, s)

        # for each frame
        score_lst = []
        mask_lst = []
        for i in range(L):


            # adaptive output selection off
            if B != 1 or not self.aos:

                # query frame prediction
                app_feats = self.vos.app_encoder(imgs[:, i])
                mo_feats = self.vos.mo_encoder(flows[:, i])

                # Apply Dropout to all features
                app_feats['s4'] = self.dropout(app_feats['s4'])
                app_feats['s8'] = self.dropout(app_feats['s8'])
                app_feats['s16'] = self.dropout(app_feats['s16'])
                app_feats['s32'] = self.dropout(app_feats['s32'])
                mo_feats['s4'] = self.dropout(mo_feats['s4'])
                mo_feats['s8'] = self.dropout(mo_feats['s8'])
                mo_feats['s16'] = self.dropout(mo_feats['s16'])
                mo_feats['s32'] = self.dropout(mo_feats['s32'])

                app_fused_feats = self.app_hff_fusion(app_feats)
                mo_fused_feats = self.mo_hff_fusion(mo_feats)

                app_feats = app_fused_feats
                mo_feats = mo_fused_feats

                app_feats = {
                    's4': app_feats['s4'],
                    's8': app_feats['s8'],
                    's16': app_feats['s16'],
                    's32': app_feats['s32']
                }
                mo_feats = {
                    's4': mo_feats['s4'],
                    's8': mo_feats['s8'],
                    's16': mo_feats['s16'],
                    's32': mo_feats['s32']
                }

                score = self.vos.decoder(app_feats, mo_feats)
                score = F.interpolate(score, size=(H1, W1), mode='bicubic')

            # adaptive output selection on
            if B == 1 and self.aos:
                # query frame prediction
                app_feats = self.vos.app_encoder(imgs[:, i])
                mo_feats_img = self.vos.mo_encoder(imgs[:, i])
                mo_feats_flow = self.vos.mo_encoder(flows[:, i])
                score_img = self.vos.decoder(app_feats, mo_feats_img)
                score_flow = self.vos.decoder(app_feats, mo_feats_flow)

                h = 0.1
                pred_seg = torch.softmax(score_img, dim=1)
                conf_img = torch.sum((h - pred_seg[pred_seg < h]) ** 2) ** 0.5
                pred_seg = torch.softmax(score_flow, dim=1)
                conf_flow = torch.sum((h - pred_seg[pred_seg < h]) ** 2) ** 0.5
                w = (conf_img > conf_flow).float()
                score = w * score_img + (1 - w) * score_flow
                score = F.interpolate(score, size=(H1, W1), mode='bicubic')

            # store soft scores
            if B != 1:
                score_lst.append(score)

            # store hard masks
            if B == 1:
                pred_seg = torch.softmax(score, dim=1)
                pred_mask = torch.max(pred_seg, dim=1, keepdim=True)[1]
                mask_lst.append(pred_mask)

        # generate output
        output = {}
        if B != 1:
            output['scores'] = torch.stack(score_lst, dim=1)
        if B == 1:
            output['masks'] = torch.stack(mask_lst, dim=1)
        return output