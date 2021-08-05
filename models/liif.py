import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord


@register('liif')
class LIIF(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = models.make(encoder_spec)

        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            if self.feat_unfold:
                imnet_in_dim *= 9
            imnet_in_dim += 3 # attach coord
            if self.cell_decode:
                imnet_in_dim += 3
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        # FEATURE UNFOLDING
        # Enrich the information of every pixel (latent code) with the information of its neighbours 3x3
        # it concatenates the neighbouring and as a result it multyplies per 9 the dimensionality of the channels
        if self.feat_unfold:
            feat = F.unfold(feat.view(feat.shape[1], feat.shape[2], feat.shape[3], feat.shape[4]), 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3], feat.shape[4])

        # LOCAL ENSEMBLING
        # Achieve four latent codes for independently predicting the signal of one coordinate
        # the four predictions then are merged to achieve a continuous transition
        if self.local_ensemble:
            vz_lst = [-1, 1]
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, vz_lst, eps_shift = [0], [0], [0], 0

        # field radius (global: [-1, 1])
        # calculate distance between each latent code (0.333333)
        rz = 2 / feat.shape[-3] / 2
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-3:], flatten=False).cuda() \
            .permute(3,0,1,2) \
            .unsqueeze(0).expand(feat.shape[0], 3, *feat.shape[-3:])

        preds = []
        areas = []

        # loop for every latent code
        # the pixels we want to predict are in between 4 latent codes (vx and vy coordinates)
        for vz in vz_lst:
            for vx in vx_lst:
                for vy in vy_lst:
                    # clone coords, the coordinates of the pixels we want to predict
                    coord_ = coord.clone()
                    coord_[:, :, 0] += vz * rz + eps_shift
                    coord_[:, :, 1] += vx * rx + eps_shift
                    coord_[:, :, 2] += vy * ry + eps_shift
                    # return the same tensor but with min and max thresholds
                    # values under min stay as min and values over max stay as max
                    coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                    # transform features to a grid of same size of coord, using its coordinates to get the nearest
                    #feat.shape = torch.Size([1, 576, 3, 3])
                    # coord_.flip(-1).unsqueeze(1) = torch.Size([1, 1, 36, 2])
                    q_feat = F.grid_sample(
                        feat, coord_.flip(-1).unsqueeze(1).unsqueeze(1), # I think they do a flip to convert W,H to H,W, unsqueeze to reshape to same size of feat
                        mode='nearest', align_corners=False)[:, :, 0, 0, :] \
                        .permute(0, 2, 1)

                    #feat_coord.shape = torch.Size([1, 2, 3, 3])
                    #coord_.shape = torch.Size([1, 36, 2])
                    q_coord = F.grid_sample(
                        feat_coord, coord_.flip(-1).unsqueeze(1).unsqueeze(1),
                        mode='nearest', align_corners=False)[:, :, 0, 0, :] \
                        .permute(0, 2, 1)
                    #q_coord.shape = torch.Size([1, 36, 2])
                    rel_coord = coord - q_coord
                    rel_coord[:, :, 0] *= feat.shape[-3]
                    rel_coord[:, :, 1] *= feat.shape[-2]
                    rel_coord[:, :, 2] *= feat.shape[-1]
                    #concatenate coordiantes with features to have everything in a unique vector for every pixel that we will pas to the mlp
                    inp = torch.cat([q_feat, rel_coord], dim=-1)

                    if self.cell_decode:
                        rel_cell = cell.clone()
                        rel_cell[:, :, 0] *= feat.shape[-3]
                        rel_cell[:, :, 1] *= feat.shape[-2]
                        rel_cell[:, :, 2] *= feat.shape[-1]
                        inp = torch.cat([inp, rel_cell], dim=-1)

                    bs, q = coord.shape[:2]
                    #call mlp, input=580 features and output=3 features rgb
                    pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                    preds.append(pred)

                    area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1] * rel_coord[:, :, 2])
                    areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[7]; areas[7] = t
            t = areas[1]; areas[1] = areas[6]; areas[6] = t
            t = areas[2]; areas[2] = areas[5]; areas[5] = t
            t = areas[3]; areas[3] = areas[4]; areas[4] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
