from typing import Optional, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from copy import deepcopy
from .unet import UNet


def unfold_w_center(x, kernel_size, dilation):
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(x, kernel_size=kernel_size, padding=padding, dilation=dilation)

    unfolded_x = unfolded_x.reshape(x.size(0), x.size(1), -1, x.size(2), x.size(3))

    return unfolded_x


class MultiLevelModel(nn.Module):
    def __init__(self, in_channels, num_levels=2, num_classes=1):
        super().__init__()

        self.num_levels = num_levels

        model_list = [UNet(in_channels=in_channels, out_channels=num_classes, feat_dim=128)]
        for _ in range(num_levels - 1):
            model_list.append(deepcopy(model_list[0]))
        self.model = nn.ModuleList(model_list)

    def feature_propogate(self, feat):
        B, C, H, W = feat.shape

        # Value transformation
        feat_value = feat

        # Similarity calculation
        feat_value_neighbor = unfold_w_center(feat_value, kernel_size=5, dilation=2)
        feat_neighbor = unfold_w_center(feat, kernel_size=5, dilation=2)

        corr = torch.einsum("bchw,bcnhw->bhwn", feat, feat_neighbor)
        corr = torch.clamp(corr, min=0)
        corr = F.softmax(corr, dim=-1)

        feat = torch.einsum("bcnhw,bhwn->bchw", feat_value_neighbor, corr)

        return feat.view(B, C, H, W)

    def forward_ensemble(self, x):
        model_ensemble = deepcopy(self.model[0])
        model_ensemble = self.ensemble_weights(model_ensemble)

        return model_ensemble(x)

    def ensemble_weights(self, model):
        with torch.no_grad():
            for key in model.state_dict().keys():
                temp = torch.zeros_like(model.state_dict()[key].data)
                for i in range(self.num_levels):
                    temp += self.model[i].state_dict()[key].data * (1 / self.num_levels)
                model.state_dict()[key].data.copy_(temp)

        return model

    def forward(self, x):
        out = {}

        for i in range(self.num_levels):
            out_i = self.model[i](x)
            for k in out_i.keys():
                out[f"{k}_{i+1}"] = out_i[k]

            out[f"feature_prop_{i+1}"] = self.feature_propogate(out[f"feature_{i+1}"])
            out[f"logits_prop_{i+1}"] = self.model[i].classifier(out[f"feature_prop_{i+1}"])

        out["logits"] = torch.stack([out[f"logits_{i+1}"] for i in range(self.num_levels)]).mean(0)

        return out
