import torch
import torch.nn as nn

from mmcv.cnn import normal_init
from ..centers.transformer import CA, CALayer

class MCTransAuxHead(nn.Module):
    def __init__(self,
                 d_model=240,
                 d_ffn=1024,
                 dropout=0.1,
                 act="relu",
                 n_head=8,
                 n_layers=4,
                 num_classes=1,
                 in_channles=[64, 64, 128, 256, 512],
                 proj_idxs=(2, 3, 4),
                 losses=None
                 ):
        super(MCTransAuxHead, self).__init__()
        self.in_channles = in_channles
        self.proj_idxs = proj_idxs

        ca_layer = CALayer(d_model=d_model,
                           d_ffn=d_ffn,
                           dropout=dropout,
                           activation=act,
                           n_heads=n_head)

        self.ca = CA(att_layer=ca_layer,
                     n_layers=n_layers,
                     n_category=num_classes,
                     d_model=d_model)

        self.head = nn.Sequential(nn.Linear(num_classes*d_model, d_model),
                                  nn.Linear(d_model, num_classes))

    def forward(self, inputs):
        # flatten
        inputs = [inputs[idx] for idx in self.proj_idxs]
        inputs_flatten = [item.flatten(2).transpose(1, 2) for item in inputs]
        inputs_flatten = torch.cat(inputs_flatten, 1)
        # ca
        outputs = self.ca(inputs_flatten)
        logits = self.head(outputs.flatten(1))
       
        return logits

    def init_weights(self):
        normal_init(self.head, mean=0, std=0.01)