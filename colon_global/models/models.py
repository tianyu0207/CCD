"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NormalizedLinear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(NormalizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        with torch.no_grad():
            w = self.weight / self.weight.data.norm(keepdim=True, dim=0)
        return F.linear(x, w, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )



class ContrastiveModel(nn.Module):
    def __init__(self, backbone,  mlp_number, cls_head_number, head='mlp', features_dim=128):
        super(ContrastiveModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.head = head
        self.class_num = 4
        self.linear = nn.Linear(self.backbone_dim, self.backbone_dim)
        self.relu = nn.ReLU()
        self.mlp_number = mlp_number
        self.cls_head_number = cls_head_number
        if head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, features_dim)

        elif head == 'mlp':

            self.contrastive_head = nn.Sequential(
                    nn.Linear(self.backbone_dim, self.backbone_dim),
                    nn.ReLU(), nn.Linear(self.backbone_dim, features_dim))
            if self.cls_head_number ==2:
                self.classification_head = nn.Sequential(nn.Linear(self.backbone_dim, features_dim),
                                                         nn.ReLU(),
                                                         NormalizedLinear(features_dim, self.class_num))
            else:
                self.classification_head = nn.Sequential(
                    NormalizedLinear(self.backbone_dim, self.class_num))

        else:
            raise ValueError('Invalid head {}'.format(head))

    def forward(self, x):

        if self.mlp_number == 3:
            out = self.relu(self.linear(self.backbone(x)))
        else:
            out = self.backbone(x)
        features = self.contrastive_head(out)
        features = F.normalize(features, dim=1)

        # original_img_out = out[:32,:]
        # print(original_img_out.shape)

        logits = self.classification_head(out)

        return features, logits



class ClusteringModel(nn.Module):
    def __init__(self, backbone, nclusters, nheads=1): # back bone is encoder
        super(ClusteringModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.nheads = nheads
        assert(isinstance(self.nheads, int))
        assert(self.nheads > 0)
        self.cluster_head = nn.ModuleList([nn.Linear(self.backbone_dim, nclusters) for _ in range(self.nheads)])

    def forward(self, x, forward_pass='default'):
        if forward_pass == 'default':
            features = self.backbone(x)
            out = [cluster_head(features) for cluster_head in self.cluster_head]

        elif forward_pass == 'backbone':
            out = self.backbone(x)

        elif forward_pass == 'head':
            out = [cluster_head(x) for cluster_head in self.cluster_head]

        elif forward_pass == 'return_all':
            features = self.backbone(x)
            out = {'features': features, 'output': [cluster_head(features) for cluster_head in self.cluster_head]}
        
        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))        

        return out
