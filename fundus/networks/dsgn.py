from torch import nn
import torch

class _DomainSpecificBatchNorm(nn.Module):
    _version = 2

    def __init__(self, num_features, num_domains, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_DomainSpecificBatchNorm, self).__init__()
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_domains)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, domain_label):
        self._check_input_dim(x)
        outs = []
        for i in range(x.size(0)):
            bn = self.bns[domain_label[i].item()]
            outs.append(bn(x[i].unsqueeze(0)))
        out = torch.cat(outs, dim=0)
        return out, domain_label



class DomainSpecificBatchNorm2d(_DomainSpecificBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class DomainSpecificGroupNorm(nn.Module):

    def __init__(self, num_groups, num_channels, num_domains, eps=1e-5, affine=True):
        super(DomainSpecificGroupNorm, self).__init__()
        self.gns = nn.ModuleList([
            nn.GroupNorm(num_groups, num_channels, eps, affine) for _ in range(num_domains)
        ])

    def forward(self, x, domain_label):

        outs = []
        for i in range(x.size(0)):
            gn = self.gns[domain_label[i].item()]
            outs.append(gn(x[i].unsqueeze(0)))
        out = torch.cat(outs, dim=0)
        return out, domain_label