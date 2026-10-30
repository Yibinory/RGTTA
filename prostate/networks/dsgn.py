from torch import nn
import torch



class DomainSpecificGroupNorm(nn.Module):
    """
    Domain-Specific Group Normalization (DSGN) layer.
    """
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