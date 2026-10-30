from torch import nn
import torch
from networks.resnet import resnet34, resnet18, resnet50, resnet101, resnet152
import torch.nn.functional as F
from networks.dsgn import DomainSpecificGroupNorm

def normalization(planes, norm='gn', num_domains=None):
    if norm == 'bn':
        m = nn.BatchNorm2d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(1, planes)
    elif norm == 'in':
        m = nn.InstanceNorm2d(planes)
    elif norm == 'dsgn':
        # m = DomainSpecificGroupNorm(1, planes, num_domains)
        m = DomainSpecificGroupNorm(16, planes, num_domains)
    else:
        raise ValueError('Normalization type {} is not supporter'.format(norm))
    return m

class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)

        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        return self.bn(F.relu(cat_p))


class Rec_UnetBlock(nn.Module):
    def __init__(self, planes, norm='dsgn', num_domains=4):
        super(Rec_UnetBlock, self).__init__()
        # self.x_conv = nn.Conv2d(x_in, x_out, 1)
        self.tr_conv = nn.ConvTranspose2d(planes, planes//2, 2, stride=2)

        self.bn = normalization(planes//2, norm=norm, num_domains=num_domains)

    def forward(self, x, domain_label=None):
        x = self.tr_conv(x)
        x, _ = self.bn(F.relu(x), domain_label)
        return x


class ResUnet(nn.Module):
    def __init__(self, resnet='resnet34', num_classes=2, pretrained=False, mixstyle_layers=[],
                 random_type=None, p=0.5):
        super().__init__()
        if resnet == 'resnet34':
            base_model = resnet34
            feature_channels = [64, 64, 128, 256, 512]
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
            feature_channels = [64, 256, 512, 1024, 2048]
        elif resnet == 'resnet101':
            base_model = resnet101
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')

        self.mixstyle_layers = mixstyle_layers
        self.res = base_model(pretrained=pretrained, mixstyle_layers=mixstyle_layers, p=p)

        self.num_classes = num_classes

        self.up1 = UnetBlock(feature_channels[4], feature_channels[3], 256)
        self.up2 = UnetBlock(256, feature_channels[2], 256)
        self.up3 = UnetBlock(256, feature_channels[1], 256)
        self.up4 = UnetBlock(256, feature_channels[0], 256)

        self.up5 = nn.ConvTranspose2d(256, 32, 2, stride=2)
        self.bnout = nn.BatchNorm2d(32)

        self.seg_head = nn.Conv2d(32, self.num_classes, 1)

    def forward(self, input, is_mix=True):
        x, sfs = self.res(input, is_mix)
        x = F.relu(x)

        x = self.up1(x, sfs[3])
        x = self.up2(x, sfs[2])
        x = self.up3(x, sfs[1])
        x = self.up4(x, sfs[0])
        x = self.up5(x)
        head_input = F.relu(self.bnout(x))

        seg_output = self.seg_head(head_input)
        return seg_output
    

class Encoder(nn.Module):
    def __init__(self, resnet='resnet34', pretrained=False, mixstyle_layers=[],
                 random_type=None, p=0.5):
        super().__init__()
        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        elif resnet == 'resnet101':
            base_model = resnet101
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')

        self.mixstyle_layers = mixstyle_layers
        self.res = base_model(pretrained=pretrained, mixstyle_layers=mixstyle_layers, p=p)

    def forward(self, input, is_mix=False):
        x, sfs = self.res(input, is_mix)
        return x, sfs
    
class Decoder(nn.Module):
    def __init__(self, feature_channels=[64, 64, 128, 256, 512], num_classes=1):
        super().__init__()
        self.up1 = UnetBlock(feature_channels[4], feature_channels[3], 256)
        self.up2 = UnetBlock(256, feature_channels[2], 256)
        self.up3 = UnetBlock(256, feature_channels[1], 256)
        self.up4 = UnetBlock(256, feature_channels[0], 256)

        self.up5 = nn.ConvTranspose2d(256, 32, 2, stride=2)
        self.bnout = nn.BatchNorm2d(32)

        self.seg_head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x, sfs):
        x = F.relu(x)

        x = self.up1(x, sfs[3])
        x = self.up2(x, sfs[2])
        x = self.up3(x, sfs[1])
        x = self.up4(x, sfs[0])
        x = self.up5(x)
        head_input = F.relu(self.bnout(x))

        seg_output = self.seg_head(head_input)
        return seg_output
    
class Rec_Decoder(nn.Module):
    def __init__(self, feature_channels=[32, 64, 128, 256, 512], num_classes=3, norm='dsgn', num_domains=4):
        super().__init__()
        self.up1 = Rec_UnetBlock(feature_channels[4], norm=norm, num_domains=num_domains)
        self.up2 = Rec_UnetBlock(feature_channels[3], norm=norm, num_domains=num_domains)
        self.up3 = Rec_UnetBlock(feature_channels[2], norm=norm, num_domains=num_domains)
        self.up4 = Rec_UnetBlock(feature_channels[1], norm=norm, num_domains=num_domains)
        self.up5 = Rec_UnetBlock(feature_channels[0], norm=norm, num_domains=num_domains)

        self.rec_head = nn.Conv2d(16, num_classes, 1)

    def forward(self, x, domain_label):
        x = self.up1(x, domain_label)
        x = self.up2(x, domain_label)
        x = self.up3(x, domain_label)
        x = self.up4(x, domain_label)
        x = self.up5(x, domain_label)
        return self.rec_head(x)



class Rec_ResUnet(nn.Module):
    def __init__(self, resnet='resnet34', num_classes=2, pretrained=False, mixstyle_layers=[],
                p=0.5, norm='bn', activation='relu', n=32, rec_num_classes=3):
        super().__init__()
        if resnet == 'resnet34':
            base_model = resnet34
            feature_channels = [64, 64, 128, 256, 512]
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
            feature_channels = [64, 256, 512, 1024, 2048]
        elif resnet == 'resnet101':
            base_model = resnet101
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')

        self.mixstyle_layers = mixstyle_layers
        self.res = base_model(pretrained=pretrained, mixstyle_layers=mixstyle_layers, p=p)

        self.num_classes = num_classes

        self.rec_1 = UnetBlock(feature_channels[4], feature_channels[3], 256)
        self.rec_2 = UnetBlock(256, feature_channels[2], 256)
        self.rec_3 = UnetBlock(256, feature_channels[1], 256)
        self.rec_4 = UnetBlock(256, feature_channels[0], 256)

        self.rec_5 = nn.ConvTranspose2d(256, 32, 2, stride=2)
        self.rec_bnout = nn.BatchNorm2d(32)
        self.rec_head = nn.Conv2d(32, rec_num_classes, 1)


        self.up1 = UnetBlock(feature_channels[4], feature_channels[3], 256)
        self.up2 = UnetBlock(256, feature_channels[2], 256)
        self.up3 = UnetBlock(256, feature_channels[1], 256)
        self.up4 = UnetBlock(256, feature_channels[0], 256)

        self.up5 = nn.ConvTranspose2d(256, 32, 2, stride=2)
        self.bnout = nn.BatchNorm2d(32)

        self.seg_head = nn.Conv2d(32, self.num_classes, 1)
        

    def forward(self, input):
        input1 = input.clone()

        # mix style x
        x, sfs = self.res(input)
        x = F.relu(x)

        x1 = self.rec_1(x, sfs[3])
        x2 = self.rec_2(x1, sfs[2])
        x3 = self.rec_3(x2, sfs[1])
        x4 = self.rec_4(x3, sfs[0])
        x5 = self.rec_5(x4)
        rec_x = F.relu(self.rec_bnout(x5))

        rec_x = self.rec_head(rec_x)

        x = self.up1(x, sfs[3])
        x = self.up2(x, sfs[2])
        x = self.up3(x, sfs[1])
        x = self.up4(x, sfs[0])
        x = self.up5(x)
        head_input = F.relu(self.bnout(x))

        rec_seg_output = self.seg_head(head_input)

        # original x
        original_x, orig_sfs = self.res(input1, is_mix=False)
        original_x = F.relu(original_x)

        original_rec_x = self.rec_1(original_x, orig_sfs[3])
        original_rec_x = self.rec_2(original_rec_x, orig_sfs[2])
        original_rec_x = self.rec_3(original_rec_x, orig_sfs[1])
        original_rec_x = self.rec_4(original_rec_x, orig_sfs[0])
        original_rec_x = self.rec_5(original_rec_x)
        orig_rec_x = F.relu(self.rec_bnout(original_rec_x))
        orig_rec_x = self.rec_head(orig_rec_x) 

        orig_x = self.up1(original_x, orig_sfs[3])
        orig_x = self.up2(orig_x, orig_sfs[2])
        orig_x = self.up3(orig_x, orig_sfs[1])
        orig_x = self.up4(orig_x, orig_sfs[0])
        orig_x = self.up5(orig_x)
        orig_head_input = F.relu(self.bnout(orig_x))

        orig_seg_output = self.seg_head(orig_head_input)

        return rec_seg_output, rec_x, orig_seg_output, orig_rec_x

    def close(self):
        for sf in self.sfs:
            sf.remove()


if __name__ == "__main__":
    model = ResUnet(resnet='resnet34', num_classes=2, pretrained=False, mixstyle_layers=['layer1'])
    print(model.res)
    model.cuda().eval()
    input = torch.rand(2, 3, 512, 512).cuda()
    seg_output, x_iw_list, iw_loss = model(input)
    print(seg_output.size())

