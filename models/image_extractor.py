import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import ResNet, BasicBlock
from models.gfnet import GFNet, GFNetPyramid
from functools import partial
#from .vision_transformer import vit_base


def get_image_extractor(arch='resnet18', pretrained=True):
    if 'resnet' in arch:
        resnet = Get_resnet(arch=arch, pretrained=pretrained)
        return resnet
    elif 'gfn' in arch:
        gfn = Get_gfn(arch=arch, pretrained=pretrained)
        return gfn
    else:
        raise NotImplementedError


class Get_resnet(nn.Module):
    def __init__(self, arch='resnet18', pretrained=True):
        super(Get_resnet, self).__init__()

        if arch == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            # model.fc = nn.Sequential()
            # self.model = model
            modules = list(model.children())[:-2]  # 返回resnet子模块
            self.model = nn.Sequential(*modules)
        elif arch == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            model.fc = nn.Sequential()
            self.model = model

    def forward(self, x):
        return self.model(x)


class Get_gfn(nn.Module):
    def __init__(self, arch='gfnet-h-ti', pretrained=True):
        super(Get_gfn, self).__init__()

        if arch == 'gfnet-h-ti':
            gfn = GFNetPyramid(
                img_size=224,
                patch_size=4, embed_dim=[64, 128, 256, 512], depth=[3, 3, 10, 3],
                mlp_ratio=[4, 4, 4, 4],
                norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1,
            )
            if pretrained:
                model_weight = torch.load('./models/gfnet-h-ti.pth')['model']
                gfn.load_state_dict(model_weight)
            gfn.head = nn.Sequential()
            self.gfn = gfn
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.gfn(x)
        B, N, C = x.size()
        x = x.view(B, 7, 7, C).permute(0, 3, 1, 2)  # return B*C*7*7
        return x

class ResNet18_conv(ResNet):
    def __init__(self):
        super(ResNet18_conv, self).__init__(BasicBlock, [2, 2, 2, 2])

    def forward(self, x):
        # change forward here
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
