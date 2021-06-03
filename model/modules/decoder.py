import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, dataset, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()

        low_level_inplanes = 32
        in_channels = 2048
            

    
        if dataset == "NTID":
            limbsNum = 18
        else:
            limbsNum = 13

        self.conv1 = nn.Conv2d(low_level_inplanes, 96, 1, bias=False)
        self.bn1 = BatchNorm(96)
        #self.relu = nn.ReLU()
        self.relu = nn.Hardswish()

        self.conv2 = nn.Conv2d(in_channels, 256, 1, bias=False)
        self.bn2 = BatchNorm(256)
        self.last_conv = nn.Sequential(nn.Conv2d(352, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       # nn.ReLU(),
                                       nn.Hardswish(),
                                       
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       # nn.ReLU(),
                                       nn.Hardswish(),
                                       
                                       nn.Dropout(0.1),
                                       # nn.Conv2d(256, num_classes + 1, kernel_size = 1, stride = 1))
                                       nn.Conv2d(256, num_classes + 1 , kernel_size = 2, stride=2),
#                                       nn.Hardswish(),
#                                       nn.MaxPool2d((2,2),stride=2)
                                       # nn.Conv2d(256, num_classes+5+1, kernel_size=1, stride=1)) # Use in case of extacting the bounding box
        )

        self.output = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(256, num_classes+1),
        )        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        
        #x = self.conv2(x)
        #x = self.bn2(x)
        #x = self.relu(x)

        low_level_feat = self.maxpool(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)
        # x = self.output(x)
        #x = self.maxpool(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(dataset, num_classes, backbone, BatchNorm):
    return Decoder(dataset ,num_classes, backbone, BatchNorm)
