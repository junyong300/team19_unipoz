from model.modules.backbone import resnet
from model.modules.backbone.efficientnet import _efficientnet

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    if backbone == 'efficient':
        return _efficientnet('efficientnet_b0',True,True)
    else:
        raise NotImplementedError
        
