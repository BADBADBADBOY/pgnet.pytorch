import torch
import torch.nn as nn
# from pgnet_model_backbone import ResNet
from pgnet_model_backbone import resnet50
from pgnet_model_neck import PGFPN
from pgnet_model_head import PGHead

class PGnet(nn.Module):
    def __init__(self):
        super(PGnet,self).__init__()
#         self.backbone = ResNet(layers=50)
        self.backbone = resnet50(load_url=False)
        self.neck = PGFPN(256)
        self.head = PGHead(128)
    def forward(self,x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

# img = torch.rand(1,3,512,512)
# model = PGnet()
# out = model(img)
# print(out['f_score'].shape)
# print(out['f_border'].shape)
# print(out['f_char'].shape)
# print(out['f_direction'].shape)