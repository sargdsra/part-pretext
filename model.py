import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50


class Network(nn.Module):
    def __init__(self, num_outputs, use_pretrained = False):
        super(Network, self).__init__()
        self.network = resnet50(pretrained = use_pretrained)
        self.network = torch.nn.Sequential(*list(self.network.children())[:-1])
        self.projection_original_features = nn.Linear(2048, 128)
        self.act_h = nn.ReLU()
        self.last_layer = nn.Linear(128, num_outputs)


    def forward(self, images):
        features = self.network(images)
        features = features.view(-1, 2048)
        features = self.projection_original_features(features)
        features = self.act_h(features)
        features = self.act_h(features)
        output = self.last_layer(features)
        return output
