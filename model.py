import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50


class Network(nn.Module):
    def __init__(self, num_outputs):
        super(Network, self).__init__()
        self.network = resnet50()
        self.network = torch.nn.Sequential(*list(self.network.children())[:-1])
        self.projection_original_features = nn.Linear(2048, 128)
        self.act_h = nn.ReLU()
        self.last_layer = nn.Linear(128, num_outputs)
        self.act_out = nn.Softmax(dim = 1)


    def forward(self, images):
        features = self.network(images)
        features = features.view(-1, 2048)
        features = self.projection_original_features(features)
        features = self.act_h(features)
        features = self.act_h(features)
        features = self.last_layer(features)
        output = self.act_out(features)
        return output
