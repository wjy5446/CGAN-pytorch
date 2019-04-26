import torch
import torch.nn as nn

class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.target_real_label = target_real_label
        self.target_fake_label = target_fake_label
        self.loss = nn.BCELoss()
        self.Tensor = torch.FloatTensor

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.Tensor(prediction.size()).fill_(self.target_real_label)
        else:
            target_tensor = self.Tensor(prediction.size()).fill_(self.target_fake_label)
        return target_tensor

    def forward(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)