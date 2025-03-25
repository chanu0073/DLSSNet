import torch
import torch.nn as nn

class reg_loss(nn.Module):
    def __init__(self, reg_factor):
        super().__init__()

        self.reg_factor = reg_factor
    
    def forward(self, weight):
        WtW = torch.bmm(weight, torch.transpose(weight, -1, -2))
        I = torch.eye(WtW.size(1)).to(weight.device)
        loss = self.reg_factor *  torch.norm(WtW - I, dim = (-2,-1), p='fro')
        return loss.mean()