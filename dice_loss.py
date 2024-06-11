import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth = 0.001):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred[:, 0, :, :]
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum, B_sum = iflat.sum(), tflat.sum()
        return 1-((2. * intersection + self.smooth) / (A_sum + B_sum + self.smooth))

class Sensitivity(nn.Module):
    def __init__(self, smooth = 0.001):
        super(Sensitivity, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        # pred=F.softmax(pred,dim=1)
        pred = pred[:, 0, :, :]
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum, B_sum = iflat.sum(), tflat.sum()
        return (intersection + self.smooth) / (B_sum + self.smooth)
    
class StraightnessLoss(nn.Module):
    def __init__(self):
        super(StraightnessLoss, self).__init__()
    
    def forward(self, pred):
        # dx, dy = torch.gradient(pred, dim = [-2,-1])
        # loss = torch.sqrt(dx**2 + dy**2).mean() 
        loss = torch.sum(torch.abs(pred[:, :, :-1] - pred[:, :, 1:])) + torch.sum(torch.abs(pred[:, :-1, :] - pred[:, 1:, :]))
        return loss