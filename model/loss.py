import torch
import torch.nn.functional as F

def weightedBCELoss(pos_output, neg_output, pos_weight=1):
    loss = torch.nn.BCEWithLogitsLoss()
    return pos_weight * loss(pos_output, torch.ones_like(pos_output)) + \
            loss(neg_output, torch.zeros_like(neg_output))
def nll_loss(output, target):
    return F.nll_loss(output, target)
