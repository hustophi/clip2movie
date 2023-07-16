import torch
from torchmetrics.classification import MulticlassAUROC
def recall(output, threshold=0.6):
    pos_output, neg_output = output
    with torch.no_grad():
        pos_prob = torch.sigmoid(pos_output)    #B * num_clip * 1
        #TP = (pos_prob > threshold).reshape(-1).nonzero().numel()
        TP = int(torch.sum(pos_prob >= threshold))
        return TP / torch.numel(pos_output)
'''
def val_inBatchrecall(output, Bm):    #validation时Bc == Bm
    with torch.no_grad():
        Bc, B_, _ = output.shape   #B_ = num_clip*Bm
        num_clip = B_ // Bm
        output = output.reshape(-1, Bm)
        pred = torch.argmax(output, dim=1)
        target = torch.arange(Bm).repeat_interleave(num_clip)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
        return correct / len(target)
'''
def auc(output, thresholds=None):
    pos_output, neg_output = output
    with torch.no_grad():
        pos_prob, neg_prob = torch.sigmoid(pos_output).reshape(-1, 1), torch.sigmoid(neg_output).reshape(-1, 1)
        preds = torch.cat([pos_prob, neg_prob], dim=0)
        target = torch.tensor([1] * pos_output.shape[0]+[0] * neg_output.shape[0])
        metric = BinaryAUROC(thresholds=thresholds)
        return metric(preds, target).item()

def test_recall(output):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        target = torch.arange(1100).repeat_interleave(10)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
        return correct / len(target)
def test_auc(output):   #output shape: (1100*10, 1100)
    with torch.no_grad():
        target = torch.arange(1100).repeat_interleave(10)
        metric = MulticlassAUROC(num_classes=1100, average="macro", thresholds=None)
    return metric(output, target).item()
'''
def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def top_k_acc(output, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        target = torch.arange(1100).repeat_interleave(10)
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
'''
