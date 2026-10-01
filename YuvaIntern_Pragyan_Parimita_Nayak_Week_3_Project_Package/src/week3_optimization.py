# YuvaIntern Week 3
# Pragyan Parimita Nayak
# Structured channel pruning + fine-tuning + TorchScript export

import torch
from torch import nn

class TinyShelfDetector(nn.Module):
    def __init__(self, channels=(16,32,64,96)):
        super().__init__()
        c1,c2,c3,c4=channels
        self.features=nn.Sequential(
            nn.Conv2d(3,c1,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(c1,c2,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(c2,c3,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(c3,c4,3,padding=1),nn.ReLU(),nn.MaxPool2d(2))
        self.head=nn.Conv2d(c4,5,1)
    def forward(self,x): return self.head(self.features(x))

def structured_prune(baseline, keep_ratio=0.75):
    """Remove complete filters using L1 filter importance."""
    old=[baseline.features[0],baseline.features[3],
         baseline.features[6],baseline.features[9]]
    counts=[max(4, int(round(c.out_channels*keep_ratio))) for c in old]
    indices=[]
    for conv,k in zip(old,counts):
        score=conv.weight.detach().abs().sum(dim=(1,2,3))
        indices.append(torch.topk(score,k).indices.sort().values)

    pruned=TinyShelfDetector(tuple(counts))
    new=[pruned.features[0],pruned.features[3],
         pruned.features[6],pruned.features[9]]
    prev_in=torch.arange(3)
    for o,n,idx in zip(old,new,indices):
        with torch.no_grad():
            n.weight.copy_(o.weight[idx][:,prev_in])
            n.bias.copy_(o.bias[idx])
        prev_in=idx
    with torch.no_grad():
        pruned.head.weight.copy_(baseline.head.weight[:,prev_in])
        pruned.head.bias.copy_(baseline.head.bias)
    return pruned

# After fine-tuning, create a deployment-friendly inference artifact:
# scripted=torch.jit.trace(pruned, torch.zeros(1,3,128,128))
# scripted=torch.jit.optimize_for_inference(scripted)
# scripted.save("optimized_pruned_shelf_detector_torchscript.pt")
