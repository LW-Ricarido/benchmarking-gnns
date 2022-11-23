import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        # print("inputs", inputs.shape)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()

        dist.addmm(1, -2, inputs, inputs.t())

        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        # print("dist_ap", dist_ap)
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)

        y = torch.ones_like(dist_an)

        return self.ranking_loss(dist_an, dist_ap, y)
