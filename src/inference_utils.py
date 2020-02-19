import torch


def elementwise_distances(X: torch.Tensor):
    return torch.triu(torch.abs(torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)))


def real_distance(x_ind, x_lab, attn_mask):
    dist = torch.abs(x_ind - x_lab).float()
    dist = dist * attn_mask
    dist = torch.sum(dist, dim=1) / torch.sum(attn_mask, dim=1)
    return dist.sum().item()


def relative_distance(x_ind, x_lab, attn_mask):
    dist = torch.abs(
        elementwise_distances(x_ind) - elementwise_distances(x_lab)
    ).float()
    dist = torch.sum(dist, dim=1) * attn_mask
    dist = torch.sum(dist, dim=1) / torch.sum(attn_mask, dim=1)

    return dist.sum().item()
