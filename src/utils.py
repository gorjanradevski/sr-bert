import torch


def elementwise_distances(X: torch.Tensor):
    return torch.triu(torch.abs(torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)))


def real_distance(inds, labs, attn_mask):
    # Obtain the distance matrix
    dist = torch.abs(inds - labs).float()
    # Remove the distance for the padding tokens
    dist = dist * attn_mask
    # Remove the distance for the masked tokens (During training)
    mask_masked = torch.ones_like(inds)
    mask_masked[torch.where(labs < 0)] = 0.0
    dist = dist * mask_masked
    # Obtain average distance for each scene without considering the padding tokens
    dist = dist.sum(-1) / attn_mask.sum(-1)

    return dist.sum()


def relative_distance(inds, labs, attn_mask):
    dist = torch.abs(elementwise_distances(inds) - elementwise_distances(labs)).float()
    # Remove the distance for the padding tokens - Both columns and rows
    dist = (
        dist
        * attn_mask.unsqueeze(1).expand(dist.size())
        * attn_mask.unsqueeze(-1).expand(dist.size())
    )
    # Remove the distance for the masked tokens (During training)
    mask_masked = torch.ones_like(labs)
    mask_masked[torch.where(labs < 0)] = 0.0
    dist = (
        dist
        * mask_masked.unsqueeze(1).expand(dist.size())
        * mask_masked.unsqueeze(-1).expand(dist.size())
    )
    # Obtain average distance for each scene without considering the padding tokens
    dist = dist.sum(-1).sum(-1) / attn_mask.sum(-1)

    return dist.sum()
