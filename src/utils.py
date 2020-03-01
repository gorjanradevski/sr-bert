import torch
from datasets import SCENE_WIDTH


def elementwise_distances(X: torch.Tensor):
    return torch.triu(torch.abs(torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)))


def real_distance(inds, labs, attn_mask, xy: str):
    # Obtain normal dist
    dist_normal = real_distance_single(inds, labs, attn_mask)
    if xy == "y":
        return dist_normal.sum()
    # Get flipped indices
    pad_ids = torch.where(labs == -100)
    labs_flipped = torch.abs(SCENE_WIDTH - labs)
    labs_flipped[pad_ids] = -100
    dist_flipped = real_distance_single(inds, labs_flipped, attn_mask)

    dist = torch.min(dist_normal, dist_flipped)

    return dist


def real_distance_single(inds, labs, attn_mask):
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

    return dist


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
