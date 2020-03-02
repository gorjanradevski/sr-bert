import torch
from datasets import SCENE_WIDTH


def elementwise_distances(X: torch.Tensor, l_type: str):
    return (
        torch.triu(torch.pow(torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2), 2))
        if l_type == "mse"
        else torch.triu(torch.abs(torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)))
    )


def flip_scene(labs):
    # Get flipped indices
    pad_ids = torch.where(labs < 0)
    labs_flipped = torch.abs(SCENE_WIDTH - labs)
    labs_flipped[pad_ids] = -100

    return labs_flipped


def real_distance(
    inds, labs, attn_mask, check_flipped: bool = False, l_type: str = "mse"
):
    # Obtain normal dist
    dist_normal = real_distance_single(inds, labs, attn_mask, l_type)
    if check_flipped is False:
        return dist_normal.sum()

    labs_flipped = flip_scene(labs)
    dist_flipped = real_distance_single(inds, labs_flipped, attn_mask, l_type)

    dist = torch.min(dist_normal, dist_flipped)

    return dist.sum()


def real_distance_single(inds, labs, attn_mask, l_type):
    # Obtain the distance matrix
    dist = (
        torch.pow(inds - labs, 2).float()
        if l_type == "mse"
        else torch.abs(inds - labs).float()
    )
    # Remove the distance for the padding tokens
    dist = dist * attn_mask
    # Remove the distance for the masked tokens (During training)
    mask_masked = torch.ones_like(inds)
    mask_masked[torch.where(labs < 0)] = 0.0
    dist = dist * mask_masked
    # Obtain average distance for each scene without considering the padding tokens
    # and the sep token
    dist = dist.sum(-1) / (attn_mask.sum(-1) - 1)

    return dist


def relative_distance(
    inds, labs, attn_mask, check_flipped: bool = False, l_type: str = "mse"
):
    dist_normal = relative_distance_single(inds, labs, attn_mask, l_type)
    if check_flipped is False:
        return dist_normal.sum()

    labs_flipped = flip_scene(labs)
    dist_flipped = relative_distance_single(inds, labs_flipped, attn_mask, l_type)

    dist = torch.min(dist_normal, dist_flipped)

    return dist.sum()


def relative_distance_single(inds, labs, attn_mask, l_type):
    dist = torch.abs(
        elementwise_distances(inds, l_type) - elementwise_distances(labs, l_type)
    ).float()
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
    # Obtain average distance for each scene without considering the padding tokens and
    # the sep token
    dist = dist.sum(-1).sum(-1) / (attn_mask.sum(-1) - 1)

    return dist
