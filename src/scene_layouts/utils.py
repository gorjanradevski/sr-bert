import torch
from scene_layouts.datasets import SCENE_WIDTH_TEST
from typing import Dict
import re


def elementwise_distances(X: torch.Tensor):
    return torch.abs(torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2))


def flip_scene(labs):
    # Get flipped indices
    pad_ids = torch.where(labs < 0)
    labs_flipped = torch.abs(SCENE_WIDTH_TEST - labs)
    labs_flipped[pad_ids] = -100

    return labs_flipped


def real_distance(inds, labs, attn_mask, check_flipped: bool = False):
    # Obtain normal dist
    dist_normal = real_distance_single(inds, labs, attn_mask)
    if check_flipped is False:
        return dist_normal.sum()

    labs_flipped = flip_scene(labs)
    dist_flipped = real_distance_single(inds, labs_flipped, attn_mask)

    dist = torch.min(dist_normal, dist_flipped)

    return dist.sum(), (dist == dist_flipped).float()


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
    dist = dist.sum(-1) / attn_mask.sum(-1).unsqueeze(-1)
    # Obtain average distance for each scene without considering the padding tokens
    dist = dist.sum(-1) / attn_mask.sum(-1)
    return dist.sum()


def flip_acc(inds, labs, attn_mask, flips):
    inds = torch.abs(inds - flips.unsqueeze(-1))
    return ((inds == labs).sum(-1).float() / attn_mask.sum(-1)).sum()


def get_reference_indices(visual2index: Dict[str, int], reference_type: str):
    if reference_type == "animals":
        return [v for k, v in visual2index.items() if k.startswith("a")]
    elif reference_type == "food":
        return [v for k, v in visual2index.items() if k.startswith("e")]
    elif reference_type == "toys":
        return [v for k, v in visual2index.items() if k.startswith("t")]
    elif reference_type == "people":
        return [v for k, v in visual2index.items() if k.startswith("hb")]
    elif reference_type == "sky":
        return [v for k, v in visual2index.items() if k.startswith("s")]
    elif reference_type == "large":
        return [v for k, v in visual2index.items() if k.startswith("p")]
    elif reference_type == "clothes":
        return [v for k, v in visual2index.items() if k.startswith("c")]
    elif reference_type is None:
        return []
    else:
        raise ValueError(f"{reference_type} doesn't exist!")


def contains_word(word, text):
    # https://stackoverflow.com/a/45587730/3987085
    pattern = r"(^|[^\w]){}([^\w]|$)".format(word)
    pattern = re.compile(pattern, re.IGNORECASE)
    matches = re.search(pattern, text)
    return bool(matches)


# https://academicguides.waldenu.edu/writingcenter/grammar/prepositions
explicit_rels = [
    "on",
    "next to",
    "above",
    "over",
    "below",
    "behind",
    "along",
    "through",
    "in",
    "in front of",
    "near",
    "beyond",
    "with",
    "by",
    "inside of",
    "on top of",
    "down",
    "up",
    "beneath",
    "inside",
    "left",
    "right",
    "under",
    "across from",
    "underneath",
    "atop",
    "across",
    "beside",
    "around",
    "outside",
    "next",
    "against",
    "at",
    "between",
    "front",
    "aside",
]
