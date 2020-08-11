import torch
from scene_layouts.datasets import SCENE_WIDTH_TEST
import numpy as np


class Evaluator:
    def __init__(self, total_elements):
        self.total_elements = total_elements
        self.abs_dist = np.zeros(self.total_elements)
        self.rel_dist = np.zeros(self.total_elements)
        self.o_acc = np.zeros(self.total_elements)
        self.index = 0

    def update_metrics(self, x_out, x_lab, y_out, y_lab, o_out, o_lab, attn_mask):
        # Update absolute distance
        batch_size = x_out.size()[0]
        abs_dist, flips = abs_distance(
            x_out, x_lab, y_out, y_lab, attn_mask, check_flipped=True
        )
        self.abs_dist[self.index : self.index + batch_size] = abs_dist.cpu().numpy()
        # Update relative distance
        self.rel_dist[self.index : self.index + batch_size] = (
            relative_distance(x_out, x_lab, y_out, y_lab, attn_mask).cpu().numpy()
        )
        # Update flip accuracy
        self.o_acc[self.index : self.index + batch_size] += (
            flip_acc(o_out, o_lab, attn_mask, flips).cpu().numpy()
        )
        self.index += batch_size

    def reset_metrics(self):
        self.abs_dist = np.zeros(self.total_elements)
        self.rel_dist = np.zeros(self.total_elements)
        self.o_acc = np.zeros(self.total_elements)
        self.index = 0

    def get_abs_dist(self):
        return np.round(self.abs_dist.mean(), decimals=1)

    def get_rel_dist(self):
        return np.round(self.rel_dist.mean(), decimals=1)

    def get_o_acc(self):
        return np.round(self.o_acc.mean() * 100, decimals=1)

    def get_abs_error_bar(self):
        return np.round(
            np.std(self.abs_dist, ddof=1) / np.sqrt(self.total_elements), decimals=1
        )

    def get_rel_error_bar(self):
        return np.round(
            np.std(self.rel_dist, ddof=1) / np.sqrt(self.total_elements), decimals=1
        )

    def dump_results(self, abs_dump_path: str, rel_dump_path: str):
        np.save(abs_dump_path, self.abs_dist, allow_pickle=False)
        np.save(rel_dump_path, self.rel_dist, allow_pickle=False)


def flip_scene(labs: torch.Tensor):
    # Get flipped indices
    pad_ids = torch.where(labs < 0)
    labs_flipped = torch.abs(SCENE_WIDTH_TEST - labs)
    labs_flipped[pad_ids] = -100

    return labs_flipped


def abs_distance(
    x_inds, x_labs, y_inds, y_labs, attn_mask, check_flipped: bool = False
):
    # Obtain normal dist
    dist_normal = abs_distance_single(x_inds, x_labs, y_inds, y_labs, attn_mask)
    if check_flipped is False:
        return dist_normal

    x_labs_flipped = flip_scene(x_labs)
    dist_flipped = abs_distance_single(
        x_inds, x_labs_flipped, y_inds, y_labs, attn_mask
    )

    dist = torch.min(dist_normal, dist_flipped)

    return dist, (dist == dist_flipped).float()


def abs_distance_single(x_inds, x_labs, y_inds, y_labs, attn_mask):
    # REBUTTAL: Normalize coordinates
    # x_inds /= 500
    # y_inds /= 400
    # x_labs = x_labs.float() / 500
    # y_labs = y_labs.float() / 400
    # Obtain dist for X and Y
    dist_x = torch.pow(x_inds - x_labs, 2).float()
    dist_y = torch.pow(y_inds - y_labs, 2).float()
    dist = torch.sqrt(dist_x + dist_y + (torch.ones_like(dist_x) * 1e-15))
    # Remove the distance for the padding tokens
    dist = dist * attn_mask
    # Remove the distance for the masked tokens (During training)
    mask_masked = torch.ones_like(x_labs)
    mask_masked[torch.where(x_labs < 0)] = 0.0
    dist = dist * mask_masked
    # Obtain average distance for each scene without considering the padding tokens
    dist = dist.sum(-1) / attn_mask.sum(-1)
    # REBUTTAL: Gaussian kernel
    # dist = torch.exp(-dist / 0.2)

    return dist


def elementwise_distances(X: torch.Tensor, Y: torch.Tensor):
    x_dist = torch.pow(torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2), 2).float()
    y_dist = torch.pow(torch.unsqueeze(Y, 1) - torch.unsqueeze(Y, 2), 2).float()

    return torch.sqrt(x_dist + y_dist + (torch.ones_like(x_dist) * 1e-15))


def relative_distance(x_inds, x_labs, y_inds, y_labs, attn_mask):
    dist = torch.abs(
        elementwise_distances(x_inds, y_inds) - elementwise_distances(x_labs, y_labs)
    ).float()
    # Remove the distance for the padding tokens - Both columns and rows
    dist = (
        dist
        * attn_mask.unsqueeze(1).expand(dist.size())
        * attn_mask.unsqueeze(-1).expand(dist.size())
    )
    # Remove the distance for the masked tokens (During training)
    mask_masked = torch.ones_like(x_labs)
    mask_masked[torch.where(x_labs < 0)] = 0.0
    dist = (
        dist
        * mask_masked.unsqueeze(1).expand(dist.size())
        * mask_masked.unsqueeze(-1).expand(dist.size())
    )
    dist = dist.sum(-1) / attn_mask.sum(-1).unsqueeze(-1)
    # Obtain average distance for each scene without considering the padding tokens
    dist = dist.sum(-1) / attn_mask.sum(-1)

    return dist


def flip_acc(inds, labs, attn_mask, flips):
    inds = torch.abs(inds - flips.unsqueeze(-1))

    return (inds == labs).sum(-1).float() / attn_mask.sum(-1)


class QaEvaluator:
    def __init__(self, total_elements):
        self.total_elements = total_elements
        self.abs_dist = np.zeros(self.total_elements)
        self.rel_dist = np.zeros(self.total_elements)
        self.flip_acc = np.zeros(self.total_elements)
        self.index = 0

    def update_metrics(self, x_out, x_lab, y_out, y_lab, o_out, o_lab, mask):
        # Update absolute distance
        batch_size = x_out.size()[0]
        self.abs_dist[self.index : self.index + batch_size] = (
            abs_distance_qa(x_out, x_lab, y_out, y_lab, mask).cpu().numpy()
        )
        # Update relative distance
        self.rel_dist[self.index : self.index + batch_size] = (
            relative_distance_qa(x_out, x_lab, y_out, y_lab, mask).cpu().numpy()
        )
        self.flip_acc[self.index : self.index + batch_size] = (
            flip_acc_qa(o_out, o_lab, mask).cpu().numpy()
        )

        self.index += batch_size

    def reset_metrics(self):
        self.abs_dist = np.zeros(self.total_elements)
        self.rel_dist = np.zeros(self.total_elements)
        self.flip_acc = np.zeros(self.total_elements)
        self.index = 0

    def get_abs_dist(self):
        return np.round(
            self.abs_dist.sum() / np.count_nonzero(self.abs_dist), decimals=2
        )

    def get_rel_dist(self):
        return np.round(
            self.rel_dist.sum() / np.count_nonzero(self.rel_dist), decimals=2
        )

    def get_o_acc(self):
        return np.round(
            (self.flip_acc.sum() / np.count_nonzero(self.flip_acc)) * 100, decimals=2
        )

    def get_abs_error_bar(self):
        return np.round(
            np.std(self.abs_dist, ddof=1) / np.sqrt(np.count_nonzero(self.abs_dist)),
            decimals=2,
        )

    def get_rel_error_bar(self):
        return np.round(
            np.std(self.rel_dist, ddof=1) / np.sqrt(np.count_nonzero(self.rel_dist)),
            decimals=2,
        )

    def get_o_acc_error_bar(self):
        return np.round(
            (np.std(self.flip_acc, ddof=1) / np.sqrt(np.count_nonzero(self.flip_acc)))
            * 100,
            decimals=2,
        )


def relative_distance_qa(x_inds, x_labs, y_inds, y_labs, mask):
    dist = torch.abs(
        elementwise_distances(x_inds, y_inds) - elementwise_distances(x_labs, y_labs)
    ).float()
    dist = dist * mask.unsqueeze(-1).expand(dist.size())
    dist = dist.mean(-1)
    dist = dist.sum(-1) / (mask.sum(-1) + 1e-15)

    return dist


def abs_distance_qa(x_inds, x_labs, y_inds, y_labs, mask):
    # Obtain dist for X and Y
    dist_x = torch.pow(x_inds - x_labs, 2).float()
    dist_y = torch.pow(y_inds - y_labs, 2).float()
    dist = torch.sqrt(dist_x + dist_y + (torch.ones_like(dist_x) * 1e-15))
    # Remove the distance from the non-target elements
    dist = dist * mask
    # Obtain average distance for each scene without considering the padding tokens
    dist = dist.sum(-1) / (mask.sum(-1) + 1e-15)

    return dist


def flip_acc_qa(inds, labs, mask):
    acc = ((inds == labs).float() + 1e-15) * mask
    return acc.sum(-1) / (mask.sum(-1) + 1e-15)
