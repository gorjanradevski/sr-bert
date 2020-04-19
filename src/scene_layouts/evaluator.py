import torch
from scene_layouts.datasets import SCENE_WIDTH_TEST


class Evaluator:
    def __init__(self, total_elements):
        self.abs_dist_x = 0.0
        self.abs_dist_y = 0.0
        self.rel_dist_x = 0.0
        self.rel_dist_y = 0.0
        self.f_acc = 0.0
        self.total_elements = total_elements

    def update_metrics(self, x_out, x_lab, y_out, y_lab, f_out, f_lab, attn_mask):
        # Update absolute distance
        dist_x_abs, flips = self.abs_distance(
            x_out, x_lab, attn_mask, check_flipped=True
        )
        self.abs_dist_x += dist_x_abs.item()
        self.abs_dist_y += self.abs_distance(
            y_out, y_lab, attn_mask, check_flipped=False
        ).item()
        # Update relative distance
        self.rel_dist_x += self.relative_distance(x_out, x_lab, attn_mask).item()
        self.rel_dist_y += self.relative_distance(y_out, y_lab, attn_mask).item()
        # Update flip accuracy
        self.f_acc += self.flip_acc(f_out, f_lab, attn_mask, flips).item()

    def reset_metrics(self):
        self.abs_dist_x = 0.0
        self.abs_dist_y = 0.0
        self.rel_dist_x = 0.0
        self.rel_dist_y = 0.0
        self.f_acc = 0.0

    def get_abs_x(self):
        return round(self.abs_dist_x / self.total_elements, 2)

    def get_abs_y(self):
        return round(self.abs_dist_y / self.total_elements, 2)

    def get_rel_x(self):
        return round(self.rel_dist_x / self.total_elements, 2)

    def get_rel_y(self):
        return round(self.rel_dist_y / self.total_elements, 2)

    def get_f_acc(self):
        return round(self.f_acc / self.total_elements * 100, 2)

    @staticmethod
    def elementwise_distances(X: torch.Tensor):
        return torch.abs(torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2))

    @staticmethod
    def flip_scene(labs: torch.Tensor):
        # Get flipped indices
        pad_ids = torch.where(labs < 0)
        labs_flipped = torch.abs(SCENE_WIDTH_TEST - labs)
        labs_flipped[pad_ids] = -100

        return labs_flipped

    @classmethod
    def abs_distance(cls, inds, labs, attn_mask, check_flipped: bool = False):
        # Obtain normal dist
        dist_normal = cls.abs_distance_single(inds, labs, attn_mask)
        if check_flipped is False:
            return dist_normal.sum()

        labs_flipped = cls.flip_scene(labs)
        dist_flipped = cls.abs_distance_single(inds, labs_flipped, attn_mask)

        dist = torch.min(dist_normal, dist_flipped)

        return dist.sum(), (dist == dist_flipped).float()

    @staticmethod
    def abs_distance_single(inds, labs, attn_mask):
        # Obtain the distance matrix
        dist = torch.abs(inds - labs).float()
        # Remove the distance for the padding tokens
        dist = dist * attn_mask
        # Remove the distance for the masked tokens (During training)
        mask_masked = torch.ones_like(labs)
        mask_masked[torch.where(labs < 0)] = 0.0
        dist = dist * mask_masked
        # Obtain average distance for each scene without considering the padding tokens
        dist = dist.sum(-1) / attn_mask.sum(-1)

        return dist

    @classmethod
    def relative_distance(cls, inds, labs, attn_mask):
        dist = torch.abs(
            cls.elementwise_distances(inds) - cls.elementwise_distances(labs)
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
        dist = dist.sum(-1) / attn_mask.sum(-1).unsqueeze(-1)
        # Obtain average distance for each scene without considering the padding tokens
        dist = dist.sum(-1) / attn_mask.sum(-1)

        return dist.sum()

    @staticmethod
    def flip_acc(inds, labs, attn_mask, flips):
        inds = torch.abs(inds - flips.unsqueeze(-1))

        return ((inds == labs).sum(-1).float() / attn_mask.sum(-1)).sum()
