from operator import xor
import torch
from scene_layouts.datasets import SCENE_WIDTH_TEST
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score

index2pose_hb0 = {
    23: 0,
    34: 0,
    45: 0,
    51: 0,
    52: 0,
    53: 1,
    54: 1,
    55: 1,
    56: 1,
    57: 1,
    24: 2,
    25: 2,
    26: 2,
    27: 2,
    28: 2,
    29: 3,
    30: 3,
    31: 3,
    32: 3,
    33: 3,
    35: 4,
    36: 4,
    37: 4,
    38: 4,
    39: 4,
    40: 5,
    41: 5,
    42: 5,
    43: 5,
    44: 5,
    46: 6,
    47: 6,
    48: 6,
    49: 6,
    50: 6,
}
index2pose_hb1 = {
    58: 0,
    69: 0,
    80: 0,
    86: 0,
    87: 0,
    88: 1,
    89: 1,
    90: 1,
    91: 1,
    92: 1,
    59: 2,
    60: 2,
    61: 2,
    62: 2,
    63: 2,
    64: 3,
    65: 3,
    66: 3,
    67: 3,
    68: 3,
    70: 4,
    71: 4,
    72: 4,
    73: 4,
    74: 4,
    75: 5,
    76: 5,
    77: 5,
    78: 5,
    79: 5,
    81: 6,
    82: 6,
    83: 6,
    84: 6,
    85: 6,
}
index2expression_hb0 = {
    23: 0,
    53: 0,
    24: 0,
    29: 0,
    35: 0,
    40: 0,
    46: 0,
    34: 1,
    54: 1,
    25: 1,
    30: 1,
    36: 1,
    41: 1,
    47: 1,
    45: 2,
    55: 2,
    26: 2,
    31: 2,
    37: 2,
    42: 2,
    48: 2,
    51: 3,
    56: 3,
    27: 3,
    32: 3,
    38: 3,
    43: 3,
    49: 3,
    52: 4,
    57: 4,
    28: 4,
    33: 4,
    39: 4,
    44: 4,
    50: 4,
}
index2expression_hb1 = {
    58: 0,
    88: 0,
    59: 0,
    64: 0,
    70: 0,
    75: 0,
    81: 0,
    69: 1,
    89: 1,
    60: 1,
    65: 1,
    71: 1,
    76: 1,
    82: 1,
    80: 2,
    90: 2,
    61: 2,
    66: 2,
    72: 2,
    77: 2,
    83: 2,
    86: 3,
    91: 3,
    62: 3,
    67: 3,
    73: 3,
    78: 3,
    84: 3,
    87: 4,
    92: 4,
    63: 4,
    68: 4,
    74: 4,
    79: 4,
    85: 4,
}


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

    def find_common_pos(self, pred_clips, gt_clips, pred_pos, gt_pos):
        # https://github.com/uvavision/Text2Scene/blob/master/lib/modules/abstract_evaluator.py#L309
        common_preds = []
        common_gts = []
        for i in range(pred_clips):
            if pred_clips[i] in gt_clips:
                common_preds.append([pred_pos[i]["x"], pred_pos[i]["y"]])
                index_gt = gt_clips.index(pred_clips[i])
                common_gts.append([gt_pos[index_gt]["x"], gt_pos[index_gt]["y"]])
        
        return common_preds, common_gts


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
    # BECAUSE OF THE SIMILARITY FUNCTION
    dist = torch.min(dist_normal, dist_flipped)

    return dist, (dist == dist_flipped).float()


def abs_distance_single(x_inds, x_labs, y_inds, y_labs, attn_mask):
    # REBUTTAL: Normalize coordinates
    x_inds_norm = x_inds.clone()
    y_inds_norm = y_inds.clone()
    x_labs_norm = x_labs.clone().float()
    y_labs_norm = y_labs.clone().float()
    # Obtain dist for X and Y
    dist_x = torch.pow(x_inds_norm - x_labs_norm, 2).float()
    dist_y = torch.pow(y_inds_norm - y_labs_norm, 2).float()
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
    # https://github.com/uvavision/Text2Scene/blob/master/lib/abstract_utils.py#L366
    # dist = torch.exp(-0.5 * dist / 0.2)

    return dist


def elementwise_distances(X: torch.Tensor, Y: torch.Tensor):
    X_inds_norm = X.clone().float()
    Y_inds_norm = Y.clone().float()
    x_dist = torch.pow(
        torch.unsqueeze(X_inds_norm, 1) - torch.unsqueeze(X_inds_norm, 2), 2
    ).float()
    y_dist = torch.pow(
        torch.unsqueeze(Y_inds_norm, 1) - torch.unsqueeze(Y_inds_norm, 2), 2
    ).float()

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
    # REBUTTAL: Gaussian kernel
    # https://github.com/uvavision/Text2Scene/blob/master/lib/abstract_utils.py#L366
    # dist = torch.exp(-0.5 * dist / 0.2)

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
            self.abs_dist.sum() / np.count_nonzero(self.abs_dist), decimals=1
        )

    def get_rel_dist(self):
        return np.round(
            self.rel_dist.sum() / np.count_nonzero(self.rel_dist), decimals=1
        )

    def get_o_acc(self):
        return np.round(
            (self.flip_acc.sum() / np.count_nonzero(self.flip_acc)) * 100, decimals=1
        )

    def get_abs_error_bar(self):
        return np.round(
            np.std(self.abs_dist, ddof=1) / np.sqrt(np.count_nonzero(self.abs_dist)),
            decimals=1,
        )

    def get_rel_error_bar(self):
        return np.round(
            np.std(self.rel_dist, ddof=1) / np.sqrt(np.count_nonzero(self.rel_dist)),
            decimals=1,
        )

    def get_o_acc_error_bar(self):
        return np.round(
            (np.std(self.flip_acc, ddof=1) / np.sqrt(np.count_nonzero(self.flip_acc)))
            * 100,
            decimals=1,
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


class ClipartsPredictionEvaluator:
    def __init__(self, dataset_size, visual2index):
        self.dataset_size = dataset_size
        self.visual2index = visual2index
        self.predictions = np.zeros((self.dataset_size, len(self.visual2index)))
        self.targets = np.zeros((self.dataset_size, len(self.visual2index)))
        self.index = 0

    def update_counters(self, preds, targets):
        batch_size = preds.shape[0]
        self.predictions[self.index : self.index + batch_size] = preds
        self.targets[self.index : self.index + batch_size] = targets
        self.index += batch_size

    def reset_counters(self):
        self.predictions = np.zeros((self.dataset_size, len(self.visual2index)))
        self.targets = np.zeros((self.dataset_size, len(self.visual2index)))
        self.index = 0

    def f1_score(self):
        return f1_score(self.targets, self.predictions, average="micro")

    def per_object_pr(self):
        targets_obj = np.concatenate(
            [self.targets[:, :23], self.targets[:, 93:]], axis=1
        )
        preds_obj = np.concatenate(
            [self.predictions[:, :23], self.predictions[:, 93:]], axis=1
        )
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets_obj, preds_obj, average="micro"
        )
        return (
            np.round(precision * 100, decimals=1),
            np.round(recall * 100, decimals=1),
            np.round(f1 * 100, decimals=1),
        )

    def posses_expressions_accuracy(self):
        num_targets_hbo = len(
            [target for target in self.targets[:, 23:58] if target.sum() > 0]
        )
        num_targets_hb1 = len(
            [target for target in self.targets[:, 58:93] if target.sum() > 0]
        )
        targets_pose = np.zeros((num_targets_hbo + num_targets_hb1,))
        targets_expr = np.zeros((num_targets_hbo + num_targets_hb1,))
        predicts_pose = np.zeros((num_targets_hbo + num_targets_hb1,))
        predicts_expr = np.zeros((num_targets_hbo + num_targets_hb1,))
        index = 0
        for i in range(self.targets.shape[0]):
            if self.targets[i, 23:58].sum() > 0:
                # Get target index
                target_index = self.targets[i, 23:58].argmax() + 23
                # Get predictions index
                pred_index = self.predictions[i, 23:58].argmax() + 23
                # Update pose arrays
                targets_pose[index] = index2pose_hb0[target_index]
                predicts_pose[index] = index2pose_hb0[pred_index]
                # Update expression arrays
                targets_expr[index] = index2expression_hb0[target_index]
                predicts_expr[index] = index2expression_hb0[pred_index]
                # Update index
                index += 1
            if self.targets[i, 58:93].sum() > 0:
                # Get target index
                target_index = self.targets[i, 58:93].argmax() + 58
                # Get predictions index
                pred_index = self.predictions[i, 58:93].argmax() + 58
                # Update pose arrays
                targets_pose[index] = index2pose_hb1[target_index]
                predicts_pose[index] = index2pose_hb1[pred_index]
                # Update expression arrays
                targets_expr[index] = index2expression_hb1[target_index]
                predicts_expr[index] = index2expression_hb1[pred_index]
                # Update index
                index += 1

        return (
            np.round(accuracy_score(targets_pose, predicts_pose) * 100, decimals=1),
            np.round(accuracy_score(targets_expr, predicts_expr) * 100, decimals=1),
        )

