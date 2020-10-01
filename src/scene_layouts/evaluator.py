from operator import xor
import torch
from scene_layouts.datasets import SCENE_WIDTH_TEST
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score


class Evaluator:
    def __init__(self, total_elements):
        self.total_elements = total_elements
        self.abs_dist = np.zeros(self.total_elements)
        self.rel_dist = np.zeros(self.total_elements)
        self.o_acc = np.zeros(self.total_elements)
        self.u_obj_coord = np.zeros(self.total_elements)
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
        # Update U-obj coord
        self.u_obj_coord[self.index : self.index + batch_size] = (
            u_obj_coord(x_out, x_lab, y_out, y_lab, attn_mask).cpu().numpy()
        )

        self.index += batch_size

    def reset_metrics(self):
        self.abs_dist = np.zeros(self.total_elements)
        self.rel_dist = np.zeros(self.total_elements)
        self.o_acc = np.zeros(self.total_elements)
        self.u_obj_coord = np.zeros(self.total_elements)
        self.index = 0

    def get_abs_dist(self):
        return np.round(self.abs_dist.mean(), decimals=1)

    def get_rel_dist(self):
        return np.round(self.rel_dist.mean(), decimals=1)

    def get_o_acc(self):
        return np.round(self.o_acc.mean() * 100, decimals=1)

    def get_u_obj_coord(self):
        return np.round(self.u_obj_coord.mean(), decimals=3)

    def get_abs_error_bar(self):
        return np.round(
            np.std(self.abs_dist, ddof=1) / np.sqrt(self.total_elements), decimals=3
        )

    def get_rel_error_bar(self):
        return np.round(
            np.std(self.rel_dist, ddof=1) / np.sqrt(self.total_elements), decimals=3
        )

    def dump_results(self, abs_dump_path: str, rel_dump_path: str):
        np.save(abs_dump_path, self.abs_dist, allow_pickle=False)
        np.save(rel_dump_path, self.rel_dist, allow_pickle=False)

    def find_common_cliparts(
        self,
        pred_clips,
        gt_clips,
        pred_pos_x,
        gt_pos_x,
        pred_pos_y,
        gt_pos_y,
        pred_pos_o,
        gt_pos_o,
    ):
        # https://github.com/uvavision/Text2Scene/blob/master/lib/modules/abstract_evaluator.py#L309
        common_pred_x, common_pred_y, common_pred_o = [], [], []
        common_gts_x, common_gts_y, common_gts_o = [], [], []
        for i in range(len(pred_clips)):
            if pred_clips[i] in gt_clips:
                common_pred_x.append(pred_pos_x[i])
                common_pred_y.append(pred_pos_y[i])
                common_pred_o.append(pred_pos_o[i])
                index_gt = gt_clips.index(pred_clips[i])
                common_gts_x.append(gt_pos_x[index_gt])
                common_gts_y.append(gt_pos_y[index_gt])
                common_gts_o.append(gt_pos_o[index_gt])

        return (
            torch.tensor(common_pred_x).unsqueeze(0),
            torch.tensor(common_pred_y).unsqueeze(0),
            torch.tensor(common_pred_o).unsqueeze(0),
            torch.tensor(common_gts_x).unsqueeze(0),
            torch.tensor(common_gts_y).unsqueeze(0),
            torch.tensor(common_gts_o).unsqueeze(0),
        )


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

    return dist


# def u_obj_coord(x_inds, x_labs, y_inds, y_labs, attn_mask):
#     # Obtain normal dist
#     normal_sim = u_obj_coord_single(x_inds, x_labs, y_inds, y_labs, attn_mask)
#     x_labs_flipped = flip_scene(x_labs)
#     flipped_sim = u_obj_coord_single(
#         x_inds, x_labs_flipped, y_inds, y_labs, attn_mask
#     )
#     sim = torch.max(normal_sim, flipped_sim)

#     return sim


def u_obj_coord(x_inds, x_labs, y_inds, y_labs, attn_mask):
    x_inds_norm = x_inds.clone() / 500
    y_inds_norm = y_inds.clone() / 400
    x_labs_norm = x_labs.clone().float() / 500
    y_labs_norm = y_labs.clone().float() / 400
    # Obtain dist for X and Y
    sim_x = torch.pow(x_inds_norm - x_labs_norm, 2).float()
    sim_y = torch.pow(y_inds_norm - y_labs_norm, 2).float()
    sim = torch.sqrt(sim_x + sim_y + (torch.ones_like(sim_x) * 1e-15))
    # Remove the distance for the padding tokens
    sim = sim * attn_mask
    # Obtain average distance for each scene without considering the padding tokens
    sim = sim.sum(-1) / attn_mask.sum(-1)
    # https://github.com/uvavision/Text2Scene/blob/master/lib/abstract_utils.py#L366
    sim = torch.exp(-0.5 * sim / 0.2)

    return sim


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
    dist = dist.sum(-1) / (attn_mask.sum(-1) - 1)
    # Obtain average distance for each scene without considering the padding tokens
    dist = dist.sum(-1) / (attn_mask.sum(-1) - 1)

    return dist


def flip_acc(inds, labs, attn_mask, flips):
    inds = torch.abs(inds - flips.unsqueeze(-1))

    return (inds == labs).sum(-1).float() / attn_mask.sum(-1)


class ScEvaluator:
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
            abs_distance_sc(x_out, x_lab, y_out, y_lab, mask).cpu().numpy()
        )
        # Update relative distance
        self.rel_dist[self.index : self.index + batch_size] = (
            relative_distance_sc(x_out, x_lab, y_out, y_lab, mask).cpu().numpy()
        )
        self.flip_acc[self.index : self.index + batch_size] = (
            flip_acc_sc(o_out, o_lab, mask).cpu().numpy()
        )

        self.index += batch_size

    def reset_metrics(self):
        self.abs_dist = np.zeros(self.total_elements)
        self.rel_dist = np.zeros(self.total_elements)
        self.flip_acc = np.zeros(self.total_elements)
        self.index = 0

    def get_abs_dist(self):
        return np.round(
            self.abs_dist.sum() / np.count_nonzero(self.abs_dist), decimals=3
        )

    def get_rel_dist(self):
        return np.round(
            self.rel_dist.sum() / np.count_nonzero(self.rel_dist), decimals=3
        )

    def get_o_acc(self):
        return np.round(
            (self.flip_acc.sum() / np.count_nonzero(self.flip_acc)) * 100, decimals=1
        )

    def get_abs_error_bar(self):
        return np.round(
            np.std(self.abs_dist, ddof=1) / np.sqrt(np.count_nonzero(self.abs_dist)),
            decimals=3,
        )

    def get_rel_error_bar(self):
        return np.round(
            np.std(self.rel_dist, ddof=1) / np.sqrt(np.count_nonzero(self.rel_dist)),
            decimals=3,
        )

    def get_o_acc_error_bar(self):
        return np.round(
            (np.std(self.flip_acc, ddof=1) / np.sqrt(np.count_nonzero(self.flip_acc)))
            * 100,
            decimals=1,
        )


def relative_distance_sc(x_inds, x_labs, y_inds, y_labs, mask):
    dist = torch.abs(
        elementwise_distances(x_inds, y_inds) - elementwise_distances(x_labs, y_labs)
    ).float()
    dist = dist * mask.unsqueeze(-1).expand(dist.size())
    dist = dist.mean(-1)
    dist = dist.sum(-1) / (mask.sum(-1) + 1e-15)

    return dist


def abs_distance_sc(x_inds, x_labs, y_inds, y_labs, mask):
    # Obtain dist for X and Y
    dist_x = torch.pow(x_inds - x_labs, 2).float()
    dist_y = torch.pow(y_inds - y_labs, 2).float()
    dist = torch.sqrt(dist_x + dist_y + (torch.ones_like(dist_x) * 1e-15))
    # Remove the distance from the non-target elements
    dist = dist * mask
    # Obtain average distance for each scene without considering the padding tokens
    dist = dist.sum(-1) / (mask.sum(-1) + 1e-15)

    return dist


def flip_acc_sc(inds, labs, mask):
    acc = ((inds == labs).float() + 1e-15) * mask
    return acc.sum(-1) / (mask.sum(-1) + 1e-15)


class ClipartsPredictionEvaluator:
    def __init__(self, dataset_size, visual2index):
        self.dataset_size = dataset_size
        self.visual2index = visual2index
        # Create prediction arrays
        self.object_preds = np.zeros((self.dataset_size, 56))
        self.hb0_hb1_poses_preds = np.zeros((self.dataset_size * 2,))
        self.hb0_hb1_exprs_preds = np.zeros((self.dataset_size * 2,))
        # Create target arrays
        self.object_targets = np.zeros((self.dataset_size, 56))
        self.hb0_hb1_poses_targets = np.zeros((self.dataset_size * 2,))
        self.hb0_hb1_exprs_targets = np.zeros((self.dataset_size * 2,))
        self.index_objects = 0
        self.index_poses_exprs = 0

    def update_counters(
        self,
        one_hot_objects_targets,
        one_hot_objects_preds,
        hb0_hb1_poses_targets,
        hb0_hb1_poses_preds,
        hb0_hb1_exprs_targets,
        hb0_hb1_exprs_preds,
    ):
        batch_size = one_hot_objects_targets.shape[0]
        # Update target counters
        self.object_targets[
            self.index_objects : self.index_objects + batch_size
        ] = one_hot_objects_targets
        self.hb0_hb1_poses_targets[
            self.index_poses_exprs : self.index_poses_exprs + batch_size * 2
        ] = hb0_hb1_poses_targets
        self.hb0_hb1_exprs_targets[
            self.index_poses_exprs : self.index_poses_exprs + batch_size * 2
        ] = hb0_hb1_exprs_targets
        # Update prediction counters
        self.object_preds[
            self.index_objects : self.index_objects + batch_size
        ] = one_hot_objects_preds
        self.hb0_hb1_poses_preds[
            self.index_poses_exprs : self.index_poses_exprs + batch_size * 2
        ] = hb0_hb1_poses_preds
        self.hb0_hb1_exprs_preds[
            self.index_poses_exprs : self.index_poses_exprs + batch_size * 2
        ] = hb0_hb1_exprs_preds
        self.index_objects += batch_size
        self.index_poses_exprs += batch_size * 2

    def reset_counters(self):
        # Reset prediction arrays
        self.object_preds = np.zeros((self.dataset_size, 56))
        self.hb0_hb1_poses_preds = np.zeros((self.dataset_size * 2,))
        self.hb0_hb1_exprs_preds = np.zeros((self.dataset_size * 2,))
        # Reset target arrays
        self.object_targets = np.zeros((self.dataset_size, 56))
        self.hb0_hb1_poses_targets = np.zeros((self.dataset_size * 2,))
        self.hb0_hb1_exprs_targets = np.zeros((self.dataset_size * 2,))
        # Reset indexes
        self.index_objects = 0
        self.index_poses_exprs = 0

    def per_object_pr(self):
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            self.object_targets, self.object_preds, average="micro"
        )
        return (
            np.round(precision * 100, decimals=1),
            np.round(recall * 100, decimals=1),
            np.round(f1_score * 100, decimals=1)
        )

    def posses_expressions_accuracy(self):
        num_targets_poses = len(
            [target for target in self.hb0_hb1_poses_targets if target.item() != -100]
        )
        num_targets_exprs = len(
            [target for target in self.hb0_hb1_exprs_targets if target.item() != -100]
        )
        targets_poses = np.zeros((num_targets_poses,))
        predicts_poses = np.zeros((num_targets_poses,))
        targets_exprs = np.zeros((num_targets_exprs,))
        predicts_exprs = np.zeros((num_targets_exprs,))
        index_pose = 0
        index_expr = 0
        for i in range(self.hb0_hb1_poses_targets.shape[0]):
            if self.hb0_hb1_poses_targets[i].item() != -100:
                # Update pose arrays
                targets_poses[index_pose] = self.hb0_hb1_poses_targets[i].item()
                predicts_poses[index_pose] = self.hb0_hb1_poses_preds[i].item()
                # Update index
                index_pose += 1
            if self.hb0_hb1_exprs_targets[i].item() != -100:
                # Update expr arrays
                targets_exprs[index_expr] = self.hb0_hb1_exprs_targets[i].item()
                predicts_exprs[index_expr] = self.hb0_hb1_exprs_preds[i].item()
                # Update index
                index_expr += 1

        return (
            np.round(accuracy_score(targets_poses, predicts_poses) * 100, decimals=1),
            np.round(accuracy_score(targets_exprs, predicts_exprs) * 100, decimals=1),
        )
