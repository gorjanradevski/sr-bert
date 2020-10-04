import torch
from scene_layouts.datasets import SCENE_WIDTH_TEST, SCENE_HEIGHT_TEST
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


class Evaluator:
    def __init__(self, total_elements):
        self.total_elements = total_elements
        self.abs_sim = np.zeros(self.total_elements)
        self.rel_sim = np.zeros(self.total_elements)
        self.o_acc = np.zeros(self.total_elements)
        self.index = 0

    def update_metrics(self, x_out, x_lab, y_out, y_lab, o_out, o_lab, attn_mask):
        # Update absolute similarity
        batch_size = x_out.size()[0]
        abs_sim, flips = abs_similarity(x_out, x_lab, y_out, y_lab, attn_mask)
        self.abs_sim[self.index : self.index + batch_size] = abs_sim.cpu().numpy()
        # Update relative similarity
        self.rel_sim[self.index : self.index + batch_size] = (
            relative_similarity(x_out, x_lab, y_out, y_lab, attn_mask).cpu().numpy()
        )
        # Update orientation accuracy
        self.o_acc[self.index : self.index + batch_size] += (
            orientation_acc(o_out, o_lab, attn_mask, flips).cpu().numpy()
        )

        self.index += batch_size

    def reset_metrics(self):
        self.abs_sim = np.zeros(self.total_elements)
        self.rel_sim = np.zeros(self.total_elements)
        self.o_acc = np.zeros(self.total_elements)
        self.index = 0

    def get_abs_sim(self):
        return np.round(self.abs_sim.mean(), decimals=3)

    def get_rel_sim(self):
        return np.round(self.rel_sim.mean(), decimals=3)

    def get_o_acc(self):
        return np.round(self.o_acc.mean() * 100, decimals=3)

    def get_abs_error_bar(self):
        return np.round(
            np.std(self.abs_sim, ddof=1) / np.sqrt(self.total_elements), decimals=3
        )

    def get_rel_error_bar(self):
        return np.round(
            np.std(self.rel_sim, ddof=1) / np.sqrt(self.total_elements), decimals=3
        )

    def dump_results(self, abs_dump_path: str, rel_dump_path: str):
        np.save(abs_dump_path, self.abs_sim, allow_pickle=False)
        np.save(rel_dump_path, self.rel_sim, allow_pickle=False)

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


def abs_similarity(x_inds, x_labs, y_inds, y_labs, attn_mask):
    # Obtain normal similarity
    sim_normal = abs_similarity_single(x_inds, x_labs, y_inds, y_labs, attn_mask)
    # Obtain flipped similarity
    x_labs_flipped = flip_scene(x_labs)
    sim_flipped = abs_similarity_single(
        x_inds, x_labs_flipped, y_inds, y_labs, attn_mask
    )
    # Get the maximum of both
    sim = torch.max(sim_normal, sim_flipped)

    return sim, (sim == sim_flipped).float()


def abs_similarity_single(x_inds, x_labs, y_inds, y_labs, attn_mask):
    x_inds_norm = x_inds.clone().float() / SCENE_WIDTH_TEST
    y_inds_norm = y_inds.clone().float() / SCENE_HEIGHT_TEST
    x_labs_norm = x_labs.clone().float() / SCENE_WIDTH_TEST
    y_labs_norm = y_labs.clone().float() / SCENE_HEIGHT_TEST
    # Obtain dist for X and Y
    dist_x = torch.pow(x_inds_norm - x_labs_norm, 2).float()
    dist_y = torch.pow(y_inds_norm - y_labs_norm, 2).float()
    dist = torch.sqrt(dist_x + dist_y + torch.full_like(dist_x, 1e-15))
    # Convert to similarity by applying Gaussian Kernel
    # https://github.com/uvavision/Text2Scene/blob/master/lib/abstract_utils.py#L366
    sim = torch.exp(-0.5 * dist / 0.2)
    # Set 0 similarity for the padding tokens
    sim = sim * attn_mask
    # Obtain average sim for each scene without considering the padding tokens
    sim = sim.sum(-1) / attn_mask.sum(-1)

    return sim


def elementwise_distances(X: torch.Tensor, Y: torch.Tensor):
    X_inds_norm = X.clone().float() / SCENE_WIDTH_TEST
    Y_inds_norm = Y.clone().float() / SCENE_HEIGHT_TEST
    x_dist = torch.pow(
        torch.unsqueeze(X_inds_norm, 1) - torch.unsqueeze(X_inds_norm, 2), 2
    ).float()
    y_dist = torch.pow(
        torch.unsqueeze(Y_inds_norm, 1) - torch.unsqueeze(Y_inds_norm, 2), 2
    ).float()

    return torch.sqrt(x_dist + y_dist + torch.full_like(x_dist, 1e-15))


def relative_similarity(x_inds, x_labs, y_inds, y_labs, attn_mask):
    dist = torch.abs(
        elementwise_distances(x_inds, y_inds) - elementwise_distances(x_labs, y_labs)
    ).float()
    # Remove the distance for the padding tokens - Both columns and rows
    dist = (
        dist
        * attn_mask.unsqueeze(1).expand(dist.size())
        * attn_mask.unsqueeze(-1).expand(dist.size())
    )
    # Convert to similarity by applying Gaussian Kernel
    # https://github.com/uvavision/Text2Scene/blob/master/lib/abstract_utils.py#L366
    sim = torch.exp(-0.5 * dist / 0.2)
    # Set diagonal to 0
    diag = torch.diagonal(sim, dim1=1, dim2=2)
    diag.fill_(0.0)
    # Obtain average distance for each scene without considering the padding tokens
    # and the main diagonal
    # Unsqueeze because it should work within a batch
    sim = sim.sum(-1) / (attn_mask.sum(-1).unsqueeze(-1) - 1)
    sim = sim.sum(-1) / (attn_mask.sum(-1) - 1)

    return sim


def orientation_acc(inds, labs, attn_mask, flips):
    inds = torch.abs(inds - flips.unsqueeze(-1))

    return (inds == labs).sum(-1).float() / attn_mask.sum(-1)


class ScEvaluator:
    def __init__(self, total_elements):
        self.total_elements = total_elements
        self.abs_sim = np.zeros(self.total_elements)
        self.rel_sim = np.zeros(self.total_elements)
        self.orientation_acc = np.zeros(self.total_elements)
        self.index = 0

    def update_metrics(self, x_out, x_lab, y_out, y_lab, o_out, o_lab, mask):
        # Update absolute similarity
        batch_size = x_out.size()[0]
        self.abs_sim[self.index : self.index + batch_size] = (
            abs_similarity_sc(x_out, x_lab, y_out, y_lab, mask).cpu().numpy()
        )
        # Update relative similarity
        self.rel_sim[self.index : self.index + batch_size] = (
            relative_similarity_sc(x_out, x_lab, y_out, y_lab, mask).cpu().numpy()
        )
        self.orientation_acc[self.index : self.index + batch_size] = (
            orientation_acc_sc(o_out, o_lab, mask).cpu().numpy()
        )

        self.index += batch_size

    def reset_metrics(self):
        self.abs_sim = np.zeros(self.total_elements)
        self.rel_sim = np.zeros(self.total_elements)
        self.orientation_acc = np.zeros(self.total_elements)
        self.index = 0

    def get_abs_sim(self):
        return np.round(self.abs_sim.sum() / np.count_nonzero(self.abs_sim), decimals=3)

    def get_rel_sim(self):
        return np.round(self.rel_sim.sum() / np.count_nonzero(self.rel_sim), decimals=3)

    def get_o_acc(self):
        return np.round(
            (self.orientation_acc.sum() / np.count_nonzero(self.orientation_acc)) * 100,
            decimals=1,
        )

    def get_abs_error_bar(self):
        return np.round(
            np.std(self.abs_sim, ddof=1) / np.sqrt(np.count_nonzero(self.abs_sim)),
            decimals=3,
        )

    def get_rel_error_bar(self):
        return np.round(
            np.std(self.rel_sim, ddof=1) / np.sqrt(np.count_nonzero(self.rel_sim)),
            decimals=3,
        )

    def get_o_acc_error_bar(self):
        return np.round(
            (
                np.std(self.orientation_acc, ddof=1)
                / np.sqrt(np.count_nonzero(self.orientation_acc))
            )
            * 100,
            decimals=2,
        )


def relative_similarity_sc(x_inds, x_labs, y_inds, y_labs, mask):
    dist = torch.abs(
        elementwise_distances(x_inds, y_inds) - elementwise_distances(x_labs, y_labs)
    ).float()
    # Convert to similarity by applying Gaussian Kernel
    # https://github.com/uvavision/Text2Scene/blob/master/lib/abstract_utils.py#L366
    sim = torch.exp(-0.5 * dist / 0.2)
    sim = sim * mask.unsqueeze(-1).expand(sim.size())
    # Set diagonal to 0
    diag = torch.diagonal(sim, dim1=1, dim2=2)
    diag.fill_(0.0)
    # Obtain average over the selected group
    sim = sim.sum(-1) / ((sim != 0).sum(-1) + 1e-15)
    sim = sim.sum() / (mask.sum(-1) + 1e-15)

    return sim


def abs_similarity_sc(x_inds, x_labs, y_inds, y_labs, mask):
    # Obtain dist for X and Y
    x_inds_norm = x_inds.clone().float() / SCENE_WIDTH_TEST
    y_inds_norm = y_inds.clone().float() / SCENE_HEIGHT_TEST
    x_labs_norm = x_labs.clone().float() / SCENE_WIDTH_TEST
    y_labs_norm = y_labs.clone().float() / SCENE_HEIGHT_TEST
    dist_x = torch.pow(x_inds_norm - x_labs_norm, 2).float()
    dist_y = torch.pow(y_inds_norm - y_labs_norm, 2).float()
    dist = torch.sqrt(dist_x + dist_y + torch.full_like(dist_x, 1e-15))
    # Convert to similarity by applying Gaussian Kernel
    # https://github.com/uvavision/Text2Scene/blob/master/lib/abstract_utils.py#L366
    sim = torch.exp(-0.5 * dist / 0.2)
    # Set 0 similarity for the non-target elements
    sim = sim * mask
    # Obtain average distance for each scene without considering the padding tokens
    sim = sim.sum(-1) / (mask.sum(-1) + 1e-15)

    return sim


def orientation_acc_sc(inds, labs, mask):
    acc = ((inds == labs).float() + 1e-15) * mask
    return acc.sum(-1) / (mask.sum(-1) + 1e-15)


class ClipartsPredictionEvaluator:
    def __init__(
        self,
        dataset_size,
        visual2index,
        index2pose_hb0,
        index2pose_hb1,
        index2expression_hb0,
        index2expression_hb1,
    ):
        self.dataset_size = dataset_size
        self.visual2index = visual2index
        self.index2pose_hb0 = index2pose_hb0
        self.index2pose_hb1 = index2pose_hb1
        self.index2expression_hb0 = index2expression_hb0
        self.index2expression_hb1 = index2expression_hb1
        # Create target and target arrays
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

    def per_object_prf(self):
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
                target_index = str(self.targets[i, 23:58].argmax() + 23)
                # Get predictions index
                pred_index = str(self.predictions[i, 23:58].argmax() + 23)
                # Update pose arrays
                targets_pose[index] = self.index2pose_hb0[target_index]
                predicts_pose[index] = self.index2pose_hb0[pred_index]
                # Update expression arrays
                targets_expr[index] = self.index2expression_hb0[target_index]
                predicts_expr[index] = self.index2expression_hb0[pred_index]
                # Update index
                index += 1
            if self.targets[i, 58:93].sum() > 0:
                # Get target index
                target_index = str(self.targets[i, 58:93].argmax() + 58)
                # Get predictions index
                pred_index = str(self.predictions[i, 58:93].argmax() + 58)
                # Update pose arrays
                targets_pose[index] = self.index2pose_hb1[target_index]
                predicts_pose[index] = self.index2pose_hb1[pred_index]
                # Update expression arrays
                targets_expr[index] = self.index2expression_hb1[target_index]
                predicts_expr[index] = self.index2expression_hb1[pred_index]
                # Update index
                index += 1

        return (
            np.round(accuracy_score(targets_pose, predicts_pose) * 100, decimals=1),
            np.round(accuracy_score(targets_expr, predicts_expr) * 100, decimals=1),
        )
