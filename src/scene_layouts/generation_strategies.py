import torch
import numpy as np
from torch.nn import functional as F  # type: ignore
from typing import List

from scene_layouts.datasets import X_MASK, Y_MASK, O_MASK


class Hypothesis:
    def __init__(
        self,
        prob: float,
        x_inds: torch.Tensor,
        y_inds: torch.Tensor,
        o_inds: torch.Tensor,
        predicted: List,
    ):
        self.prob = prob
        self.x_inds = x_inds
        self.y_inds = y_inds
        self.o_inds = o_inds
        self.predicted = predicted

    def __eq__(self, other):
        return self.prob == other.prob

    def __lt__(self, other):
        return self.prob > other.prob

    def __repr__(self):
        return f"{self.x_inds}, {self.predicted}, {self.prob}"

    def expand(self, x_ind, y_ind, o_ind, latest_prob, batch_indices, index):
        # Change the index with the max probability with its prediction
        x_inds_clone = self.x_inds.clone()
        y_inds_clone = self.y_inds.clone()
        o_inds_clone = self.o_inds.clone()
        x_inds_clone[batch_indices, index] = x_ind
        y_inds_clone[batch_indices, index] = y_ind
        o_inds_clone[batch_indices, index] = o_ind
        new_predicted = [pred for pred in self.predicted]
        new_predicted.append(index.tolist())

        return Hypothesis(
            self.prob * latest_prob,
            x_inds_clone,
            y_inds_clone,
            o_inds_clone,
            new_predicted,
        )


def highest_confidence_beam(ids_text, ids_vis, pos_text, t_types, attn_mask, model):
    # Set all indices to MASK tokens
    x_ind = torch.full_like(ids_vis, X_MASK)
    y_ind = torch.full_like(ids_vis, Y_MASK)
    o_ind = torch.full_like(ids_vis, O_MASK)
    beam_size = 3
    batch_indices = list(range(ids_text.size()[0]))
    pad_indices = torch.where(attn_mask[:, ids_text.size()[1] :] == 0)
    x_scores, y_scores, o_scores = model(
        ids_text, ids_vis, pos_text, x_ind, y_ind, o_ind, t_types, attn_mask
    )
    # Set pad indices to low probs
    x_scores[pad_indices] = -1e15
    y_scores[pad_indices] = -1e15
    o_scores[pad_indices] = -1e15
    cur_beam_hypothesis = []
    predicted_indices = []
    for _ in range(ids_vis.size()[1]):

        if len(predicted_indices) > 0:
            x_scores[batch_indices, predicted_indices] = -1e15
            y_scores[batch_indices, predicted_indices] = -1e15
            o_scores[batch_indices, predicted_indices] = -1e15

        # Obtain the probabilities and the prediction for all elements
        prob_x, pred_x = torch.max(x_scores, dim=-1)
        prob_y, pred_y = torch.max(y_scores, dim=-1)
        prob_f, pred_f = torch.max(o_scores, dim=-1)
        joint_prob = prob_x * prob_y * prob_f

        # Obtain the the indexes of the elements with the highest probability
        index = torch.argmax(joint_prob, dim=-1)

        # Remember the chosen indices
        predicted_indices.append(index.tolist())

        # Obtain the the indexes of the elements with the highest probability
        new_pred_x = pred_x[batch_indices, index]
        new_pred_y = pred_y[batch_indices, index]
        new_pred_f = pred_f[batch_indices, index]

        # Get clones for the indices
        x_ind_clone = x_ind.clone()
        y_ind_clone = y_ind.clone()
        o_ind_clone = o_ind.clone()

        # Change the index with the max probability with its prediction
        x_ind_clone[batch_indices, index] = new_pred_x
        y_ind_clone[batch_indices, index] = new_pred_y
        o_ind_clone[batch_indices, index] = new_pred_f

        # Obtain the predicted prob
        pred_prob = (
            torch.exp(prob_x[batch_indices, index]).item()
            * torch.exp(prob_y[batch_indices, index]).item()
            * torch.exp(prob_f[batch_indices, index]).item()
        )

        cur_beam_hypothesis.append(
            Hypothesis(
                pred_prob, x_ind_clone, y_ind_clone, o_ind_clone, [index.tolist()]
            )
        )

    cur_beam_hypothesis = sorted(cur_beam_hypothesis)[:beam_size]

    for _ in range(ids_vis.size()[1] - 1):
        new_beam_hypothesis = []
        for b in range(beam_size):
            x_scores, y_scores, o_scores = model(
                ids_text,
                ids_vis,
                pos_text,
                cur_beam_hypothesis[b].x_inds,
                cur_beam_hypothesis[b].y_inds,
                cur_beam_hypothesis[b].o_inds,
                t_types,
                attn_mask,
            )
            # Set pad indices to low probs
            x_scores[pad_indices] = -1e15
            y_scores[pad_indices] = -1e15
            o_scores[pad_indices] = -1e15
            # Get current predicted indices
            tmp_predicted_indices = [
                pred_ind for pred_ind in cur_beam_hypothesis[b].predicted
            ]
            for _ in range(ids_vis.size()[1]):
                # If there are indices which are already chosen, change to a small number
                x_scores[batch_indices, tmp_predicted_indices] = -1e15
                y_scores[batch_indices, tmp_predicted_indices] = -1e15
                o_scores[batch_indices, tmp_predicted_indices] = -1e15
                # Obtain the probabilities and the prediction for all elements
                prob_x, pred_x = torch.max(x_scores, dim=-1)
                prob_y, pred_y = torch.max(y_scores, dim=-1)
                prob_f, pred_f = torch.max(o_scores, dim=-1)
                joint_prob = prob_x * prob_y * prob_f

                # Obtain the the indexes of the elements with the highest probability
                index = torch.argmax(joint_prob, dim=-1)

                # Remember the chosen indices
                tmp_predicted_indices.append(index.tolist())

                new_pred_x = pred_x[batch_indices, index]
                new_pred_y = pred_y[batch_indices, index]
                new_pred_f = pred_f[batch_indices, index]

                # Obtain the predicted prob
                pred_prob = (
                    torch.exp(prob_x[batch_indices, index]).item()
                    * torch.exp(prob_y[batch_indices, index]).item()
                    * torch.exp(prob_f[batch_indices, index]).item()
                )
                new_beam_hypothesis.append(
                    cur_beam_hypothesis[b].expand(
                        new_pred_x,
                        new_pred_y,
                        new_pred_f,
                        pred_prob,
                        batch_indices,
                        index,
                    )
                )

        cur_beam_hypothesis = sorted(new_beam_hypothesis)[:beam_size]

    return (
        cur_beam_hypothesis[0].x_inds,
        cur_beam_hypothesis[0].y_inds,
        cur_beam_hypothesis[0].o_inds,
    )


def one_step_all(mode, ids_text, ids_vis, pos_text, t_types, attn_mask, model):
    # Set all indices to MASK tokens
    x_ind = torch.full_like(ids_vis, X_MASK)
    y_ind = torch.full_like(ids_vis, Y_MASK)
    o_ind = torch.full_like(ids_vis, O_MASK)
    x_scores, y_scores, o_scores = model(
        ids_text, ids_vis, pos_text, x_ind, y_ind, o_ind, t_types, attn_mask
    )
    if mode == "discrete":
        x_ind[:, :] = torch.argmax(x_scores, dim=-1)
        y_ind[:, :] = torch.argmax(y_scores, dim=-1)
    elif mode == "continuous":
        x_ind[:, :] = torch.ceil(x_scores)
        y_ind[:, :] = torch.ceil(y_scores)
    else:
        raise ValueError("Invalid mode!")

    o_ind[:, :] = torch.argmax(o_scores, dim=-1)

    return x_ind, y_ind, o_ind


def human_order(mode, ids_text, ids_vis, pos_text, t_types, attn_mask, model):
    # Order: Sky, Large, People, Animals, Clothing, Food, Toys
    # Set all indices to MASK tokens
    x_ind = torch.full_like(ids_vis, X_MASK)
    y_ind = torch.full_like(ids_vis, Y_MASK)
    o_ind = torch.full_like(ids_vis, O_MASK)
    for i in range(ids_vis.size()[1]):
        x_ind[:, i] = X_MASK
        y_ind[:, i] = Y_MASK
        o_ind[:, i] = O_MASK
        x_scores, y_scores, o_scores = model(
            ids_text, ids_vis, pos_text, x_ind, y_ind, o_ind, t_types, attn_mask
        )
        if mode == "continuous":
            x_ind[:, i] = torch.ceil(x_scores[:, i])
            y_ind[:, i] = torch.ceil(y_scores[:, i])
        elif mode == "discrete":
            x_ind[:, i] = torch.argmax(x_scores, dim=-1)[:, i]
            y_ind[:, i] = torch.argmax(y_scores, dim=-1)[:, i]
        else:
            raise ValueError("Invalid mode!")

        o_ind[:, i] = torch.argmax(o_scores, dim=-1)[:, i]

    return x_ind, y_ind, o_ind


def train_cond(
    mode, ids_text, ids_vis, pos_text, x_ind, y_ind, o_ind, t_types, attn_mask, model
):
    x_out = torch.ones_like(x_ind)
    y_out = torch.ones_like(y_ind)
    o_out = torch.ones_like(o_ind)
    for i in range(ids_vis.size()[1]):
        tmp_x = x_ind[:, i].clone()
        tmp_y = y_ind[:, i].clone()
        tmp_f = o_ind[:, i].clone()
        x_ind[:, i] = X_MASK
        y_ind[:, i] = Y_MASK
        o_ind[:, i] = O_MASK
        x_scores, y_scores, o_scores = model(
            ids_text, ids_vis, pos_text, x_ind, y_ind, o_ind, t_types, attn_mask
        )
        if mode == "discrete":
            x_out[:, i] = torch.argmax(x_scores, dim=-1)[:, i]
            y_out[:, i] = torch.argmax(y_scores, dim=-1)[:, i]
        elif mode == "continuous":
            x_out[:, i] = torch.ceil(x_scores[:, i])
            y_out[:, i] = torch.ceil(y_scores[:, i])
        else:
            raise ValueError("Invalid mode!")
        o_out[:, i] = torch.argmax(o_scores, dim=-1)[:, i]
        x_ind[:, i] = tmp_x.clone()
        y_ind[:, i] = tmp_y.clone()
        o_ind[:, i] = tmp_f.clone()

    return x_out, y_out, o_out


def sc_discrete(
    group_inds,
    ids_text,
    ids_vis,
    pos_text,
    x_ind,
    y_ind,
    o_ind,
    t_types,
    attn_mask,
    model,
):
    x_out = x_ind.clone()
    y_out = y_ind.clone()
    o_out = o_ind.clone()
    mask = torch.zeros_like(ids_vis)
    for i in range(ids_vis.size()[1]):
        if ids_vis[0, i].item() not in group_inds:
            continue
        mask[0, i] = 1
        tmp_x = x_ind[:, i].clone()
        tmp_y = y_ind[:, i].clone()
        tmp_f = o_ind[:, i].clone()
        x_ind[:, i] = X_MASK
        y_ind[:, i] = Y_MASK
        o_ind[:, i] = O_MASK
        x_scores, y_scores, o_scores = model(
            ids_text, ids_vis, pos_text, x_ind, y_ind, o_ind, t_types, attn_mask
        )
        x_out[:, i] = torch.argmax(x_scores, dim=-1)[:, i]
        y_out[:, i] = torch.argmax(y_scores, dim=-1)[:, i]
        o_out[:, i] = torch.argmax(o_scores, dim=-1)[:, i]
        x_ind[:, i] = tmp_x.clone()
        y_ind[:, i] = tmp_y.clone()
        o_ind[:, i] = tmp_f.clone()

    return x_out, y_out, o_out, mask


def random_order(mode, ids_text, ids_vis, pos_text, t_types, attn_mask, model):
    # Order: Sky, Large, People, Animals, Clothing, Food, Toys
    # Set all indices to MASK tokens
    x_ind = torch.full_like(ids_vis, X_MASK)
    y_ind = torch.full_like(ids_vis, Y_MASK)
    o_ind = torch.full_like(ids_vis, O_MASK)
    indices = np.random.permutation(list(range(ids_vis.size()[1])))
    for i in indices:
        x_ind[:, i] = X_MASK
        y_ind[:, i] = Y_MASK
        o_ind[:, i] = O_MASK
        x_scores, y_scores, o_scores = model(
            ids_text, ids_vis, pos_text, x_ind, y_ind, o_ind, t_types, attn_mask
        )
        if mode == "discrete":
            x_ind[:, i] = torch.argmax(x_scores, dim=-1)[:, i]
            y_ind[:, i] = torch.argmax(y_scores, dim=-1)[:, i]
        elif mode == "continuous":
            x_ind[:, i] = torch.ceil(x_scores[:, i])
            y_ind[:, i] = torch.ceil(y_scores[:, i])
        else:
            raise ValueError("Invalid mode!")
        o_ind[:, i] = torch.argmax(o_scores, dim=-1)[:, i]

    return x_ind, y_ind, o_ind


def highest_confidence(ids_text, ids_vis, pos_text, t_types, attn_mask, model):
    # Set all indices to MASK tokens
    x_ind = torch.full_like(ids_vis, X_MASK)
    y_ind = torch.full_like(ids_vis, Y_MASK)
    o_ind = torch.full_like(ids_vis, O_MASK)
    batch_indices = list(range(ids_text.size()[0]))
    pad_indices = torch.where(attn_mask[:, ids_text.size()[1] :] == 0)
    predicted_indices = []
    for i in range(ids_vis.size()[1]):
        # Obtain model outputs
        x_scores, y_scores, o_scores = model(
            ids_text, ids_vis, pos_text, x_ind, y_ind, o_ind, t_types, attn_mask
        )
        # Set pad indices to low probs
        x_scores[pad_indices] = -1e15
        y_scores[pad_indices] = -1e15
        o_scores[pad_indices] = -1e15
        # If there are indices which are already chosen, change to a small number
        if len(predicted_indices) > 0:
            x_scores[batch_indices, predicted_indices] = -1e15
            y_scores[batch_indices, predicted_indices] = -1e15
            o_scores[batch_indices, predicted_indices] = -1e15

        # Obtain the probabilities and the prediction for all elements
        prob_x, pred_x = torch.max(x_scores, dim=-1)
        prob_y, pred_y = torch.max(y_scores, dim=-1)
        prob_f, pred_f = torch.max(o_scores, dim=-1)
        joint_prob = prob_x * prob_y * prob_f

        # Obtain the the indexes of the elements with the highest probability
        index = torch.argmax(joint_prob, dim=-1)

        # Remember the chosen indices
        predicted_indices.append(index.tolist())

        # Change the index with the max probability with its prediction
        x_ind[batch_indices, index] = pred_x[batch_indices, index]
        y_ind[batch_indices, index] = pred_y[batch_indices, index]
        o_ind[batch_indices, index] = pred_f[batch_indices, index]

    return x_ind, y_ind, o_ind


def entropy(inputs: torch.Tensor):
    return torch.sum(-torch.exp(inputs) * torch.log2(torch.exp(inputs)), dim=-1)


def lowest_entropy(ids_text, ids_vis, pos_text, t_types, attn_mask, model, device):
    # Set all indices to MASK tokens
    x_ind = torch.full_like(ids_vis, X_MASK)
    y_ind = torch.full_like(ids_vis, Y_MASK)
    o_ind = torch.full_like(ids_vis, O_MASK)
    batch_indices = list(range(ids_text.size()[0]))
    pad_indices = torch.where(attn_mask[:, ids_text.size()[1] :] == 0)
    predicted_indices = []
    for i in range(ids_vis.size()[1]):
        # Obtain model outputs
        x_scores, y_scores, o_scores = model(
            ids_text, ids_vis, pos_text, x_ind, y_ind, o_ind, t_types, attn_mask
        )
        # Set pad indices to high entropy values
        x_scores[pad_indices] = F.log_softmax(
            torch.ones(1, x_scores.size()[-1]).to(device), dim=-1
        )
        y_scores[pad_indices] = F.log_softmax(
            torch.ones(1, y_scores.size()[-1]).to(device), dim=-1
        )
        o_scores[pad_indices] = F.log_softmax(
            torch.ones(1, o_scores.size()[-1]).to(device), dim=-1
        )
        # Set predicted indices to high entropy values
        if len(predicted_indices) > 0:
            x_scores[batch_indices, predicted_indices] = F.log_softmax(
                torch.ones(1, x_scores.size()[-1]).to(device), dim=-1
            )
            y_scores[batch_indices, predicted_indices] = F.log_softmax(
                torch.ones(1, y_scores.size()[-1]).to(device), dim=-1
            )
            o_scores[batch_indices, predicted_indices] = F.log_softmax(
                torch.ones(1, o_scores.size()[-1]).to(device), dim=-1
            )

        # Compute entropies
        entropies_x = entropy(x_scores)
        entropies_y = entropy(y_scores)
        entropies_f = entropy(o_scores)
        joint_entropy = entropies_x * entropies_y * entropies_f

        # Obtain the the indexes of the elements with the highest probability
        index = torch.argmin(joint_entropy, dim=-1)

        # Remember the chosen indices
        predicted_indices.append(index.tolist())

        # Change the index with the max probability with its prediction
        x_ind[batch_indices, index] = torch.argmax(x_scores, dim=-1)[
            batch_indices, index
        ]
        y_ind[batch_indices, index] = torch.argmax(y_scores, dim=-1)[
            batch_indices, index
        ]
        o_ind[batch_indices, index] = torch.argmax(o_scores, dim=-1)[
            batch_indices, index
        ]

    return x_ind, y_ind, o_ind


def generation_strategy_factory(
    gen_strategy: str,
    mode: str,
    ids_text,
    ids_vis,
    pos_text,
    t_types,
    attn_mask,
    model,
    device,
):
    if gen_strategy == "one_step_all":
        return one_step_all(
            mode, ids_text, ids_vis, pos_text, t_types, attn_mask, model
        )
    elif gen_strategy == "human_order":
        return human_order(mode, ids_text, ids_vis, pos_text, t_types, attn_mask, model)
    elif gen_strategy == "highest_confidence_beam":
        return highest_confidence_beam(
            ids_text, ids_vis, pos_text, t_types, attn_mask, model
        )
    elif gen_strategy == "highest_confidence":
        return highest_confidence(
            ids_text, ids_vis, pos_text, t_types, attn_mask, model
        )
    elif gen_strategy == "lowest_entropy":
        return lowest_entropy(
            ids_text, ids_vis, pos_text, t_types, attn_mask, model, device
        )
    elif gen_strategy == "random_order":
        return random_order(
            mode, ids_text, ids_vis, pos_text, t_types, attn_mask, model
        )
    elif gen_strategy == "train_cond":
        return train_cond(mode, ids_text, ids_vis, pos_text, t_types, attn_mask, model)
    else:
        raise ValueError(f"{gen_strategy} doesn't exist!")
