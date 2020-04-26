import torch
import numpy as np
from torch.nn import functional as F

from scene_layouts.datasets import X_MASK, Y_MASK, F_MASK


class Hypothesis:
    def __init__(
        self,
        prob: float,
        x_inds: torch.Tensor,
        y_inds: torch.Tensor,
        f_inds: torch.Tensor,
    ):
        self.prob = prob
        self.x_inds = x_inds
        self.y_inds = y_inds
        self.f_inds = f_inds

    def __eq__(self, other):
        return self.prob == other.prob

    def __lt__(self, other):
        return self.prob > other.prob

    def expand(self, x_ind, y_ind, f_ind, latest_prob, index):
        x_inds_clone = self.x_inds.clone()
        y_inds_clone = self.y_inds.clone()
        f_inds_clone = self.f_inds.clone()
        x_inds_clone[0, index] = x_ind
        y_inds_clone[0, index] = y_ind
        f_inds_clone[0, index] = f_ind

        return Hypothesis(
            self.prob * latest_prob, x_inds_clone, y_inds_clone, f_inds_clone
        )


def one_step_all_continuous(
    ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
):
    # Set all indices to MASK tokens
    x_ind[:, :] = X_MASK
    y_ind[:, :] = Y_MASK
    f_ind[:, :] = F_MASK
    max_ids_text = ids_text.size()[1]
    x_scores, y_scores, f_scores = model(
        ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask
    )
    x_ind[:, :] = torch.ceil(x_scores[:, max_ids_text:])
    y_ind[:, :] = torch.ceil(y_scores[:, max_ids_text:])
    f_ind[:, :] = torch.argmax(f_scores, dim=-1)[:, max_ids_text:]

    return x_ind, y_ind, f_ind, []


def one_step_all_discrete(
    ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
):
    # Set all indices to MASK tokens
    x_ind[:, :] = X_MASK
    y_ind[:, :] = Y_MASK
    f_ind[:, :] = F_MASK
    max_ids_text = ids_text.size()[1]
    x_scores, y_scores, f_scores = model(
        ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask
    )
    x_ind[:, :] = torch.argmax(x_scores, dim=-1)[:, max_ids_text:]
    y_ind[:, :] = torch.argmax(y_scores, dim=-1)[:, max_ids_text:]
    f_ind[:, :] = torch.argmax(f_scores, dim=-1)[:, max_ids_text:]

    return x_ind, y_ind, f_ind, []


def human_order_continuous(
    ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
):
    # Order: Sky, Large, People, Animals, Clothing, Food, Toys
    # Set all indices to MASK tokens
    x_ind[:, :] = X_MASK
    y_ind[:, :] = Y_MASK
    f_ind[:, :] = F_MASK
    max_ids_text = ids_text.size()[1]
    for i in range(ids_vis.size()[1]):
        x_ind[:, i] = X_MASK
        y_ind[:, i] = Y_MASK
        f_ind[:, i] = F_MASK
        x_scores, y_scores, f_scores = model(
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask
        )
        x_ind[:, i] = torch.ceil(x_scores[:, max_ids_text:][:, i])
        y_ind[:, i] = torch.ceil(y_scores[:, max_ids_text:][:, i])
        f_ind[:, i] = torch.argmax(f_scores, dim=-1)[:, max_ids_text:][:, i]

    return x_ind, y_ind, f_ind, []


def train_cond_continuous(
    ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
):
    x_out = torch.ones_like(x_ind)
    y_out = torch.ones_like(y_ind)
    f_out = torch.ones_like(f_ind)
    max_ids_text = ids_text.size()[1]
    for i in range(ids_vis.size()[1]):
        tmp_x = x_ind[:, i].clone()
        tmp_y = y_ind[:, i].clone()
        tmp_f = f_ind[:, i].clone()
        x_ind[:, i] = X_MASK
        y_ind[:, i] = Y_MASK
        f_ind[:, i] = F_MASK
        x_scores, y_scores, f_scores = model(
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask
        )
        x_out[:, i] = torch.ceil(x_scores[:, max_ids_text:][:, i])
        y_out[:, i] = torch.ceil(y_scores[:, max_ids_text:][:, i])
        f_out[:, i] = torch.argmax(f_scores, dim=-1)[:, max_ids_text:][:, i]
        x_ind[:, i] = tmp_x.clone()
        y_ind[:, i] = tmp_y.clone()
        f_ind[:, i] = tmp_f.clone()

    return x_out, y_out, f_out


def train_cond_discrete(
    ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
):
    x_out = torch.ones_like(x_ind)
    y_out = torch.ones_like(y_ind)
    f_out = torch.ones_like(f_ind)
    max_ids_text = ids_text.size()[1]
    for i in range(ids_vis.size()[1]):
        tmp_x = x_ind[:, i].clone()
        tmp_y = y_ind[:, i].clone()
        tmp_f = f_ind[:, i].clone()
        x_ind[:, i] = X_MASK
        y_ind[:, i] = Y_MASK
        f_ind[:, i] = F_MASK
        x_scores, y_scores, f_scores = model(
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask
        )
        x_out[:, i] = torch.argmax(x_scores, dim=-1)[:, max_ids_text:][:, i]
        y_out[:, i] = torch.argmax(y_scores, dim=-1)[:, max_ids_text:][:, i]
        f_out[:, i] = torch.argmax(f_scores, dim=-1)[:, max_ids_text:][:, i]
        x_ind[:, i] = tmp_x.clone()
        y_ind[:, i] = tmp_y.clone()
        f_ind[:, i] = tmp_f.clone()

    return x_out, y_out, f_out


def human_order_discrete(
    ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
):
    # Order: Sky, Large, People, Animals, Clothing, Food, Toys
    # Set all indices to MASK tokens
    x_ind[:, :] = X_MASK
    y_ind[:, :] = Y_MASK
    f_ind[:, :] = F_MASK
    max_ids_text = ids_text.size()[1]
    for i in range(ids_vis.size()[1]):
        x_ind[:, i] = X_MASK
        y_ind[:, i] = Y_MASK
        f_ind[:, i] = F_MASK
        x_scores, y_scores, f_scores = model(
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask
        )
        x_ind[:, i] = torch.argmax(x_scores, dim=-1)[:, max_ids_text + i]
        y_ind[:, i] = torch.argmax(y_scores, dim=-1)[:, max_ids_text + i]
        f_ind[:, i] = torch.argmax(f_scores, dim=-1)[:, max_ids_text + i]

    return x_ind, y_ind, f_ind, []


def human_order_discrete_beam_search(
    ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
):
    # --- Note: Assume batch size 1 ---
    # https://github.com/tensorflow/models/blob/master/research/textsum/beam_search.py
    # Order: Sky, Large, People, Animals, Clothing, Food, Toys
    # Set all indices to MASK tokens
    beam_size = 5
    x_ind[:, :] = X_MASK
    y_ind[:, :] = Y_MASK
    f_ind[:, :] = F_MASK
    max_ids_text = ids_text.size()[1]
    x_scores, y_scores, f_scores = model(
        ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask
    )
    cur_beam_hypothesis = []
    for x in range(x_scores.size()[-1]):
        for y in range(y_scores.size()[-1]):
            for f in range(f_scores.size()[-1]):
                x_ind[0, 0] = x
                y_ind[0, 0] = y
                f_ind[0, 0] = f
                joint_prob = (
                    torch.exp(x_scores[:, max_ids_text:][0, 0, x]).item()
                    * torch.exp(y_scores[:, max_ids_text:][0, 0, y]).item()
                    * torch.exp(f_scores[:, max_ids_text:][0, 0, f]).item()
                )
                cur_beam_hypothesis.append(
                    Hypothesis(joint_prob, x_ind.clone(), y_ind.clone(), f_ind.clone())
                )
    cur_beam_hypothesis = sorted(cur_beam_hypothesis)[:beam_size]

    for index in range(1, ids_vis.size()[1]):
        x_ind[:, index] = X_MASK
        y_ind[:, index] = Y_MASK
        f_ind[:, index] = F_MASK
        new_beam_hypothesis = []
        for i in range(beam_size):
            x_scores, y_scores, f_scores = model(
                ids_text,
                ids_vis,
                pos_text,
                cur_beam_hypothesis[i].x_inds,
                cur_beam_hypothesis[i].y_inds,
                cur_beam_hypothesis[i].f_inds,
                t_types,
                attn_mask,
            )
            for x in range(x_scores.size()[-1]):
                for y in range(y_scores.size()[-1]):
                    for f in range(f_scores.size()[-1]):
                        joint_prob = (
                            torch.exp(x_scores[:, max_ids_text:][0, index, x]).item()
                            * torch.exp(y_scores[:, max_ids_text:][0, index, y]).item()
                            * torch.exp(f_scores[:, max_ids_text:][0, index, f]).item()
                        )
                        new_beam_hypothesis.append(
                            cur_beam_hypothesis[i].expand(x, y, f, joint_prob, index)
                        )
        cur_beam_hypothesis = sorted(new_beam_hypothesis)[:beam_size]

    return (
        cur_beam_hypothesis[0].x_inds,
        cur_beam_hypothesis[0].y_inds,
        cur_beam_hypothesis[0].f_inds,
        [],
    )


def random_discrete(
    ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
):
    # Order: Sky, Large, People, Animals, Clothing, Food, Toys
    # Set all indices to MASK tokens
    x_ind[:, :] = X_MASK
    y_ind[:, :] = Y_MASK
    f_ind[:, :] = F_MASK
    max_ids_text = ids_text.size()[1]
    indices = np.random.permutation(list(range(ids_vis.size()[1])))
    for i in indices:
        x_ind[:, i] = X_MASK
        y_ind[:, i] = Y_MASK
        f_ind[:, i] = F_MASK
        x_scores, y_scores, f_scores = model(
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask
        )
        x_ind[:, i] = torch.argmax(x_scores, dim=-1)[:, max_ids_text + i]
        y_ind[:, i] = torch.argmax(y_scores, dim=-1)[:, max_ids_text + i]
        f_ind[:, i] = torch.argmax(f_scores, dim=-1)[:, max_ids_text + i]

    return x_ind, y_ind, f_ind, []


def random_continuous(
    ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
):
    # Set all indices to MASK tokens
    x_ind[:, :] = X_MASK
    y_ind[:, :] = Y_MASK
    f_ind[:, :] = F_MASK
    max_ids_text = ids_text.size()[1]
    indices = np.random.permutation(list(range(ids_vis.size()[1])))
    for i in indices:
        x_ind[:, i] = X_MASK
        y_ind[:, i] = Y_MASK
        f_ind[:, i] = F_MASK
        x_scores, y_scores, f_scores = model(
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask
        )
        x_ind[:, i] = torch.ceil(x_scores[:, max_ids_text + i])
        y_ind[:, i] = torch.ceil(y_scores[:, max_ids_text + i])
        f_ind[:, i] = torch.argmax(f_scores, dim=-1)[:, max_ids_text + i]

    return x_ind, y_ind, f_ind, []


def highest_confidence(
    ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
):
    # Set all indices to MASK tokens
    x_ind[:, :] = X_MASK
    y_ind[:, :] = Y_MASK
    f_ind[:, :] = F_MASK
    batch_indices = list(range(ids_text.size()[0]))
    max_ids_text = ids_text.size()[1]
    pad_indices = torch.where(attn_mask[:, max_ids_text:] == 0)
    predicted_indices = []
    for i in range(ids_vis.size()[1]):
        # Obtain model outputs
        x_scores, y_scores, f_scores = model(
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask
        )
        # Set pad indices to low probs
        x_scores[:, max_ids_text:][pad_indices] = -1e15
        y_scores[:, max_ids_text:][pad_indices] = -1e15
        f_scores[:, max_ids_text:][pad_indices] = -1e15
        # If there are indices which are already chosen, change to a small number
        if len(predicted_indices) > 0:
            x_scores[:, max_ids_text:][batch_indices, predicted_indices] = -1e15
            y_scores[:, max_ids_text:][batch_indices, predicted_indices] = -1e15
            f_scores[:, max_ids_text:][batch_indices, predicted_indices] = -1e15

        # Obtain the probabilities and the prediction for all elements
        prob_x, pred_x = torch.max(x_scores, dim=-1)
        prob_y, pred_y = torch.max(y_scores, dim=-1)
        prob_f, pred_f = torch.max(f_scores, dim=-1)
        joint_prob = prob_x * prob_y * prob_f

        # Obtain the the indexes of the elements with the highest probability
        index = torch.argmax(joint_prob[:, max_ids_text:], dim=-1)

        # Remember the chosen indices
        predicted_indices.append(index.tolist())

        # Change the index with the max probability with its prediction
        x_ind[batch_indices, index] = pred_x[:, max_ids_text:][batch_indices, index]
        y_ind[batch_indices, index] = pred_y[:, max_ids_text:][batch_indices, index]
        f_ind[batch_indices, index] = pred_f[:, max_ids_text:][batch_indices, index]

    order = [ids_vis[0, index[0]].item() for index in predicted_indices]

    return x_ind, y_ind, f_ind, order


def entropy(inputs: torch.Tensor):
    return torch.sum(-torch.exp(inputs) * torch.log2(torch.exp(inputs)), dim=-1)


def lowest_entropy(
    ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model, device
):
    # Set all indices to MASK tokens
    x_ind[:, :] = X_MASK
    y_ind[:, :] = Y_MASK
    f_ind[:, :] = F_MASK
    batch_indices = list(range(ids_text.size()[0]))
    max_ids_text = ids_text.size()[1]
    pad_indices = torch.where(attn_mask[:, max_ids_text:] == 0)
    predicted_indices = []
    for i in range(ids_vis.size()[1]):
        # Obtain model outputs
        x_scores, y_scores, f_scores = model(
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask
        )
        # Set pad indices to high entropy values
        x_scores[:, max_ids_text:][pad_indices] = F.log_softmax(
            torch.ones(1, x_scores.size()[-1]).to(device), dim=-1
        )
        y_scores[:, max_ids_text:][pad_indices] = F.log_softmax(
            torch.ones(1, y_scores.size()[-1]).to(device), dim=-1
        )
        f_scores[:, max_ids_text:][pad_indices] = F.log_softmax(
            torch.ones(1, f_scores.size()[-1]).to(device), dim=-1
        )
        # Set predicted indices to high entropy values
        if len(predicted_indices) > 0:
            x_scores[:, max_ids_text:][
                batch_indices, predicted_indices
            ] = F.log_softmax(torch.ones(1, x_scores.size()[-1]).to(device), dim=-1)
            y_scores[:, max_ids_text:][
                batch_indices, predicted_indices
            ] = F.log_softmax(torch.ones(1, y_scores.size()[-1]).to(device), dim=-1)
            f_scores[:, max_ids_text:][
                batch_indices, predicted_indices
            ] = F.log_softmax(torch.ones(1, f_scores.size()[-1]).to(device), dim=-1)

        # Compute entropies
        entropies_x = entropy(x_scores)
        entropies_y = entropy(y_scores)
        entropies_f = entropy(f_scores)
        joint_entropy = entropies_x * entropies_y * entropies_f

        # Obtain the the indexes of the elements with the highest probability
        index = torch.argmin(joint_entropy[:, max_ids_text:], dim=-1)

        # Remember the chosen indices
        predicted_indices.append(index.tolist())

        # Change the index with the max probability with its prediction
        x_ind[batch_indices, index] = torch.argmax(x_scores[:, max_ids_text:], dim=-1)[
            batch_indices, index
        ]
        y_ind[batch_indices, index] = torch.argmax(y_scores[:, max_ids_text:], dim=-1)[
            batch_indices, index
        ]
        f_ind[batch_indices, index] = torch.argmax(f_scores[:, max_ids_text:], dim=-1)[
            batch_indices, index
        ]

    order = [ids_vis[0, index[0]].item() for index in predicted_indices]

    return x_ind, y_ind, f_ind, order


def generation_strategy_factory(
    gen_strategy: str,
    ids_text,
    ids_vis,
    pos_text,
    x_ind,
    y_ind,
    f_ind,
    t_types,
    attn_mask,
    model,
    device,
):
    if gen_strategy == "one_step_all_continuous":
        return one_step_all_continuous(
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
        )
    elif gen_strategy == "one_step_all_discrete":
        return one_step_all_discrete(
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
        )
    elif gen_strategy == "human_order_continuous":
        return human_order_continuous(
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
        )
    elif gen_strategy == "human_order_discrete":
        return human_order_discrete(
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
        )
    elif gen_strategy == "human_order_discrete_beam_search":
        return human_order_discrete_beam_search(
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
        )
    elif gen_strategy == "highest_confidence":
        return highest_confidence(
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
        )
    elif gen_strategy == "lowest_entropy":
        return lowest_entropy(
            ids_text,
            ids_vis,
            pos_text,
            x_ind,
            y_ind,
            f_ind,
            t_types,
            attn_mask,
            model,
            device,
        )
    elif gen_strategy == "random_discrete":
        return random_discrete(
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
        )
    elif gen_strategy == "random_continuous":
        return random_continuous(
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
        )
    elif gen_strategy == "train_cond_discrete":
        return train_cond_discrete(
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
        )
    elif gen_strategy == "train_cond_continuous":
        return train_cond_continuous(
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
        )
    else:
        raise ValueError(f"{gen_strategy} doesn't exist!")
