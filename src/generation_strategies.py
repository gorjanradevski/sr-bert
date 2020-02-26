import torch

from datasets import X_MASK, Y_MASK, F_MASK


def one_step_all_left_to_right_continuous(
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

    return x_ind, y_ind, f_ind


def one_step_all_left_to_right_discrete(
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
    for i in range(ids_vis.size()[1]):
        x_ind[:, i] = X_MASK
        y_ind[:, i] = Y_MASK
        f_ind[:, i] = F_MASK
        x_scores, y_scores, f_scores = model(
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask
        )
        x_ind[:, i] = torch.argmax(x_scores, dim=-1)[:, max_ids_text:][:, i]
        y_ind[:, i] = torch.argmax(y_scores, dim=-1)[:, max_ids_text:][:, i]
        f_ind[:, i] = torch.argmax(f_scores, dim=-1)[:, max_ids_text:][:, i]

    return x_ind, y_ind, f_ind


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

    return x_ind, y_ind, f_ind


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

    return x_ind, y_ind, f_ind


def left_to_right_continuous(
    ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
):
    # Set all indices to MASK tokens
    x_ind[:, :] = X_MASK
    y_ind[:, :] = Y_MASK
    f_ind[:, :] = F_MASK
    max_ids_text = ids_text.size()[1]
    for _ in range(2):
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

    return x_ind, y_ind, f_ind


def left_to_right_discrete(
    ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
):
    # Set all indices to MASK tokens
    x_ind[:, :] = X_MASK
    y_ind[:, :] = Y_MASK
    f_ind[:, :] = F_MASK
    max_ids_text = ids_text.size()[1]
    for _ in range(2):
        for i in range(ids_vis.size()[1]):
            x_ind[:, i] = X_MASK
            y_ind[:, i] = Y_MASK
            f_ind[:, i] = F_MASK
            x_scores, y_scores, f_scores = model(
                ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask
            )
            x_ind[:, i] = torch.argmax(x_scores, dim=-1)[:, max_ids_text:][:, i]
            y_ind[:, i] = torch.argmax(y_scores, dim=-1)[:, max_ids_text:][:, i]
            f_ind[:, i] = torch.argmax(f_scores, dim=-1)[:, max_ids_text:][:, i]

    return x_ind, y_ind, f_ind


def highest_probability(
    ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
):
    # Set all indices to MASK tokens
    x_ind[:, :] = X_MASK
    y_ind[:, :] = Y_MASK
    f_ind[:, :] = F_MASK
    batch_indices = list(range(ids_text.size()[0]))
    max_ids_text = ids_text.size()[1]
    predicted_indices_x = []
    predicted_indices_y = []
    predicted_indices_f = []
    for _ in range(ids_vis.size()[1]):
        # Obtain model outputs
        x_scores, y_scores, f_scores = model(
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask
        )
        # If there are indices which are already chosen, change to a small number
        if len(predicted_indices_x) > 0:
            x_scores[batch_indices, predicted_indices_x] = -1e15
            y_scores[batch_indices, predicted_indices_y] = -1e15
            f_scores[batch_indices, predicted_indices_f] = -1e15

        # Obtain the probabilities and the prediction for all elements
        prob_x, pred_x = torch.max(x_scores, dim=-1)
        prob_y, pred_y = torch.max(y_scores, dim=-1)
        prob_f, pred_f = torch.max(f_scores, dim=-1)

        # Obtain the the indexes of the elements with the highest probability
        index_x = torch.argmax(prob_x[:, max_ids_text:], dim=-1)
        index_y = torch.argmax(prob_y[:, max_ids_text:], dim=-1)
        index_f = torch.argmax(prob_f[:, max_ids_text:], dim=-1)

        # Remember the chosen indices
        predicted_indices_x.append(index_x.tolist())
        predicted_indices_y.append(index_y.tolist())
        predicted_indices_f.append(index_f.tolist())

        # Change the index with the max probability with its prediction
        x_ind[batch_indices, index_x] = pred_x[batch_indices, index_x]
        y_ind[batch_indices, index_y] = pred_y[batch_indices, index_y]
        f_ind[batch_indices, index_f] = pred_f[batch_indices, index_f]

    return x_ind, y_ind, f_ind


def entropy(inputs: torch.Tensor):
    return -torch.sum(torch.exp(inputs) * torch.log2(torch.exp(inputs)), dim=-1)


def lowest_entropy(
    ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
):
    # Set all indices to MASK tokens
    x_ind[:, :] = X_MASK
    y_ind[:, :] = Y_MASK
    f_ind[:, :] = F_MASK
    batch_indices = list(range(ids_text.size()[0]))
    max_ids_text = ids_text.size()[1]
    predicted_indices_x = []
    predicted_indices_y = []
    predicted_indices_f = []
    for _ in range(ids_vis.size()[1]):
        # Obtain model outputs
        x_scores, y_scores, f_scores = model(
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask
        )
        # If there are indices which are already chosen, change their value to 0
        if len(predicted_indices_x) > 0:
            x_scores[batch_indices, predicted_indices_x] = 0
            y_scores[batch_indices, predicted_indices_y] = 0
            f_scores[batch_indices, predicted_indices_f] = 0

        # Compute entropies
        entropies_x = entropy(x_scores)
        entropies_y = entropy(y_scores)
        entropies_f = entropy(f_scores)

        # Obtain the the indexes of the elements with the highest probability
        index_x = torch.argmin(entropies_x[:, max_ids_text:], dim=-1)
        index_y = torch.argmin(entropies_y[:, max_ids_text:], dim=-1)
        index_f = torch.argmin(entropies_f[:, max_ids_text:], dim=-1)

        # Remember the chosen indices
        predicted_indices_x.append(index_x.tolist())
        predicted_indices_y.append(index_y.tolist())
        predicted_indices_f.append(index_f.tolist())

        # Change the index with the max probability with its prediction
        x_ind[batch_indices, index_x] = torch.argmax(x_scores, dim=-1)[
            batch_indices, index_x
        ]
        y_ind[batch_indices, index_y] = torch.argmax(y_scores, dim=-1)[
            batch_indices, index_y
        ]
        f_ind[batch_indices, index_f] = torch.argmax(f_scores, dim=-1)[
            batch_indices, index_f
        ]

    return x_ind, y_ind, f_ind


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
):
    if gen_strategy == "one_step_all_left_to_right_continuous":
        return one_step_all_left_to_right_continuous(
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
        )
    elif gen_strategy == "one_step_all_left_to_right_discrete":
        return one_step_all_left_to_right_discrete(
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
        )
    elif gen_strategy == "one_step_all_continuous":
        return one_step_all_continuous(
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
        )
    elif gen_strategy == "one_step_all_discrete":
        return one_step_all_discrete(
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
        )
    elif gen_strategy == "left_to_right_continuous":
        return left_to_right_continuous(
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
        )
    elif gen_strategy == "left_to_right_discrete":
        return left_to_right_discrete(
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
        )
    elif gen_strategy == "highest_probability":
        return highest_probability(
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
        )
    elif gen_strategy == "lowest_entropy":
        return lowest_entropy(
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model
        )
    else:
        raise ValueError(f"{gen_strategy} doesn't exist!")
