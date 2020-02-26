import torch

from datasets import X_MASK, Y_MASK, F_MASK


def one_step_all_left_to_right(
    ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model, device
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


def one_step_all(
    ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model, device
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


def left_to_right(
    ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask, model, device
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
    if gen_strategy == "one_step_all_left_to_right":
        return one_step_all_left_to_right(
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
    elif gen_strategy == "one_step_all":
        return one_step_all(
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
    elif gen_strategy == "left_to_right":
        return left_to_right(
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
