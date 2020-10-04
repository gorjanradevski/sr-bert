import torch


def abs_distance(x_inds, x_labs, y_inds, y_labs, attn_mask):
    x_inds_norm = x_inds.clone().float()
    y_inds_norm = y_inds.clone().float()
    x_labs_norm = x_labs.clone().float()
    y_labs_norm = y_labs.clone().float()
    # Obtain dist for X and Y
    dist_x = torch.pow(x_inds_norm - x_labs_norm, 2).float()
    dist_y = torch.pow(y_inds_norm - y_labs_norm, 2).float()
    dist = torch.sqrt(dist_x + dist_y + torch.full_like(dist_x, 1e-15))
    # Set 0 similarity for the padding tokens
    dist = dist * attn_mask
    # Obtain average sim for each scene without considering the padding tokens
    dist = dist.sum(-1) / attn_mask.sum(-1)

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

    return torch.sqrt(x_dist + y_dist + torch.full_like(x_dist, 1e-15))


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
    # Remove the distance for the masked tokens
    mask_masked = torch.ones_like(x_labs)
    mask_masked[torch.where(x_labs < 0)] = 0
    dist = (
        dist
        * mask_masked.unsqueeze(1).expand(dist.size())
        * mask_masked.unsqueeze(-1).expand(dist.size())
    )
    # Obtain average distance for each scene without considering the padding tokens
    # and the main diagonal
    # Unsqueeze because it should work within a batch
    dist = dist.sum(-1) / attn_mask.sum(-1).unsqueeze(-1)
    dist = dist.sum(-1) / attn_mask.sum(-1)

    return dist
