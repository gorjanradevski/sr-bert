import argparse
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import logging
import json
from utils import relative_distance, real_distance
from datasets import (
    Text2VisualDiscreteDataset,
    collate_pad_discrete_text2visual_batch,
    X_MASK,
    Y_MASK,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def naive_inference(test_dataset_path: str, visual2index_path: str, naive_type: str):
    # Create datasets
    visual2index = json.load(open(visual2index_path))
    test_dataset = Text2VisualDiscreteDataset(
        test_dataset_path, visual2index, mask_probability=1.0, train=False
    )
    logger.info(f"Testing on {len(test_dataset)}")
    # Create samplers
    test_sampler = SequentialSampler(test_dataset)
    # Create loaders
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=collate_pad_discrete_text2visual_batch,
        sampler=test_sampler,
    )
    # Metrics
    total_dist_x_real = 0
    total_dist_y_real = 0
    total_dist_x_relative = 0
    total_dist_y_relative = 0
    total_acc_f = 0
    # Set model in evaluation mode
    with torch.no_grad():
        for (ids_text, _, _, _, _, _, x_lab, y_lab, f_lab, _, _) in tqdm(test_loader):
            max_ids_text = ids_text.size()[1]
            x_lab = x_lab[:, max_ids_text:]
            y_lab = y_lab[:, max_ids_text:]
            f_lab = f_lab[:, max_ids_text:]
            if naive_type == "random":
                x_ind = torch.randint_like(x_lab, low=0, high=X_MASK)
                y_ind = torch.randint_like(y_lab, low=0, high=Y_MASK)
                f_ind = torch.randint_like(f_lab, low=0, high=2)
            elif naive_type == "center":
                x_ind = torch.ones_like(x_lab) * (X_MASK // 2)
                y_ind = torch.ones_like(y_lab) * (Y_MASK // 2)
                f_ind = torch.zeros_like(f_lab)
            else:
                raise ValueError(f"Naive inference type {naive_type} not recognized!")

            total_dist_x_real += real_distance(
                x_ind, x_lab, torch.ones_like(x_lab), check_flipped=True
            )
            total_dist_y_real += real_distance(
                y_ind, y_lab, torch.ones_like(y_lab), check_flipped=False
            )
            total_acc_f += (f_ind == f_lab).sum().item() / f_ind.size()[1]
            total_dist_x_relative += relative_distance(
                x_ind, x_lab, torch.ones_like(x_ind), check_flipped=True
            )
            total_dist_y_relative += relative_distance(
                y_ind, y_lab, torch.ones_like(x_ind), check_flipped=False
            )

        print(
            f"The average real distance per scene for X is: {total_dist_x_real/len(test_dataset)}"
        )
        print(
            f"The average real distance per scene for Y is: {total_dist_y_real/len(test_dataset)}"
        )
        print(
            f"The average relative distance per scene for X is: {total_dist_x_relative/len(test_dataset)}"
        )
        print(
            f"The average relative distance per scene for Y is: {total_dist_y_relative/len(test_dataset)}"
        )
        print(
            f"The average accuracy per scene for F is: {total_acc_f/len(test_dataset)}"
        )


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Does a naive baseline.")
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        default="data/test_dataset.json",
        help="Path to the test dataset.",
    )
    parser.add_argument(
        "--visual2index_path",
        type=str,
        default="data/visual2index.json",
        help="Path to the visual2index mapping json.",
    )
    parser.add_argument(
        "--naive_type",
        default="random",
        help="Type of naive inference: random or center",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    naive_inference(args.test_dataset_path, args.visual2index_path, args.naive_type)


if __name__ == "__main__":
    main()
