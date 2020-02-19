import argparse
import torch
from torch.utils.data import DataLoader, SequentialSampler, Subset
from tqdm import tqdm
import logging
import json
from inference_utils import elementwise_distances
from datasets import Text2VisualDataset, collate_pad_text2visual_batch, X_MASK, Y_MASK


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def naive_inference(
    test_dataset_path: str, visual2index_path: str, naive_type: str, metric_type: str
):
    # Create datasets
    visual2index = json.load(open(visual2index_path))
    test_dataset = Subset(
        Text2VisualDataset(
            test_dataset_path, visual2index, mask_probability=1.0, train=False
        ),
        [0, 1, 2, 3, 4, 5],
    )
    logger.info(f"Testing on {len(test_dataset)}")
    # Create samplers
    test_sampler = SequentialSampler(test_dataset)
    # Create loaders
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=collate_pad_text2visual_batch,
        sampler=test_sampler,
    )
    # Metrics
    total_dist_x = 0
    total_dist_y = 0
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

            if metric_type == "real_distance":
                total_dist_x += torch.mean(torch.abs(x_ind - x_lab).float())
                total_dist_y += torch.mean(torch.abs(y_ind - y_lab).float())
                total_acc_f += (f_ind == f_lab).sum().item() / f_ind.size()[1]
            elif metric_type == "relative_distance":
                total_dist_x += torch.mean(
                    torch.abs(
                        elementwise_distances(x_ind) - elementwise_distances(x_lab)
                    )
                    .float()
                    .sum(dim=1)
                ).item()
                total_dist_y += torch.mean(
                    torch.abs(
                        elementwise_distances(y_ind) - elementwise_distances(y_lab)
                    )
                    .float()
                    .sum(dim=1)
                ).item()
            else:
                raise ValueError(f"{metric_type} not recognized")

        print(
            f"The average distance per scene for X is: {total_dist_x/len(test_dataset)}"
        )
        print(
            f"The average distance per scene for Y is: {total_dist_y/len(test_dataset)}"
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
    parser.add_argument(
        "--metric_type", type=str, default="relative_distance", help="The metric type"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    naive_inference(
        args.test_dataset_path,
        args.visual2index_path,
        args.naive_type,
        args.metric_type,
    )


if __name__ == "__main__":
    main()
