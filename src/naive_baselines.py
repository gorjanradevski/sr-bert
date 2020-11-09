import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scene_layouts.datasets import (
    BUCKET_SIZE,
    X_MASK,
    Y_MASK,
    DiscreteInferenceDataset,
    collate_pad_batch,
)
from scene_layouts.evaluator import Evaluator


def naive_inference(args):
    # Create datasets
    visual2index = json.load(
        open(os.path.join(args.visuals_dicts_path, "visual2index.json"))
    )
    train_dataset = DiscreteInferenceDataset(args.train_dataset_path, visual2index)
    test_dataset = DiscreteInferenceDataset(args.test_dataset_path, visual2index)
    print(f"Testing on {len(test_dataset)}")
    # Create loaders
    train_loader = DataLoader(
        train_dataset, batch_size=1, num_workers=4, collate_fn=collate_pad_batch
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, num_workers=4, collate_fn=collate_pad_batch
    )
    print("Aggregating from training set")
    visualindex2avgcoordinates = {}
    visualindex2occurence = {}
    for batch in tqdm(train_loader):
        for vis_id, x, y in zip(
            batch["ids_vis"][0], batch["x_lab"][0], batch["y_lab"][0]
        ):
            if vis_id.item() not in visualindex2avgcoordinates:
                visualindex2avgcoordinates[vis_id.item()] = torch.tensor([0.0, 0.0])
            if vis_id.item() not in visualindex2occurence:
                visualindex2occurence[vis_id.item()] = 0
            visualindex2avgcoordinates[vis_id.item()] += torch.tensor(
                [x.item(), y.item()]
            )
            visualindex2occurence[vis_id.item()] += 1

    for visualindex, coordinates in visualindex2avgcoordinates.items():
        visualindex2avgcoordinates[visualindex] = (
            coordinates / visualindex2occurence[visualindex]
        )
    evaluator = Evaluator(len(test_dataset))
    for batch in tqdm(test_loader):
        if args.naive_type == "random":
            x_ind = torch.randint_like(
                batch["x_lab"], low=0, high=((X_MASK - 1) * BUCKET_SIZE)
            )
            y_ind = torch.randint_like(
                batch["y_lab"], low=0, high=((Y_MASK - 1) * BUCKET_SIZE)
            )
            o_ind = torch.randint_like(batch["o_lab"], low=0, high=2)
        elif args.naive_type == "center":
            x_ind = torch.ones_like(batch["x_lab"]) * ((X_MASK - 1) * BUCKET_SIZE // 2)
            y_ind = torch.ones_like(batch["y_lab"]) * ((Y_MASK - 1) * BUCKET_SIZE // 2)
            o_ind = torch.zeros_like(batch["o_lab"])
        elif args.naive_type == "avg":
            x_ind = torch.tensor(
                [
                    [
                        visualindex2avgcoordinates[id_vis.item()][0]
                        for id_vis in batch["ids_vis"][0]
                    ]
                ]
            )
            y_ind = torch.tensor(
                [
                    [
                        visualindex2avgcoordinates[id_vis.item()][1]
                        for id_vis in batch["ids_vis"][0]
                    ]
                ]
            )
            o_ind = torch.randint_like(batch["o_lab"], low=0, high=2)
        else:
            raise ValueError(f"Naive inference type {args.naive_type} not recognized!")

        evaluator.update_metrics(
            x_ind,
            batch["x_lab"],
            y_ind,
            batch["y_lab"],
            o_ind,
            batch["o_lab"],
            torch.ones_like(batch["x_lab"]),
        )

    print(
        f"The avg ABSOLUTE sim per scene is: {evaluator.get_abs_sim()} +/- {evaluator.get_abs_error_bar()}"
    )
    print(
        f"The avg RELATIVE sim per scene is: {evaluator.get_rel_sim()} +/- {evaluator.get_rel_error_bar()}"
    )
    print(f"The average ACCURACY per scene for F is: {evaluator.get_o_acc()}")


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(
        description="Performs inference with a naive baseline."
    )
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default="data/train_dataset.json",
        help="Path to the test dataset.",
    )
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        default="data/test_dataset.json",
        help="Path to the test dataset.",
    )
    parser.add_argument(
        "--visuals_dicts_path",
        type=str,
        default="data/visuals_dicts/",
        help="Path to the directory with the visuals dictionaries.",
    )
    parser.add_argument(
        "--naive_type",
        default="random",
        help="Type of naive inference: random, center or avg",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    naive_inference(args)


if __name__ == "__main__":
    main()
