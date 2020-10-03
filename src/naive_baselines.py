import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os
from scene_layouts.evaluator import Evaluator
from scene_layouts.datasets import (
    DiscreteInferenceDataset,
    collate_pad_batch,
    X_MASK,
    Y_MASK,
    BUCKET_SIZE,
)


def naive_inference(
    train_dataset_path: str,
    test_dataset_path: str,
    visuals_dicts_path: str,
    naive_type: str,
):
    # Create datasets
    visual2index = json.load(
        open(os.path.join(visuals_dicts_path, "visual2index.json"))
    )
    train_dataset = DiscreteInferenceDataset(train_dataset_path, visual2index)
    test_dataset = DiscreteInferenceDataset(test_dataset_path, visual2index)
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
    for (_, ids_vis, _, _, _, _, x_lab, y_lab, o_lab, _, _) in tqdm(train_loader):
        for vis_id, x, y in zip(ids_vis[0], x_lab[0], y_lab[0]):
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
    for (_, ids_vis, _, _, _, _, x_lab, y_lab, o_lab, _, _) in tqdm(test_loader):
        if naive_type == "random":
            x_ind = torch.randint_like(x_lab, low=0, high=((X_MASK - 1) * BUCKET_SIZE))
            y_ind = torch.randint_like(y_lab, low=0, high=((Y_MASK - 1) * BUCKET_SIZE))
            o_ind = torch.randint_like(o_lab, low=0, high=2)
        elif naive_type == "center":
            x_ind = torch.ones_like(x_lab) * ((X_MASK - 1) * BUCKET_SIZE // 2)
            y_ind = torch.ones_like(y_lab) * ((Y_MASK - 1) * BUCKET_SIZE // 2)
            o_ind = torch.zeros_like(o_lab)
        elif naive_type == "avg":
            x_ind = torch.tensor(
                [
                    [
                        visualindex2avgcoordinates[id_vis.item()][0]
                        for id_vis in ids_vis[0]
                    ]
                ]
            )
            y_ind = torch.tensor(
                [
                    [
                        visualindex2avgcoordinates[id_vis.item()][1]
                        for id_vis in ids_vis[0]
                    ]
                ]
            )
            o_ind = torch.randint_like(o_lab, low=0, high=2)
        else:
            raise ValueError(f"Naive inference type {naive_type} not recognized!")

        evaluator.update_metrics(
            x_ind, x_lab, y_ind, y_lab, o_ind, o_lab, torch.ones_like(x_lab)
        )

    print(
        f"The avg ABSOLUTE dist per scene is: {evaluator.get_abs_dist()} +/- {evaluator.get_abs_error_bar()}"
    )
    print(
        f"The avg RELATIVE dist per scene is: {evaluator.get_rel_dist()} +/- {evaluator.get_rel_error_bar()}"
    )
    print(f"The average ACCURACY per scene for F is: {evaluator.get_o_acc()}")


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Does a naive baseline.")
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
    naive_inference(
        args.train_dataset_path,
        args.test_dataset_path,
        args.visuals_dicts_path,
        args.naive_type,
    )


if __name__ == "__main__":
    main()
