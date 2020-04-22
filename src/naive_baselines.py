import argparse
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import json
from scene_layouts.evaluator import Evaluator
from scene_layouts.datasets import (
    DiscreteInferenceDataset,
    collate_pad_discrete_batch,
    X_MASK,
    Y_MASK,
    BUCKET_SIZE,
)


def naive_inference(
    train_dataset_path: str,
    test_dataset_path: str,
    visual2index_path: str,
    naive_type: str,
):
    # Create datasets
    visual2index = json.load(open(visual2index_path))
    train_dataset = DiscreteInferenceDataset(train_dataset_path, visual2index)
    test_dataset = DiscreteInferenceDataset(test_dataset_path, visual2index)
    print(f"Testing on {len(test_dataset)}")
    # Create samplers
    train_sampler = SequentialSampler(train_dataset)
    test_sampler = SequentialSampler(test_dataset)
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=collate_pad_discrete_batch,
        sampler=train_sampler,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=collate_pad_discrete_batch,
        sampler=test_sampler,
    )
    print("Aggregating from training set")
    visualindex2avgcoordinates = {}
    visualindex2occurence = {}
    for (ids_text, ids_vis, _, _, _, _, x_lab, y_lab, f_lab, _, _) in tqdm(
        train_loader
    ):
        for vis_id, x, y in zip(
            ids_vis[0], x_lab[0, ids_text.size()[1] :], y_lab[0, ids_text.size()[1] :]
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
    for (ids_text, ids_vis, _, _, _, _, x_lab, y_lab, f_lab, _, _) in tqdm(test_loader):
        max_ids_text = ids_text.size()[1]
        x_lab = x_lab[:, max_ids_text:]
        y_lab = y_lab[:, max_ids_text:]
        f_lab = f_lab[:, max_ids_text:]
        if naive_type == "random":
            x_ind = torch.randint_like(x_lab, low=0, high=((X_MASK - 1) * BUCKET_SIZE))
            y_ind = torch.randint_like(y_lab, low=0, high=((Y_MASK - 1) * BUCKET_SIZE))
            f_ind = torch.randint_like(f_lab, low=0, high=2)
        elif naive_type == "center":
            x_ind = torch.ones_like(x_lab) * ((X_MASK - 1) * BUCKET_SIZE // 2)
            y_ind = torch.ones_like(y_lab) * ((Y_MASK - 1) * BUCKET_SIZE // 2)
            f_ind = torch.zeros_like(f_lab)
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
            f_ind = torch.randint_like(f_lab, low=0, high=2)
        else:
            raise ValueError(f"Naive inference type {naive_type} not recognized!")

        evaluator.update_metrics(
            x_ind, x_lab, y_ind, y_lab, f_ind, f_lab, torch.ones_like(x_lab)
        )

    print(
        f"The avg ABSOLUTE dist per scene is: {evaluator.get_abs_dist()} +/- {evaluator.get_abs_error_bar()}"
    )
    print(
        f"The avg RELATIVE dist per scene is: {evaluator.get_rel_dist()} +/- {evaluator.get_rel_error_bar()}"
    )
    print(f"The average ACCURACY per scene for F is: {evaluator.get_f_acc()}")


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Does a naive baseline.")
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default="data/train_dataset_testing.json",
        help="Path to the test dataset.",
    )
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        default="data/test_dataset_testing.json",
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
    naive_inference(
        args.train_dataset_path,
        args.test_dataset_path,
        args.visual2index_path,
        args.naive_type,
    )


if __name__ == "__main__":
    main()
