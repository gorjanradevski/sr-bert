import argparse
import json
import os
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig

from scene_layouts.datasets import (
    BUCKET_SIZE,
    ContinuousInferenceDataset,
    DiscreteInferenceDataset,
    collate_pad_batch,
)
from scene_layouts.evaluator import ScEvaluator
from scene_layouts.generation_strategies import sc_discrete
from scene_layouts.modeling import SpatialContinuousBert, SpatialDiscreteBert


def get_group_elements(visual2index: Dict[str, int], group_name: str):
    if group_name == "animals":
        return [v for k, v in visual2index.items() if k.startswith("a")]
    elif group_name == "food":
        return [v for k, v in visual2index.items() if k.startswith("e")]
    elif group_name == "toys":
        return [v for k, v in visual2index.items() if k.startswith("t")]
    elif group_name == "people":
        return [v for k, v in visual2index.items() if k.startswith("hb")]
    elif group_name == "sky":
        return [v for k, v in visual2index.items() if k.startswith("s")]
    elif group_name == "large":
        return [v for k, v in visual2index.items() if k.startswith("p")]
    elif group_name == "clothing":
        return [v for k, v in visual2index.items() if k.startswith("c")]
    elif group_name is None:
        return []
    else:
        raise ValueError(f"{group_name} doesn't exist!")


@torch.no_grad()
def inference(args):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device {device}! ---")
    # Create dataset
    visual2index = json.load(
        open(os.path.join(args.visuals_dicts_path, "visual2index.json"))
    )
    group_elements = get_group_elements(visual2index, args.group_name)
    test_dataset = (
        DiscreteInferenceDataset(
            args.test_dataset_path, visual2index, without_text=args.without_text
        )
        if args.model_type == "discrete"
        else ContinuousInferenceDataset(
            args.test_dataset_path, visual2index, without_text=args.without_text
        )
    )
    print(f"Testing on {len(test_dataset)}")
    # Create loader
    test_loader = DataLoader(
        test_dataset, batch_size=1, num_workers=4, collate_fn=collate_pad_batch
    )
    # Prepare model
    assert args.model_type in ["discrete", "continuous"]
    config = BertConfig.from_pretrained(args.bert_name)
    config.vocab_size = len(visual2index) + 1
    model = nn.DataParallel(
        SpatialDiscreteBert(config, args.bert_name)
        if args.model_type == "discrete"
        else SpatialContinuousBert(config, args.bert_name)
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.train(False)
    print(f"Starting inference from checkpoint {args.checkpoint_path}!")
    if args.without_text:
        print("The model won't use the text to perfrom the inference.")
    evaluator = ScEvaluator(len(test_dataset))
    for batch in tqdm(test_loader):
        # forward
        batch = {key: val.to(device) for key, val in batch.items()}
        x_out, y_out, o_out, mask = sc_discrete(
            group_elements,
            batch["ids_text"],
            batch["ids_vis"],
            batch["pos_text"],
            batch["x_ind"],
            batch["y_ind"],
            batch["o_ind"],
            batch["t_types"],
            batch["attn_mask"],
            model,
        )
        x_out, y_out = (
            x_out * BUCKET_SIZE + BUCKET_SIZE / 2,
            y_out * BUCKET_SIZE + BUCKET_SIZE / 2,
        )
        evaluator.update_metrics(
            x_out, batch["x_lab"], y_out, batch["y_lab"], o_out, batch["o_lab"], mask
        )

    print(
        f"The avg ABSOLUTE sim per scene is: {evaluator.get_abs_sim()} +/- {evaluator.get_abs_error_bar()}"
    )
    print(
        f"The avg RELATIVE sim per scene is: {evaluator.get_rel_sim()} +/- {evaluator.get_rel_error_bar()}"
    )
    print(
        f"The avg orientation acc per scene is: {evaluator.get_o_acc()} +/- {evaluator.get_o_acc_error_bar()}"
    )


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(
        description="Performs scene completeion inference with a Spatial-BERT model."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="discrete",
        help="The type of the model used to perform inference",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint to a pretrained model.",
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
        "--without_text", action="store_true", help="Whether to use the text."
    )
    parser.add_argument(
        "--group_name",
        type=str,
        default="sky",
        help="For which group is the inference performed.",
    )
    parser.add_argument(
        "--bert_name",
        type=str,
        default="bert-base-uncased",
        help="The bert model name.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    inference(args)


if __name__ == "__main__":
    main()
