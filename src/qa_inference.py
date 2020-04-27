import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import json
from transformers import BertConfig
from typing import Dict

from scene_layouts.generation_strategies import qa_discrete
from scene_layouts.datasets import (
    DiscreteInferenceDataset,
    collate_pad_discrete_batch,
    ContinuousInferenceDataset,
    collate_pad_continuous_batch,
    BUCKET_SIZE,
)
from scene_layouts.evaluator import QaEvaluator
from scene_layouts.modeling import SpatialDiscreteBert, SpatialContinuousBert


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


def inference(
    checkpoint_path: str,
    model_type: str,
    test_dataset_path: str,
    visual2index_path: str,
    group_name: str,
    bert_name: str,
    without_text: bool,
):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device {device}! ---")
    # Create dataset
    visual2index = json.load(open(visual2index_path))
    group_elements = get_group_elements(visual2index, group_name)
    test_dataset = (
        DiscreteInferenceDataset(
            test_dataset_path, visual2index, without_text=without_text
        )
        if model_type == "discrete"
        else ContinuousInferenceDataset(
            test_dataset_path, visual2index, without_text=without_text
        )
    )
    print(f"Testing on {len(test_dataset)}")
    # Create sampler
    test_sampler = SequentialSampler(test_dataset)
    # Create loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=collate_pad_discrete_batch
        if model_type == "discrete"
        else collate_pad_continuous_batch,
        sampler=test_sampler,
    )
    # Prepare model
    assert model_type in ["discrete", "continuous"]
    config = BertConfig.from_pretrained(bert_name)
    config.vocab_size = len(visual2index) + 1
    model = nn.DataParallel(
        SpatialDiscreteBert(config, bert_name)
        if model_type == "discrete"
        else SpatialContinuousBert(config, bert_name)
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.train(False)
    print(f"Starting inference from checkpoint {checkpoint_path}!")
    if without_text:
        print("The model won't use the text to perfrom the inference.")
    evaluator = QaEvaluator(len(test_dataset))
    with torch.no_grad():
        for (
            ids_text,
            ids_vis,
            pos_text,
            x_ind,
            y_ind,
            f_ind,
            x_lab,
            y_lab,
            f_lab,
            t_types,
            attn_mask,
        ) in tqdm(test_loader):
            # forward
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, x_lab, y_lab, f_lab, t_types, attn_mask = (
                ids_text.to(device),
                ids_vis.to(device),
                pos_text.to(device),
                x_ind.to(device),
                y_ind.to(device),
                f_ind.to(device),
                x_lab.to(device),
                y_lab.to(device),
                f_lab.to(device),
                t_types.to(device),
                attn_mask.to(device),
            )
            max_ids_text = ids_text.size()[1]
            x_out, y_out, f_out, mask = qa_discrete(
                group_elements,
                ids_text,
                ids_vis,
                pos_text,
                x_ind,
                y_ind,
                f_ind,
                t_types,
                attn_mask,
                model,
            )
            x_out, y_out = (
                x_out * BUCKET_SIZE + BUCKET_SIZE / 2,
                y_out * BUCKET_SIZE + BUCKET_SIZE / 2,
            )
            evaluator.update_metrics(
                x_out, x_lab[:, max_ids_text:], y_out, y_lab[:, max_ids_text:], mask
            )

        print(
            f"The avg ABSOLUTE dst per scene is: {evaluator.get_abs_dist()} +/- {evaluator.get_abs_error_bar()}"
        )
        print(
            f"The avg RELATIVE dst per scene is: {evaluator.get_rel_dist()} +/- {evaluator.get_rel_error_bar()}"
        )


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(
        description="Performs inference with a SpatialBERT model."
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
        "--visual2index_path",
        type=str,
        default="data/visual2index.json",
        help="Path to the visual2index mapping json.",
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
    inference(
        args.checkpoint_path,
        args.model_type,
        args.test_dataset_path,
        args.visual2index_path,
        args.group_name,
        args.bert_name,
        args.without_text,
    )


if __name__ == "__main__":
    main()
