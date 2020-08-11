import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import json
from transformers import BertConfig
from scene_layouts.generation_strategies import generation_strategy_factory

from scene_layouts.datasets import (
    DiscreteInferenceDataset,
    ContinuousInferenceDataset,
    collate_pad_batch,
    BUCKET_SIZE,
)
from scene_layouts.evaluator import Evaluator
from scene_layouts.modeling import SpatialDiscreteBert, SpatialContinuousBert


def inference(
    checkpoint_path: str,
    model_type: str,
    test_dataset_path: str,
    visual2index_path: str,
    gen_strategy: str,
    bert_name: str,
    without_text: bool,
    abs_dump_path: str,
    rel_dump_path: str,
):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device {device}! ---")
    # Create dataset
    visual2index = json.load(open(visual2index_path))
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
        collate_fn=collate_pad_batch,
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
    print(f"Using {gen_strategy}!")
    evaluator = Evaluator(len(test_dataset))
    with torch.no_grad():
        for (
            ids_text,
            ids_vis,
            pos_text,
            x_ind,
            y_ind,
            o_ind,
            x_lab,
            y_lab,
            o_lab,
            t_types,
            attn_mask,
        ) in tqdm(test_loader):
            # forward
            ids_text, ids_vis, pos_text, x_ind, y_ind, o_ind, x_lab, y_lab, o_lab, t_types, attn_mask = (
                ids_text.to(device),
                ids_vis.to(device),
                pos_text.to(device),
                x_ind.to(device),
                y_ind.to(device),
                o_ind.to(device),
                x_lab.to(device),
                y_lab.to(device),
                o_lab.to(device),
                t_types.to(device),
                attn_mask.to(device),
            )
            max_ids_text = ids_text.size()[1]
            x_out, y_out, o_out = generation_strategy_factory(
                gen_strategy,
                model_type,
                ids_text,
                ids_vis,
                pos_text,
                x_ind,
                y_ind,
                o_ind,
                t_types,
                attn_mask,
                model,
                device,
            )
            x_out, y_out = (
                x_out * BUCKET_SIZE + BUCKET_SIZE / 2,
                y_out * BUCKET_SIZE + BUCKET_SIZE / 2,
            )
            evaluator.update_metrics(
                x_out,
                x_lab[:, max_ids_text:].float(),
                y_out,
                y_lab[:, max_ids_text:].float(),
                o_out,
                o_lab[:, max_ids_text:],
                attn_mask[:, max_ids_text:],
            )

        print(
            f"The avg ABSOLUTE dst per scene is: {evaluator.get_abs_dist()} +/- {evaluator.get_abs_error_bar()}"
        )
        print(
            f"The avg RELATIVE dst per scene is: {evaluator.get_rel_dist()} +/- {evaluator.get_rel_error_bar()}"
        )
        print(f"The avg ACCURACY for the flip is: {evaluator.get_o_acc()}")
        if abs_dump_path is not None and rel_dump_path is not None:
            evaluator.dump_results(abs_dump_path, rel_dump_path)


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
        "--gen_strategy",
        type=str,
        default="left_to_right_discrete",
        help="How to generate the positions during inference",
    )
    parser.add_argument(
        "--bert_name",
        type=str,
        default="bert-base-uncased",
        help="The bert model name.",
    )
    parser.add_argument(
        "--abs_dump_path",
        type=str,
        default=None,
        help="Location of the absolute distance results.",
    )
    parser.add_argument(
        "--rel_dump_path",
        type=str,
        default=None,
        help="Location of the relative distance results.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    inference(
        args.checkpoint_path,
        args.model_type,
        args.test_dataset_path,
        args.visual2index_path,
        args.gen_strategy,
        args.bert_name,
        args.without_text,
        args.abs_dump_path,
        args.rel_dump_path,
    )


if __name__ == "__main__":
    main()
