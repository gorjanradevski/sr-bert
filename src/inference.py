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
    collate_pad_discrete_batch,
    ContinuousInferenceDataset,
    collate_pad_continuous_batch,
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
):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device {device}! ---")
    # Create dataset
    visual2index = json.load(open(visual2index_path))
    index2visual = {v: k for k, v in visual2index.items()}
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
    print(f"Using {gen_strategy}!")
    pos2groupcount = {}
    pos2total = {}
    group2total = {
        "sky": 0,
        "people": 0,
        "toys": 0,
        "clothing": 0,
        "animals": 0,
        "large": 0,
        "food": 0,
    }
    evaluator = Evaluator(len(test_dataset))
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
            x_out, y_out, f_out, order = generation_strategy_factory(
                gen_strategy,
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
            x_out, y_out = (
                x_out * BUCKET_SIZE + BUCKET_SIZE / 2,
                y_out * BUCKET_SIZE + BUCKET_SIZE / 2,
            )
            evaluator.update_metrics(
                x_out,
                x_lab[:, max_ids_text:],
                y_out,
                y_lab[:, max_ids_text:],
                f_out,
                f_lab[:, max_ids_text:],
                attn_mask[:, max_ids_text:],
            )
            for i, element in enumerate(order):
                if i not in pos2groupcount:
                    pos2groupcount[i] = {
                        "sky": 0,
                        "people": 0,
                        "toys": 0,
                        "clothing": 0,
                        "animals": 0,
                        "large": 0,
                        "food": 0,
                    }
                if i not in pos2total:
                    pos2total[i] = 0
                pos2total[i] += 1
                if index2visual[element].startswith("a"):
                    pos2groupcount[i]["animals"] += 1
                    group2total["animals"] += 1
                elif index2visual[element].startswith("e"):
                    pos2groupcount[i]["food"] += 1
                    group2total["food"] += 1
                elif index2visual[element].startswith("t"):
                    pos2groupcount[i]["toys"] += 1
                    group2total["toys"] += 1
                elif index2visual[element].startswith("hb"):
                    pos2groupcount[i]["people"] += 1
                    group2total["people"] += 1
                elif index2visual[element].startswith("s"):
                    pos2groupcount[i]["sky"] += 1
                    group2total["sky"] += 1
                elif index2visual[element].startswith("p"):
                    pos2groupcount[i]["large"] += 1
                    group2total["large"] += 1
                elif index2visual[element].startswith("c"):
                    pos2groupcount[i]["clothing"] += 1
                    group2total["clothing"] += 1
                else:
                    raise ValueError("Can't be possible!")

        print(
            f"The avg ABSOLUTE dst per scene is: {evaluator.get_abs_dist()} +/- {evaluator.get_abs_error_bar()}"
        )
        print(
            f"The avg RELATIVE dst per scene is: {evaluator.get_rel_dist()} +/- {evaluator.get_rel_error_bar()}"
        )
        print(f"The avg ACCURACY for the flip is: {evaluator.get_f_acc()}")
        """
        for pos in pos2groupcount.keys():
            print(
                f"================== Percentages for group totals per position {pos} =================="
            )
            for group, count in pos2groupcount[pos].items():
                print(
                    f"{group}: {(pos2groupcount[pos][group] / group2total[group]) * 100}"
                )
            print(
                f"================== Percentages for position totals per position {pos} =================="
            )
            for group, count in pos2groupcount[pos].items():
                print(f"{group}: {(pos2groupcount[pos][group] / pos2total[pos]) * 100}")
        """


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
    )


if __name__ == "__main__":
    main()
