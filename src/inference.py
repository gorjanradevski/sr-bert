import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import logging
import json
from transformers import BertConfig
from scene_layouts.utils import (
    real_distance,
    relative_distance,
    flip_acc,
    get_reference_elements,
)
from scene_layouts.generation_strategies import generation_strategy_factory

from scene_layouts.datasets import (
    DiscreteInferenceDataset,
    collate_pad_discrete_batch,
    ContinuousInferenceDataset,
    collate_pad_continuous_batch,
    BUCKET_SIZE,
)
from scene_layouts.modeling import SpatialDiscreteBert, SpatialContinuousBert


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def inference(
    checkpoint_path: str,
    model_type: str,
    test_dataset_path: str,
    visual2index_path: str,
    gen_strategy: str,
    bert_name: str,
    without_text: bool,
    ref_class: str,
):
    # https://github.com/huggingface/transformers/blob/master/examples/run_lm_finetuning.py
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.warning(f"--- Using device {device}! ---")
    # Create dataset
    visual2index = json.load(open(visual2index_path))
    index2visual = {v: k for k, v in visual2index.items()}
    ref_elements = get_reference_elements(visual2index, ref_class)
    test_dataset = (
        DiscreteInferenceDataset(
            test_dataset_path, visual2index, without_text=without_text
        )
        if model_type == "discrete"
        else ContinuousInferenceDataset(
            test_dataset_path, visual2index, without_text=without_text
        )
    )
    logger.info(f"Testing on {len(test_dataset)}")
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
    total_dist_x_relative = 0
    total_dist_y_relative = 0
    total_dist_x_real = 0
    total_dist_y_real = 0
    total_acc_f = 0
    logger.warning(f"Starting inference from checkpoint {checkpoint_path}!")
    if without_text:
        logger.warning("The model won't use the text to perfrom the inference.")
    logger.info(f"Using {gen_strategy}!")
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
                ref_elements,
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
            total_dist_x_relative += relative_distance(
                x_out, x_lab[:, max_ids_text:], attn_mask[:, max_ids_text:]
            ).item()
            total_dist_y_relative += relative_distance(
                y_out, y_lab[:, max_ids_text:], attn_mask[:, max_ids_text:]
            ).item()
            dist_x_real, flips = real_distance(
                x_out,
                x_lab[:, max_ids_text:],
                attn_mask[:, max_ids_text:],
                check_flipped=True,
            )
            total_dist_x_real += dist_x_real.item()
            total_dist_y_real += real_distance(
                y_out,
                y_lab[:, max_ids_text:],
                attn_mask[:, max_ids_text:],
                check_flipped=False,
            ).item()
            total_acc_f += flip_acc(
                f_out, f_lab[:, max_ids_text:], attn_mask[:, max_ids_text:], flips
            ).item()
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

        total_dist_x_relative /= len(test_dataset)
        total_dist_y_relative /= len(test_dataset)
        total_dist_x_real /= len(test_dataset)
        total_dist_y_real /= len(test_dataset)
        total_acc_f /= len(test_dataset)
        print(
            f"The average relative distance per scene for X is: {round(total_dist_x_relative, 2)}"
        )
        print(
            f"The average relative distance per scene for Y is: {round(total_dist_y_relative, 2)}"
        )
        print(
            f"The average real distance per scene for X is: {round(total_dist_x_real, 2)}"
        )
        print(
            f"The average real distance per scene for Y is: {round(total_dist_y_real, 2)}"
        )
        print(f"The average accuracy for the flip is: {round(total_acc_f * 100, 2)}")
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
        "--ref_class", type=str, default=None, help="The reference elements class."
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
        args.ref_class,
    )


if __name__ == "__main__":
    main()
