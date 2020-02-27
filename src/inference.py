import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import logging
import json
from transformers import BertConfig
from utils import real_distance, relative_distance
from generation_strategies import generation_strategy_factory

from datasets import Text2VisualDiscreteDataset, collate_pad_discrete_text2visual_batch
from modeling import Text2VisualDiscreteBert


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def inference(
    checkpoint_path: str,
    test_dataset_path: str,
    visual2index_path: str,
    gen_strategy: str,
    batch_size: int,
    without_text: bool,
):
    # https://github.com/huggingface/transformers/blob/master/examples/run_lm_finetuning.py
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.warning(f"--- Using device {device}! ---")
    # Create datasets
    visual2index = json.load(open(visual2index_path))
    test_dataset = Text2VisualDiscreteDataset(
        test_dataset_path,
        visual2index,
        mask_probability=1.0,
        train=False,
        without_text=without_text,
    )
    logger.info(f"Testing on {len(test_dataset)}")
    # Create samplers
    test_sampler = SequentialSampler(test_dataset)
    # Create loaders
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_discrete_text2visual_batch,
        sampler=test_sampler,
    )
    # Prepare model
    config = BertConfig.from_pretrained("bert-base-uncased")
    config.vocab_size = len(visual2index) + 3
    model = nn.DataParallel(Text2VisualDiscreteBert(config, device)).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.train(False)
    # Criterion
    total_dist_x_relative = 0
    total_dist_y_relative = 0
    total_dist_x_real = 0
    total_dist_y_real = 0
    total_acc_f = 0
    logger.warning(f"Starting inference from checkpoint {checkpoint_path}!")
    if without_text:
        logger.warning("The model won't use the text to perfrom the inference.")
    # Set model in evaluation mode
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
            x_out, y_out, f_out = generation_strategy_factory(
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
            )

            total_dist_x_relative += relative_distance(
                x_out, x_lab[:, max_ids_text:], attn_mask[:, max_ids_text:]
            )
            total_dist_y_relative += relative_distance(
                y_out, y_lab[:, max_ids_text:], attn_mask[:, max_ids_text:]
            )
            total_dist_x_real += real_distance(
                x_out, x_lab[:, max_ids_text:], attn_mask[:, max_ids_text:]
            )
            total_dist_y_real += real_distance(
                y_out, y_lab[:, max_ids_text:], attn_mask[:, max_ids_text:]
            )
            total_acc_f += (
                f_out == f_lab[:, max_ids_text:]
            ).sum().item() / f_out.size()[1]

        total_dist_x_relative /= len(test_dataset)
        total_dist_y_relative /= len(test_dataset)
        total_dist_x_real /= len(test_dataset)
        total_dist_y_real /= len(test_dataset)
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
        print(
            f"The average accuracy per scene for F is: {total_acc_f/len(test_dataset)}"
        )


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(
        description="Performs inference with a Text2Position model."
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
    parser.add_argument("--batch_size", type=int, default=128, help="The batch size.")
    parser.add_argument(
        "--without_text", action="store_true", help="Whether to use the text."
    )
    parser.add_argument(
        "--gen_strategy",
        type=str,
        default="left_to_right_discrete",
        help="How to generate the positions during inference",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    inference(
        args.checkpoint_path,
        args.test_dataset_path,
        args.visual2index_path,
        args.gen_strategy,
        args.batch_size,
        args.without_text,
    )


if __name__ == "__main__":
    main()
