import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import logging
import json
from transformers import BertConfig
from scene_layouts.generation_strategies import generation_strategy_factory
from PIL import Image
import os
from typing import List

from scene_layouts.datasets import (
    Text2VisualDiscreteTestDataset,
    collate_pad_discrete_text2visual_batch,
    Text2VisualContinuousTestDataset,
    collate_pad_continuous_text2visual_batch,
    BUCKET_SIZE,
)
from scene_layouts.modeling import Text2VisualDiscreteBert, Text2VisualContinuousBert


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dump_scene(
    pngs_path: str,
    visual_names: List[str],
    x_indexes: torch.Tensor,
    y_indexes: torch.Tensor,
    f_indexes: torch.Tensor,
    dump_image_path: str,
    bucketized: bool = False,
):
    background = Image.open(os.path.join(pngs_path, "background.png"))
    for visual_name, x_index, y_index, f_index in zip(
        visual_names, x_indexes, y_indexes, f_indexes
    ):
        image = Image.open(os.path.join(pngs_path, visual_name))
        if f_index == 1:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        # Pasting the image
        background.paste(
            image,
            (
                x_index * BUCKET_SIZE + BUCKET_SIZE // 2 - image.size[0] // 2
                if bucketized
                else x_index - image.size[0] // 2,
                y_index * BUCKET_SIZE + BUCKET_SIZE // 2 - image.size[1] // 2
                if bucketized
                else y_index - image.size[1] // 2,
            ),
            image,
        )

    background.save(dump_image_path)


def generation(
    checkpoint_path: str,
    model_type: str,
    test_dataset_path: str,
    visual2index_path: str,
    gen_strategy: str,
    batch_size: int,
    without_text: bool,
    num_iter: int,
    dump_scenes_path: str,
    pngs_path: str,
):
    # https://github.com/huggingface/transformers/blob/master/examples/run_lm_finetuning.py
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.warning(f"--- Using device {device}! ---")
    # Create dataset
    assert model_type in ["discrete", "continuous"]
    visual2index = json.load(open(visual2index_path))
    index2visual = {v: k for k, v in visual2index.items()}
    test_dataset = (
        Text2VisualDiscreteTestDataset(
            test_dataset_path, visual2index, without_text=without_text
        )
        if model_type == "discrete"
        else Text2VisualContinuousTestDataset(
            test_dataset_path, visual2index, without_text=without_text
        )
    )
    logger.info(f"Testing on {len(test_dataset)}")
    # Create sampler
    test_sampler = SequentialSampler(test_dataset)
    # Create loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_discrete_text2visual_batch
        if model_type == "discrete"
        else collate_pad_continuous_text2visual_batch,
        sampler=test_sampler,
    )
    # Prepare model
    config = BertConfig.from_pretrained("bert-base-uncased")
    config.vocab_size = len(visual2index) + 3
    model = nn.DataParallel(
        Text2VisualDiscreteBert(config, device)
        if model_type == "discrete"
        else Text2VisualContinuousBert(config, device)
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.train(False)
    logger.warning(f"Starting generation from checkpoint {checkpoint_path}!")
    if without_text:
        logger.warning("The model won't use the text to generate the scenes.")
    logger.info(f"Using {gen_strategy}! If applies, {num_iter} is the number of iters.")
    index_org = 1
    index_gen = 1
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
            ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask = (
                ids_text.to(device),
                ids_vis.to(device),
                pos_text.to(device),
                x_ind.to(device),
                y_ind.to(device),
                f_ind.to(device),
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
                device,
                num_iter,
            )

            x_out, y_out, f_out = x_out.cpu(), y_out.cpu(), f_out.cpu()
            # Dump original
            for i in range(batch_size):
                dump_image_path = os.path.join(
                    dump_scenes_path, str(index_org) + "_original.png"
                )
                visual_names = [
                    index2visual[index.item()]
                    for index in ids_vis[i]
                    if index.item() in index2visual
                ]
                dump_scene(
                    pngs_path,
                    visual_names,
                    x_lab[:, max_ids_text:][i],
                    y_lab[:, max_ids_text:][i],
                    f_lab[:, max_ids_text:][i],
                    dump_image_path,
                    bucketized=False,
                )
                index_org += 1
            # Dump model generated
            for i in range(batch_size):
                dump_image_path = os.path.join(
                    dump_scenes_path, str(index_gen) + "_generated.png"
                )
                visual_names = [
                    index2visual[index.item()]
                    for index in ids_vis[i]
                    if index.item() in index2visual
                ]
                dump_scene(
                    pngs_path,
                    visual_names,
                    x_out[i],
                    y_out[i],
                    f_out[i],
                    dump_image_path,
                    bucketized=True,
                )
                index_gen += 1


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(
        description="Performs inference with a Text2Position model."
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
    parser.add_argument("--num_iter", type=int, default=2, help="Number of iterations.")
    parser.add_argument(
        "--dump_scenes_path",
        type=str,
        default="data/generated_scenes",
        help="Where to dump the scenes",
    )
    parser.add_argument(
        "--pngs_path",
        type=str,
        default="data/AbstractScenes_v1.1/Pngs",
        help="Path to the PNGs.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    generation(
        args.checkpoint_path,
        args.model_type,
        args.test_dataset_path,
        args.visual2index_path,
        args.gen_strategy,
        args.batch_size,
        args.without_text,
        args.num_iter,
        args.dump_scenes_path,
        args.pngs_path,
    )


if __name__ == "__main__":
    main()
