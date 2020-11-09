import argparse
import json
import os
from typing import List

import torch
from PIL import Image
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
from scene_layouts.generation_strategies import generation_strategy_factory
from scene_layouts.modeling import SpatialContinuousBert, SpatialDiscreteBert


def dump_scene(
    pngs_path: str,
    visual_names: List[str],
    x_indexes: torch.Tensor,
    y_indexes: torch.Tensor,
    o_indexes: torch.Tensor,
    dump_image_path: str,
    bucketized: bool = False,
):
    background = Image.open(os.path.join(pngs_path, "background.png"))
    for visual_name, x_index, y_index, o_index in zip(
        visual_names, x_indexes, y_indexes, o_indexes
    ):
        image = Image.open(os.path.join(pngs_path, visual_name))
        if o_index == 1:
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


def generation(args):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device {device}! ---")
    # Create dataset
    assert args.model_type in ["discrete", "continuous"]
    visual2index = json.load(
        open(os.path.join(args.visuals_dicts_path, "visual2index.json"))
    )
    index2visual = {v: k for k, v in visual2index.items()}
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
    config = BertConfig.from_pretrained(args.bert_name)
    config.vocab_size = len(visual2index) + 1
    model = nn.DataParallel(
        SpatialDiscreteBert(config, args.bert_name)
        if args.model_type == "discrete"
        else SpatialContinuousBert(config, args.bert_name)
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.train(False)
    print(f"Starting generation from checkpoint {args.checkpoint_path}!")
    if args.without_text:
        print("The model won't use the text to generate the scenes.")
    print(f"Using {args.gen_strategy}!")
    index = 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # forward
            batch = {key: val.to(device) for key, val in batch.items()}
            x_out, y_out, o_out = generation_strategy_factory(
                args.gen_strategy,
                args.model_type,
                batch["ids_text"],
                batch["ids_vis"],
                batch["pos_text"],
                batch["t_types"],
                batch["attn_mask"],
                model,
                device,
            )

            x_out, y_out, o_out = x_out.cpu(), y_out.cpu(), o_out.cpu()
            visual_names = [
                index2visual[index.item()]
                for index in batch["ids_vis"][0]
                if index.item() in index2visual
            ]
            # Dump original
            dump_image_path = os.path.join(
                args.dump_scenes_path,
                str(index) + f"-{args.gen_strategy}" + "-original.png",
            )
            dump_scene(
                args.pngs_path,
                visual_names,
                batch["x_lab"][0],
                batch["y_lab"][0],
                batch["o_lab"][0],
                dump_image_path,
                bucketized=False,
            )
            # Dump model generated
            dump_image_path = os.path.join(
                args.dump_scenes_path,
                str(index) + f"-{args.gen_strategy}" + "-generated.png",
            )
            dump_scene(
                args.pngs_path,
                visual_names,
                x_out[0],
                y_out[0],
                o_out[0],
                dump_image_path,
                bucketized=True,
            )
            index += 10


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(
        description="Generates scenes with a Spatial-BERT."
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
        "--gen_strategy",
        type=str,
        default="human_order_discrete",
        help="How to generate the positions during inference",
    )
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
    parser.add_argument(
        "--bert_name",
        type=str,
        default="bert-base-uncased",
        help="The bert model name.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    generation(args)


if __name__ == "__main__":
    main()
