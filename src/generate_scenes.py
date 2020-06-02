import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import json
from transformers import BertConfig
from scene_layouts.generation_strategies import generation_strategy_factory
from PIL import Image
import os
from typing import List

from scene_layouts.datasets import (
    DiscreteInferenceDataset,
    ContinuousInferenceDataset,
    collate_pad_batch,
    BUCKET_SIZE,
)
from scene_layouts.modeling import SpatialDiscreteBert, SpatialContinuousBert


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


def generation(
    checkpoint_path: str,
    model_type: str,
    test_dataset_path: str,
    visual2index_path: str,
    gen_strategy: str,
    bert_name: str,
    without_text: bool,
    dump_scenes_path: str,
    pngs_path: str,
):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device {device}! ---")
    # Create dataset
    assert model_type in ["discrete", "continuous"]
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
        collate_fn=collate_pad_batch,
        sampler=test_sampler,
    )
    # Prepare model
    config = BertConfig.from_pretrained(bert_name)
    config.vocab_size = len(visual2index) + 1
    model = nn.DataParallel(
        SpatialDiscreteBert(config, bert_name)
        if model_type == "discrete"
        else SpatialContinuousBert(config, bert_name)
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.train(False)
    print(f"Starting generation from checkpoint {checkpoint_path}!")
    if without_text:
        print("The model won't use the text to generate the scenes.")
    print(f"Using {gen_strategy}!")
    index = 0
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
            ids_text, ids_vis, pos_text, x_ind, y_ind, o_ind, t_types, attn_mask = (
                ids_text.to(device),
                ids_vis.to(device),
                pos_text.to(device),
                x_ind.to(device),
                y_ind.to(device),
                o_ind.to(device),
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

            x_out, y_out, o_out = x_out.cpu(), y_out.cpu(), o_out.cpu()
            visual_names = [
                index2visual[index.item()]
                for index in ids_vis[0]
                if index.item() in index2visual
            ]
            # Dump original
            dump_image_path = os.path.join(
                dump_scenes_path, str(index) + f"-{gen_strategy}" + "-original.png"
            )
            dump_scene(
                pngs_path,
                visual_names,
                x_lab[0, max_ids_text:],
                y_lab[0, max_ids_text:],
                o_lab[0, max_ids_text:],
                dump_image_path,
                bucketized=False,
            )
            # Dump model generated
            dump_image_path = os.path.join(
                dump_scenes_path, str(index) + f"-{gen_strategy}" + "-generated.png"
            )
            dump_scene(
                pngs_path,
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
    parser = argparse.ArgumentParser(description="Generates scenes with a SpatialBERT.")
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
    generation(
        args.checkpoint_path,
        args.model_type,
        args.test_dataset_path,
        args.visual2index_path,
        args.gen_strategy,
        args.bert_name,
        args.without_text,
        args.dump_scenes_path,
        args.pngs_path,
    )


if __name__ == "__main__":
    main()
