import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn
import logging

from datasets import ClipartsDataset
from modeling import ImageEmbeddingsGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dump_word_embeddings(
    cliparts_path: str, visual2index_path: str, save_embeddings_path: str
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Loading ResNet152...")
    resnet_model = ImageEmbeddingsGenerator().to(device)
    dataset = ClipartsDataset(cliparts_path, visual2index_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    embedding_matrix = nn.Embedding(len(dataset) + 1, 768, padding_idx=0).to(device)
    with torch.no_grad():
        for image, index in tqdm(loader):
            image = image.to(device)
            image_embedding = resnet_model(image)
            embedding_matrix.weight[index.item(), :] = image_embedding
    torch.save(embedding_matrix.weight.data, save_embeddings_path)


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Dumps a word embedding matrix.")
    parser.add_argument(
        "--cliparts_path",
        type=str,
        default="data/AbstractScenes_v1.1/Pngs/",
        help="Path to the visual2index mapping json.",
    )
    parser.add_argument(
        "--visual2index_path",
        type=str,
        default="data/visual2index.json",
        help="Path to the visual2index mapping json.",
    )
    parser.add_argument(
        "--save_embeddings_path",
        type=str,
        default="models/cliparts_embeddings.pt",
        help="Where to save the update embeddings.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    dump_word_embeddings(
        args.cliparts_path, args.visual2index_path, args.save_embeddings_path
    )


if __name__ == "__main__":
    main()
