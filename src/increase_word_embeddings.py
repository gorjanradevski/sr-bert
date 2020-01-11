import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn
import logging

from datasets import ClipartsDataset
from modeling import ImageEmbeddingsGenerator

from transformers import BertModel, BertTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dump_word_embeddings(
    cliparts_path: str, visual2index_path: str, save_embeddings_path: str
):
    # Check for CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = ClipartsDataset(cliparts_path, visual2index_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    logger.info("Loading ResNet152...")
    # Load BERT stuff
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased")
    resnet_model = ImageEmbeddingsGenerator().to(device)
    embedding_matrix = nn.Embedding(len(dataset) + 3, 768, padding_idx=0).to(device)
    with torch.no_grad():
        # Include PAD token, MASK token and SEP token
        for i, token in enumerate(
            [tokenizer.pad_token, tokenizer.mask_token, tokenizer.sep_token]
        ):
            token_index = torch.tensor(
                tokenizer.convert_tokens_to_ids(token)
            ).unsqueeze(0)
            embedding_matrix.weight[i, :] = bert.embeddings.word_embeddings(token_index)
        logger.info("Embeddings for PAD, MASK and SEP token included.")
        for image, index in tqdm(loader):
            # https://github.com/huggingface/transformers/issues/1413
            image = image.to(device)
            image_embedding = resnet_model(image)
            embedding_matrix.weight[index, :] = image_embedding

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
