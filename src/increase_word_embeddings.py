import argparse
import torch
from transformers import BertModel
from tqdm import tqdm
from torch.utils.data import DataLoader
import logging

from datasets import ClipartsDataset
from modeling import ImageEmbeddingsGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dump_word_embeddings(
    cliparts_path: str, visual2index_path: str, save_embeddings_path: str
):
    # Check for CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Loading BERT...")
    bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
    logger.info("Loading ResNet152...")
    resnet_model = ImageEmbeddingsGenerator().to(device)
    current_num_embeds = bert_model.embeddings.word_embeddings.num_embeddings
    logger.info(f"Current size of the word embeddings matrix {current_num_embeds}")
    dataset = ClipartsDataset(cliparts_path, visual2index_path)
    bert_model.resize_token_embeddings(current_num_embeds + len(dataset))
    logger.info(
        f"Updated size of the word embedding matrix {bert_model.embeddings.word_embeddings.num_embeddings}"
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    with torch.no_grad():
        for image, index in tqdm(loader):
            # https://github.com/huggingface/transformers/issues/1413
            image = image.to(device)
            image_embedding = resnet_model(image)
            bert_model.embeddings.word_embeddings.weight[index, :] = image_embedding
    torch.save(bert_model.embeddings.word_embeddings.state_dict(), save_embeddings_path)


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
        default="models/updated_embeddings.pt",
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
