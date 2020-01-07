from transformers import BertForMaskedLM
from torch import nn
import torch
from torchvision.models import resnet152


class SceneModel(nn.Module):
    def __init__(self, bert_path_or_name: str, finetune: bool):
        super(SceneModel, self).__init__()
        self.finetune = finetune
        self.bert = BertForMaskedLM.from_pretrained("bert-large-uncased")
        self.bert.eval()


class ImageEmbeddingsGenerator(nn.Module):
    def __init__(self):
        super(ImageEmbeddingsGenerator, self).__init__()
        self.resnet = torch.nn.Sequential(
            *(list(resnet152(pretrained=True).children())[:-1])
        )
        self.resnet.eval()
        self.pooler = nn.MaxPool1d(2, stride=2)

    def forward(self, x):
        output = torch.flatten(self.resnet(x), start_dim=1).unsqueeze(1)

        return self.pooler(output)
