import json
from pathlib import Path

from torchvision import models, transforms
from torchvision.datasets.utils import download_url
import torch


def get_classes():
    if not Path("data/imagenet_class_index.json").exists():
        # ファイルが存在しない場合はダウンロードする。
        download_url("https://git.io/JebAs", "data", "imagenet_class_index.json")

    # クラス一覧を読み込む。
    with open("data/imagenet_class_index.json") as f:
        data = json.load(f)
        class_names = [x["ja"] for x in data]

    return class_names

def predict(pil_image):
    resnet = models.resnet50(pretrained=True)

    #https://pytorch.org/vision/stable/models.html
    transform = transforms.Compose([
        transforms.Resize(256),         # (256, 256)にリサイズ`
        transforms.CenterCrop(224),     # 画像の中心に合わせ(224, 224)で切り抜く
        transforms.ToTensor(),          # テンソル化
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )])

    batch_t = torch.unsqueeze(transform(pil_image), 0)

    resnet.eval()
    out = resnet(batch_t)

    class_names = get_classes()
    # with open('imagenet_classes.txt') as f:
    #     classes = [line.strip() for line in f.readlines()]

    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return [(class_names[idx], prob[idx].item()) for idx in indices[0][:10]]
