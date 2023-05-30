# TorchVisionの全データセットには、特徴量（データ）を変換処理するためのtransformと、ラベルを変換処理するためのtarget_transformという2つのパラメータがある

from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# 最初に大きさ10のゼロテンソルを作成し（10はクラス数に対応）、scatter_ を用いて、ラベルyの値のindexのみ1のワンホットエンコーディングに変換しています。
# ワンホットエンコーディングとは、クラス数分の長さを持つベクトルで、正解ラベルのindexのみ1で、それ以外は0のベクトルです。(数値として扱えない値を0,1で表現)
ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(
        lambda y: torch.zeros(10, dtype=torch.float).scatter_(
            0, torch.tensor(y), value=1
        )
    ),
)
