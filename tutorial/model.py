import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 独自のネットワークモデルを定義
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


device = "cpu"

model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# サイズ28x28の3枚の画像からなるミニバッチのサンプルを用意し、このミニバッチをネットワークに入力し、各処理による変化を確認していきます。
input_image = torch.rand(3, 28, 28)
print(input_image.size())


# nn.Flattenレイヤーで、2次元（28x28）の画像を、1次元の784ピクセルの値へと変換します。
# ミニバッチの0次元目は、サンプル番号を示す次元で、この次元はnn.Flattenを通しても変化しません（1次元目以降がFlattenされます）。
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# [nn.Linear]
# linear layerは、線形変換を施します。
# linear layerは重みとバイアスのパラメータを保持しています。
layer1 = nn.Linear(in_features=28 * 28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# nn.ReLU
# 非線形な活性化関数は、ニューラルネットワークの入力と出力の間にある、複雑な関係性を表現するために重要な要素です。
# これらの活性化関数は線形変換のあとに、非線形性を加え、ニューラルネットワークの表現力を向上させる役割をします。
# ここでは、nn.ReLUを使って、隠れ層の出力に非線形性を加えます。

print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# nn.Sequential
# nn.Sequentialは、モジュールを順番に格納する箱のような要素です。
# 入力データはnn.Sequentialに定義された順番に各モジュールを伝搬します。

# nn.Softmax
# ニューラルネットワークの最後のlinear layerはlogits [- ∞, ∞] を出力します。
# このlogitsはnn.Softmaxモジュールへと渡されます。
# その結果、採取的な値は[0, 1]の範囲となり、これは各クラスである確率を示します。
# dimパラメータは次元を示しており、dim=1の次元で和を求めると確率の総和なので1になります。
