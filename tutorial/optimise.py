import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),  # 28 * 28 = 784を入力として、512を出力とする線形変換を行う
            nn.ReLU(),  # 非線形性を加えて、出力を正の値はそのまま、負の値は0にする
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU(),
        )

    # ネットワークの順伝播を定義
    def forward(self, x):
        x = self.flatten(x)  # 28 * 28の画像を784の1次元配列に変換
        logits = self.linear_relu_stack(x)  # 線形変換とReLUを行う
        return logits


# Number of Epochs：イテレーション回数
# Batch Size：ミニバッチサイズを構成するデータ数
# Learning Rate：パラメータ更新の係数。値が小さいと変化が少なく、大きすぎると訓練に失敗する可能性が生まれる
learning_rate = 1e-3
batch_size = 64
epochs = 5

# 訓練ループ：データセットに対して訓練を実行し、パラメータを収束させます
# 検証 / テストループ：テストデータセットでモデルを評価し、性能が向上しているか確認します

# 損失関数loss functionの初期化、定義
loss_fn = nn.CrossEntropyLoss()

model = NeuralNetwork()

# 最適化アルゴリズム：Optimization algorithms
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# 実装
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # 予測と損失の計算
        pred = model(X)
        loss = loss_fn(pred, y)

        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
