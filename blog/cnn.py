# """ライブラリの準備"""
import torch
import torch.nn as nn  # ニューラルネットワーク関数
import torchvision.transforms as transforms  # データの前処理に必要なモジュール
from torchvision.datasets import MNIST  # MNISTデータセット
from torch.utils.data import DataLoader  # データセットの読み込み
import numpy as np  # 行列の演算処理
import matplotlib.pyplot as plt  # グラフの描画

# """Datasetの準備"""
train_dataset = MNIST(
    root="mydata", train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = MNIST(root="mydata", train=False, transform=transforms.ToTensor())


# """DataLoaderを作成"""
# DataLoaderとは「データセットからデータをバッチサイズにまとめて返すモジュール
train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)


# '''モデルの定義'''
class CNN(nn.Module):  # nn.Moduleを継承
    def __init__(self):  # 初期化関数(コンストラクタを定義)
        super(CNN, self).__init__()  # 親クラスのコンストラクタを実行
        self.layer1 = nn.Sequential(  # [第一層]nn.Sequentialを使って層を積み重ねる
            nn.Conv2d(
                1, 16, kernel_size=5, padding=2
            ),  # nn.Conv2dを使って二次元の畳み込み層を定義。(入力1, 出力16, 5×5マス, 余白2マス)
            nn.BatchNorm2d(16),  # 二次元のバッチ正規化(特徴量16)
            nn.ReLU(),  # 活性化関数ReLU
            nn.MaxPool2d(2),  # プーリング層(2×2マスの中で最大の値を出力)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(7 * 7 * 32, 10)  # 入力サイズは7×7×32, 出力サイズは10の全結合層

    def forward(self, x):  # 順伝播の定義
        x = self.layer1(x)  # layer1を実行
        x = self.layer2(x)  # layer2を実行
        x = x.view(x.size(0), -1)  # size0でバッチサイズを指定し、残りの次元を-1で指定することで、次元を自動的に計算
        x = self.fc(x)  # 全結合層を実行
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

# '''最適化手法の定義'''
criterion = nn.CrossEntropyLoss()  # クロスエントロピー誤差
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adamを使用して最適化 (学習率は0.01)


# '''訓練用の関数を定義'''
def train(train_loader):
    model.train()  # モデルを訓練モードに変更
    running_loss = 0  # 損失(loss)の累計を記録する変数
    for images, labels in train_loader:  # ミニバッチ単位で訓練データを取り出す
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()  # 勾配を初期化
        outputs = model(images)  # 予測ラベル
        loss = criterion(outputs, labels)  # 予測ラベルと正解ラベルの誤差(loss)を計算
        running_loss += loss.item()  # 誤差(loss)をrunning_lossへ蓄積
        loss.backward()  # 誤差(loss)を逆伝播させる
        optimizer.step()  # パラメータを更新する
    train_loss = running_loss / len(train_loader)  # 1データあたりの誤差(loss)を計算
    return train_loss


# '''評価用の関数を定義'''
def valid(test_loader):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():  # 評価時には勾配は不要
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            predicted = outputs.max(1, keepdim=True)[1]  # 最大値を持つラベルを予測ラベルとする
            labels = labels.view_as(predicted)  # ラベルを予測ラベルと同じサイズに変換
            correct += predicted.eq(labels).sum().item()  # 正解と予測ラベルが一致した数をカウント
            total += labels.size(0)
    val_loss = running_loss / len(test_loader)
    val_acc = correct / total
    return val_loss, val_acc  # 誤差と正解率を返す


# '''誤差(loss)を記録する空の配列を用意'''
loss_list = []  # 訓練データの誤差
val_loss_list = []  # 評価データの誤差
val_acc_list = []  # 評価データの正解率

# '''学習'''
for epoch in range(10):
    loss = train(train_loader)
    val_loss, val_acc = valid(test_loader)
    print(
        "epoch %d, loss: %.4f val_loss: %.4f val_acc: %.4f"
        % (epoch, loss, val_loss, val_acc)
    )
    loss_list.append(loss)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

# '''学習の結果と使用したモデルを保存'''
np.save("loss_list.npy", np.array(loss_list))  # 訓練データの誤差の推移を保存
np.save("val_loss_list.npy", np.array(val_loss_list))  # 評価データの誤差の推移を保存
np.save("val_acc_list.npy", np.array(val_acc_list))  # 評価データの正解率の推移を保存
torch.save(model.state_dict(), "cnn.pkl")  # 学習済みモデルの保存

# '''結果の表示'''
plt.plot(range(10), loss_list, "r-", label="train_loss")  # 訓練データの誤差の推移をプロット
plt.plot(range(10), val_loss_list, "b-", label="test_loss")  # 評価データの誤差の推移をプロット
plt.legend()  # 凡例を表示
plt.xlabel("epoch")  # x軸はepoch(学習回数)
plt.ylabel("loss")  # y軸はloss(誤差)
plt.figure()  # 新しいウィンドウを描画
plt.plot(range(10), val_acc_list, "g-", label="val_acc")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("acc")
print("正解率：", val_acc_list[-1] * 100, "%")
