import os
import pandas as pd
from torchvision.io import read_image


class CustomImageDataset(Dataset):
    def __init__(
        self, annotations_file, img_dir, transform=None, target_transform=None
    ):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    # データセットのサンプル数を返す
    def __len__(self):
        return len(self.img_labels)

    # 指定されたidxに対応するサンプルをデータセットから読み込んで返す関数
    # indexに基づいて、画像ファイルのパスを特定し、read_imageを使用して画像ファイルをテンソルに変換
    # self.img_labelsから対応するラベルを抜き出す
    # transform functionsを必要に応じて画像およびラベルに適用し、最終的にPythonの辞書型変数で画像とラベルを返す
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {"image": image, "label": label}
        return sample
