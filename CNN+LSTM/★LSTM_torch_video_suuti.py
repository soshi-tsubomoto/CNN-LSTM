import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from decord import VideoReader
from decord import cpu
import torch.nn.functional as F

# 設定
safe_folder = 'sample_curve_data/curve_sample_learn_videos'  # 入力パス（安全）
test_folder = 'sample_curve_data/curve_sample_test_videos'    # テストデータの入力パス
train_csv_folder = 'sample_curve_data/sample_learn_csv' # 学習データのCSVファイルのディレクトリ
test_csv_folder = 'sample_curve_data/curve_sample_test_csv'   # テストデータのCSVファイルのディレクトリ
frame_size = (64, 64)         # フレームサイズ
max_frames = 220              # 最大フレーム数
epochs = 15                   # エポック数
batch_size = 8                # バッチサイズ


def preprocess_video(video_path, frame_size=frame_size, max_frames=max_frames):
    vr = VideoReader(video_path, ctx=cpu(0))
    num_frames = len(vr)
    frames = []

    for i in range(min(max_frames, num_frames)):
        frame = vr[i].asnumpy()
        frame = cv2.resize(frame, frame_size)
        frames.append(frame)

    while len(frames) < max_frames:
        frames.append(np.zeros((frame_size[0], frame_size[1], 3), dtype=np.float32))

    frames = np.array(frames) / 255.0
    return frames


def preprocess_numerical_data(csv_path, num_frames):
    # CSVファイルを読み込み
    df = pd.read_csv(csv_path)
    
    # 必要なカラムのみを取得（ここでは速度と仮定）
    numerical_data = df[['speed']].values  # 'speed'はCSVのカラム名
    
    # num_framesに合わせてデータをトリミングまたはパディング
    if len(numerical_data) < num_frames:
        pad_size = num_frames - len(numerical_data)
        numerical_data = np.pad(numerical_data, ((0, pad_size), (0, 0)), mode='constant')
    elif len(numerical_data) > num_frames:
        numerical_data = numerical_data[:num_frames]

    return torch.tensor(numerical_data, dtype=torch.float32).unsqueeze(0)  # バッチサイズを追加



class FrameVideoDataset(Dataset):
    def __init__(self, video_path, csv_path, frame_size=frame_size, max_frames=max_frames):
        self.frames = preprocess_video(video_path, frame_size, max_frames)
        self.numerical_data = pd.read_csv(csv_path)
        
        # 動画のフレーム数に合わせてCSVデータを調整
        self.frame_count = len(self.frames)
        self.numerical_data = self.numerical_data.head(self.frame_count)
        self.numerical_data = self.numerical_data.tail(self.frame_count).reset_index(drop=True)
        
        # 必要に応じて、データの長さが動画のフレーム数より短い場合にゼロパディング
        if len(self.numerical_data) < self.frame_count:
            padding_length = self.frame_count - len(self.numerical_data)
            padding_df = pd.DataFrame(np.zeros((padding_length, self.numerical_data.shape[1])), columns=self.numerical_data.columns)
            self.numerical_data = pd.concat([self.numerical_data, padding_df], ignore_index=True)
    
    def __len__(self):
        return self.frame_count
    
    def __getitem__(self, idx):
        frame = torch.tensor(self.frames[idx], dtype=torch.float32)
        numerical_data = torch.tensor(self.numerical_data.iloc[idx].values, dtype=torch.float32)
        return frame, numerical_data


class VideoLSTM(nn.Module):
    def __init__(self):
        super(VideoLSTM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lstm_input_size = 32 * (frame_size[0] // 2) * (frame_size[1] // 2)
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=128, num_layers=2, batch_first=True, dropout=0.5)
        self.numerical_fc = nn.Linear(1, 64)  # 数値データ用のFCレイヤー
        self.fc1 = nn.Linear(128 + 64, 64)  # 元のスケール情報を追加するためのFCレイヤー
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x, numerical_data):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, w, h, c)
        x = self.conv1(x)
        x = self.pool(x)
        x = x.view(batch_size, seq_len, -1)
        x, _ = self.lstm(x)
        x = x[:, -1]  # 最後のタイムステップの出力

        # 数値データの処理
        numerical_data = numerical_data.view(batch_size, -1)
        numerical_data = self.numerical_fc(numerical_data)

        # 結合
        x = torch.cat((x, numerical_data), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)



# モデルのインスタンスを作成
model = VideoLSTM()

# 学習済みモデルの保存
torch.save(model.state_dict(), 'video_lstm_model.pth')




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VideoLSTM().to(device)
model.load_state_dict(torch.load('video_lstm_model.pth'))
model.eval()


# モデルの読み込み（安全性のために weights_only=True を使用）
model.load_state_dict(torch.load('video_lstm_model.pth', weights_only=True))



def predict_anomalies(video_path, csv_path, model, threshold=0.5):
    dataset = FrameVideoDataset(video_path, csv_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    scores = []
    for frames, numerical_data in dataloader:
        frames = frames.unsqueeze(0).to(device)  # (1, seq_len, c, h, w)
        numerical_data = numerical_data.to(device)  # (1, seq_len, num_features)
        with torch.no_grad():
            output = model(frames, numerical_data)
            score = output.item()
            scores.append(score)
    
    return scores



def process_videos(folder, csv_folder, model, threshold=0.5):
    for filename in os.listdir(folder):
        if filename.endswith('.mp4'):
            video_path = os.path.join(folder, filename)
            csv_path = os.path.join(csv_folder, filename.replace('.mp4', '.csv'))  # CSVファイルのパスを推測
            
            if os.path.exists(csv_path):
                scores = predict_anomalies(video_path, csv_path, model, threshold)
                for i, score in enumerate(scores):
                    print(f"Video {filename}, Frame {i+1}, Anomaly Score: {score:.4f}, Prediction: {'Anomaly' if score > 0.6 else 'Normal'}")
            else:
                print(f"CSV file not found for {video_path}")

# テストデータの処理
process_videos(test_folder, test_csv_folder, model)
