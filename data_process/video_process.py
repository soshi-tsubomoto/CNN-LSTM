import pandas as pd
from moviepy.editor import VideoFileClip
import os

def process_videos_from_csv(csv_file):
    # CSVファイルを読み込む
    df = pd.read_csv(csv_file)
    
    # 各行を処理
    for index, row in df.iterrows():
        input_file = row['input']
        output_file = row['output']
        start_time = float(row['start'])
        end_time = float(row['end'])
        
        # ファイルの存在確認
        if not os.path.isfile(input_file):
            print(f"Error: The file '{input_file}' does not exist.")
            continue
        
        # 動画を読み込む
        try:
            clip = VideoFileClip(input_file)
        except Exception as e:
            print(f"Error loading file '{input_file}': {e}")
            continue
        
        # 秒数範囲のチェック
        if start_time < 0 or end_time <= start_time or end_time > clip.duration:
            print(f"Error: Invalid start or end time for file '{input_file}'.")
            clip.close()
            continue
        
        # 指定した秒数範囲でクリップを切り取る
        trimmed_clip = clip.subclip(start_time, end_time)

        # 出力ファイルがすでに存在する場合、上書き確認
        if os.path.isfile(output_file):
            print(f"Warning: Output file '{output_file}' already exists. It will be overwritten.")

        # 切り取ったクリップを出力ファイルとして保存する
        try:
            trimmed_clip.write_videofile(output_file, codec="libx264", audio_codec="aac")
            print(f"Successfully processed '{input_file}' and saved to '{output_file}'.")
        except Exception as e:
            print(f"Error saving file '{output_file}': {e}")
        
        # リソースの解放
        clip.close()
        trimmed_clip.close()

if __name__ == "__main__":
    csv_file = "video_list.csv"  # CSVファイルのパスを指定
    process_videos_from_csv(csv_file)
