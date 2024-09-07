import csv
import json
import pandas as pd

# CSVファイルの読み込み
csv_file = r'/media/pcs/ボリューム/intern/[機密]カーブ/2024-09-05T11_04_55.csv'  # CSVファイルのパスを指定してください
output_path = r'/media/pcs/ボリューム/intern/[機密]カーブ/2024-09-05-processing.csv'  # 出力CSVファイルのパスを指定してください

# 出力CSVファイルのヘッダー行を定義
output_header = [
    "radar_tgt_lat_loc",
    "radar_tgt_obj_type",
    "radar_tgt_extrapolating_flg",
    "radar_tgt_lat_sp",
    "radar_tgt_low_th_flg",
    "radar_tgt_normal_th_flg",
    "radar_tgt_dist",
    "radar_tgt_relative_sp"
]


create_req_df = pd.DataFrame(columns=['event_unique_id','timestamp','profile_flattened',"OTHFCMTC","OTHTRG08","SP1","VSC_GX0","VSC_GY0"])

bf_event_unique_id = ""

# CSVファイルを読み込み、新しいCSVファイルにデータを書き込みます
with open(csv_file, mode='r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)

    # 各行のデータを処理
    for row in csv_reader:
        # E列のデータをJSONとしてパース
        e_column_data = json.loads(row['profile_flattened'])

        row_data = [row["event_unique_id"], row["timestamp"], row["profile_flattened"], e_column_data["OTHFCMTC"],e_column_data["OTHTRG08"],e_column_data["SP1"],e_column_data["VSC_GX0"],e_column_data["VSC_GY0"]]
        create_req_df = create_req_df.append(pd.Series(row_data, index=create_req_df.columns), ignore_index=True)


create_req_df = create_req_df.sort_values(by=['event_unique_id', 'timestamp', 'profile_flattened'], ascending=[True, True, True])
for value in create_req_df['event_unique_id'].unique():
    # 'A'列の値でフィルタリング
    subset_df = create_req_df[create_req_df['event_unique_id'] == value]
    # CSVファイルとして保存（ファイル名には'A'列の値を使用）
    subset_df.to_csv(output_path + "\\" + f'{value}.csv', index=False)


