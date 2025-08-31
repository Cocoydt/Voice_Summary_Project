import pandas as pd
import json


def convert_csv_to_jsonl(input_csv, output_jsonl):
    """
    将 CSV 文件转换为 JSON Lines 格式，并处理数据不一致的问题。
    """
    try:
        df = pd.read_csv(input_csv)

        # 填充 ID 列中的空值，然后强制转换为整数
        if 'ID' in df.columns:
            # First, fill any potential empty values in the ID column with a placeholder (e.g., 0)
            df['ID'] = df['ID'].fillna(0).astype(int)

        # Fill any other NaN values with an empty string for consistency
        df = df.fillna('')

        records = df.to_dict('records')

        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for record in records:
                # Automatic handling of the summary field
                summary_text = record.get("summary_ref", "")
                if summary_text:
                    record["summary_ref"] = {"bullets": [summary_text]}
                else:
                    record["summary_ref"] = {"bullets": []}

                # Write the processed record to the JSONL file
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        print(f"成功将 {input_csv} 转换为干净的 {output_jsonl}。")
    except Exception as e:
        print(f"转换失败: {e}")


if __name__ == "__main__":
    # Ensure the CSV filename and path are correct
    input_file = "CHI智能耳机_数据表.csv"
    output_file = "data/labels.jsonl"

    # Remember to replace "你的数据文件名.csv" with your actual file name
    convert_csv_to_jsonl(input_file, output_file)