import json


def check_jsonl_format(file_path):
    is_valid = True
    print(f"正在检查文件: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                # 检查关键字段是否存在
                if "msg_type" not in data or "summary_ref" not in data:
                    print(f"警告: 第 {i + 1} 行缺少关键字段 'msg_type' 或 'summary_ref'。")
                    is_valid = False
                # 检查 summary_ref 是否为 JSON 对象
                if not isinstance(data.get("summary_ref"), dict):
                    print(f"警告: 第 {i + 1} 行的 'summary_ref' 不是 JSON 对象。")
                    is_valid = False
            except json.JSONDecodeError:
                print(f"错误: 第 {i + 1} 行的 JSON 格式不正确。")
                is_valid = False

    if is_valid:
        print("✅ 数据格式验证成功，所有记录都符合要求。")
    else:
        print("❌ 数据格式存在问题，请检查你的 convert_data.py 或原始 CSV 文件。")


if __name__ == "__main__":
    check_jsonl_format("data/labels.jsonl")