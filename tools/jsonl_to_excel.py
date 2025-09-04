import json
import pandas as pd
from typing import List, Dict


def load_jsonl_data(file_path: str) -> List[Dict]:
    """加载JSONL文件数据"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def create_comparison_table(data1: List[Dict], data2: List[Dict]) -> pd.DataFrame:
    """创建对比表格"""
    comparison_data = []

    for i, (d1, d2) in enumerate(zip(data1, data2), 1):
        # 确保两个文件的任务顺序一致
        if d1.get('msg_type') != d2.get('msg_type'):
            print(f"警告: 第{i}条数据msg_type不匹配: {d1.get('msg_type')} vs {d2.get('msg_type')}")

        # 处理整体摘要
        row = {
            '任务ID': i,
            '任务类型': d1.get('msg_type', 'N/A'),
            '版本A_overall': d1.get('overall_summary', ''),
            '版本B_overall': d2.get('overall_summary', ''),
            '整体摘要差异': ''  # 留空供评估人员填写
        }

        # 处理分段摘要
        segments_a = d1.get('segments', [])
        segments_b = d2.get('segments', [])

        max_segments = max(len(segments_a), len(segments_b))

        for j in range(max_segments):
            seg_a = segments_a[j] if j < len(segments_a) else {}
            seg_b = segments_b[j] if j < len(segments_b) else {}

            row[f'时间段_{j + 1}'] = seg_a.get('time_range', '') or seg_b.get('time_range', '')
            row[f'版本A_segment_{j + 1}'] = seg_a.get('segment_summary', '')
            row[f'版本B_segment_{j + 1}'] = seg_b.get('segment_summary', '')
            row[f'分段差异_{j + 1}'] = ''  # 留空供评估人员填写

        comparison_data.append(row)

    return pd.DataFrame(comparison_data)


def main():
    # 加载数据
    print("正在加载数据...")
    data1 = load_jsonl_data("data/baseline_segments.jsonl")
    data2 = load_jsonl_data("baseline_segments_qwen.jsonl")

    print(f"版本A数据条数: {len(data1)}")
    print(f"版本B数据条数: {len(data2)}")

    # 创建对比表格
    comparison_df = create_comparison_table(data1, data2)

    # 保存为Excel文件（包含格式）
    output_file = "summary_comparison.xlsx"
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        comparison_df.to_excel(writer, sheet_name='摘要对比', index=False)

        # 获取工作簿和工作表对象用于格式设置
        workbook = writer.book
        worksheet = writer.sheets['摘要对比']

        # 设置列宽
        for idx, col in enumerate(comparison_df.columns):
            max_len = max(comparison_df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(idx, idx, min(max_len, 50))

        # 添加标题格式
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })

        # 应用标题格式
        for col_num, value in enumerate(comparison_df.columns.values):
            worksheet.write(0, col_num, value, header_format)

    # 保存为CSV文件
    csv_file = "summary_comparison.csv"
    comparison_df.to_csv(csv_file, index=False, encoding='utf-8-sig')

    print(f"对比表格已生成:")
    print(f"- Excel文件: {output_file}")
    print(f"- CSV文件: {csv_file}")

    # 显示前几行预览
    print("\n预览前2行数据:")
    print(comparison_df.head(2).to_string(index=False))


if __name__ == "__main__":
    main()