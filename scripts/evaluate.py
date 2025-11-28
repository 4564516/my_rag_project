"""
评估脚本：比较预测答案和标准答案
"""
import csv
from pathlib import Path
from typing import Dict, List

def load_answers(file_path: Path) -> Dict[str, Dict]:
    """加载答案文件"""
    answers = {}
    with file_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row.get("question", "").strip()
            if question:
                # 使用问题作为 key，同时保存 id（如果有）
                answers[question] = row
    return answers

def compare_answers(predicted: Dict, ground_truth: Dict) -> Dict:
    """比较预测答案和标准答案"""
    result = {
        "question": predicted.get("question", ""),
        "predicted_value": predicted.get("answer_value", ""),
        "ground_truth_value": ground_truth.get("answer_value", ""),
        "match": False,
        "predicted_answer": predicted.get("answer", ""),
        "ground_truth_answer": ground_truth.get("answer", "")
    }
    
    # 比较答案值
    pred_val = str(predicted.get("answer_value", "")).strip().lower()
    gt_val = str(ground_truth.get("answer_value", "")).strip().lower()
    
    if pred_val == gt_val:
        result["match"] = True
    elif pred_val == "is_blank" and gt_val == "is_blank":
        result["match"] = True
    else:
        # 尝试数值比较
        try:
            pred_num = float(pred_val.replace(",", ""))
            gt_num = float(gt_val.replace(",", ""))
            if abs(pred_num - gt_num) < 0.01:  # 允许小的浮点误差
                result["match"] = True
        except:
            pass
    
    return result

def evaluate(predicted_csv: Path, ground_truth_csv: Path):
    """评估答案质量"""
    print("=== 评估答案质量 ===\n")
    
    predicted = load_answers(predicted_csv)
    ground_truth = load_answers(ground_truth_csv)
    
    results = []
    matched = 0
    total = 0
    
    for question, pred_data in predicted.items():
        if question in ground_truth:
            total += 1
            comparison = compare_answers(pred_data, ground_truth[question])
            results.append(comparison)
            if comparison["match"]:
                matched += 1
            else:
                print(f"❌ 不匹配:")
                print(f"   问题: {question[:60]}...")
                print(f"   预测: {comparison['predicted_value']}")
                print(f"   标准: {comparison['ground_truth_value']}")
                print()
    
    accuracy = (matched / total * 100) if total > 0 else 0
    print(f"\n=== 评估结果 ===")
    print(f"总问题数: {total}")
    print(f"正确答案: {matched}")
    print(f"准确率: {accuracy:.1f}%")
    
    return results

if __name__ == "__main__":
    import sys
    project_root = Path(__file__).parent.parent
    
    # 允许通过命令行参数指定文件
    if len(sys.argv) > 1:
        predicted_csv = Path(sys.argv[1])
    else:
        predicted_csv = project_root / "artifacts/train_answers.csv"  # 默认使用训练答案
    
    if len(sys.argv) > 2:
        ground_truth_csv = Path(sys.argv[2])
    else:
        ground_truth_csv = project_root / "data/train_QA.csv"
    
    if not predicted_csv.exists():
        print(f"预测答案文件不存在: {predicted_csv}")
    elif not ground_truth_csv.exists():
        print(f"标准答案文件不存在: {ground_truth_csv}")
    else:
        evaluate(predicted_csv, ground_truth_csv)

