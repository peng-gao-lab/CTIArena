#!/usr/bin/env python3
import json
import re
from pathlib import Path
from datetime import datetime
import argparse


def extract_id(text: str, task: str):
    if not text or not isinstance(text, str):
        return None

    if task == "atd":
        match = re.search(r"(T\d{4})(?:\.\d{3})?", text)
    elif task == "esd":
        match = re.search(r"(CAPEC-\d+)", text)
    elif task == "rcm":
        match = re.search(r"(CWE-\d+)", text)
    elif task == "wim":
        match = re.search(r"(CVE-\d{4}-\d+)", text)
    else:
        return None

    return match.group(1) if match else None


def analyze_file(file_path: Path, task: str, parsed_dir: Path):
    data = json.loads(file_path.read_text(encoding="utf-8"))

    total = 0
    stats = {
        "closed_book": {"correct": 0, "total": 0},
        "memory_injected": {"correct": 0, "total": 0}
    }

    parsed_cases = []

    for case in data.get("cases", []):
        total += 1
        ground_truth = case.get("ground_truth", "")
        ground_truth_id = extract_id(ground_truth, task)

        parsed_case = {
            "case_id": case.get("case_id"),
            "question": case.get("question"),
            "ground_truth": ground_truth,
            "ground_truth_id": ground_truth_id,
            "parsed_methods": {}
        }

        for method_name in ["closed_book", "memory_injected"]:
            method_data = case.get("methods", {}).get(method_name, {})
            prediction = method_data.get("prediction", "")
            pred_id = extract_id(prediction, task)

            parsed_case["parsed_methods"][method_name] = {
                "prediction": prediction,
                "pred_id": pred_id,
                "correct": (pred_id == ground_truth_id)
            }

            stats[method_name]["total"] += 1
            if ground_truth_id and pred_id == ground_truth_id:
                stats[method_name]["correct"] += 1

        parsed_cases.append(parsed_case)

    results = {
        "file_name": file_path.name,
        "total_cases": total,
        "methods": {}
    }

    for method, s in stats.items():
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        results["methods"][method] = {
            "correct": s["correct"],
            "total": s["total"],
            "accuracy": round(acc, 3)
        }

    parsed_file = parsed_dir / f"{file_path.stem}_parsed.json"
    with open(parsed_file, "w", encoding="utf-8") as f:
        json.dump(parsed_cases, f, indent=2, ensure_ascii=False)

    return results


def process_folder(folder_path: str, task: str):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder does not exist: {folder}")
        return

    files = list(folder.glob("*_result.json"))
    if not files:
        print("No *_result.json files found")
        return

    result_dir = folder / "result"
    parsed_dir = folder / "parsed"
    result_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for file in files:
        print(f"Analyzing file: {file.name}")
        res = analyze_file(file, task, parsed_dir)
        all_results.append(res)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = result_dir / f"{task}_analysis_{timestamp}.json"

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nAnalysis complete, statistics saved to: {result_file}")
    print(f"Parsed files saved to: {parsed_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze static evaluation results")
    parser.add_argument("--task", choices=["atd", "esd", "rcm", "wim"], required=True,
                        help="Task type: atd (Txxxx), esd (CAPEC-xxx), rcm (CWE-xxx), wim (CVE-xxxx-xxxx)")
    parser.add_argument("--folder", type=str, required=True,
                        help="Path to folder containing *_result.json files")
    args = parser.parse_args()

    process_folder(args.folder, args.task)


if __name__ == "__main__":
    main()