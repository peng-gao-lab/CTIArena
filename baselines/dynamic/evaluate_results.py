#!/usr/bin/env python3

import json
import re
import csv
import argparse
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from model_interface.ollama_inference import get_single_prediction


def _strip_code_fences(text: str) -> str:
    
    if not isinstance(text, str):
        return text
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def _extract_json_object(text: str) -> str | None:
    
    s = text
    n = len(s)
    start = -1
    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(s):
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue
        else:
            if ch == '"':
                in_string = True
            elif ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        candidate = s[start:i+1].strip()
                        try:
                            json.loads(candidate)
                            return candidate
                        except Exception:
                            pass
    return None


def _string_to_bool_heuristic(text: str) -> bool | None:
    
    if not isinstance(text, str):
        return None
    t = text.strip().lower()
    pos = ["match", "matched", "true", "yes", "entailment", "correct", "exact match"]
    neg = ["no", "false", "mismatch", "wrong", "contradiction", "not match", "does not match"]
    if any(k in t for k in pos) and not any(k in t for k in neg):
        return True
    if any(k in t for k in neg) and not any(k in t for k in pos):
        return False
    return None


def parse_match_field(response_text: str) -> bool:
    
    if response_text is None:
        return False

    raw = str(response_text)
    cleaned = _strip_code_fences(raw)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "match" in parsed:
            return bool(parsed["match"])
        if isinstance(parsed, bool):
            return bool(parsed)
    except Exception:
        pass

    candidate = _extract_json_object(cleaned)
    if candidate:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "match" in parsed:
                return bool(parsed["match"])
        except Exception:
            pass

    m = re.search(r'"match"\s*:\s*(true|false)', cleaned, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true"

    h = _string_to_bool_heuristic(cleaned)
    if h is not None:
        return h

    return False


def split_entities(text: str) -> List[str]:
    
    if not text:
        return []
    lines = [line.strip("- ").strip() for line in text.split("\n") if line.strip()]
    return [line for line in lines if line]


def llm_match(pred_entity: str, golden_entities: list, model="gpt4o-mini") -> bool:
    
    golden_text = "\n- " + "\n- ".join(golden_entities)
    prompt = f"""
You are a cybersecurity evaluation expert.
Determine whether the following **predicted entity** semantically matches any of the **golden entities**. 
Only judge semantic equivalence, not formatting.

Golden Entities:
{golden_text}

Predicted Entity:
- {pred_entity}

Respond in JSON format:
{{
  "match": true/false,
  "reason": "short reasoning why the predicted entity matches or not"
}}
"""
    try:
        response = get_single_prediction(model, prompt)
        is_match = parse_match_field(response)
        return is_match                     
    except Exception as e:
        print(f"Warning: LLM parsing error: {e}")
        return False


def evaluate_case(case, model="gpt4o-mini") -> Dict[str, Any]:
    
    golden_answer = case.get("golden_answer", "")
    golden_entities = split_entities(golden_answer)

    results = {"case_id": case.get("case_id"), "golden": golden_entities, "methods": {}}

    for method in ["vanilla", "cskg"]:
        method_data = case.get("methods", {}).get(method, {})
        answer = method_data.get("llm_answer", {}).get("answer", "")
        pred_entities = split_entities(answer)

        tp, fp = 0, 0
        matched_golden = set()

        for pred in pred_entities:
            match = llm_match(pred, golden_entities, model)
            if match:
                tp += 1
                matched_golden.add(pred)
            else:
                fp += 1

        fn = max(0, len(golden_entities) - tp)  
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        results["methods"][method] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "pred_entities": pred_entities,
        }

    return results


def aggregate_results(all_cases: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    
    totals = {m: {"tp": 0, "fp": 0, "fn": 0} for m in ["vanilla", "cskg"]}

    for case in all_cases:
        for m in totals.keys():
            if m in case["methods"]:
                totals[m]["tp"] += case["methods"][m]["tp"]
                totals[m]["fp"] += case["methods"][m]["fp"]
                totals[m]["fn"] += case["methods"][m]["fn"]

    summary = {}
    for m, vals in totals.items():
        tp, fp, fn = vals["tp"], vals["fp"], vals["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        summary[m] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    return summary


class MetricsCalculator:
    
    
    def __init__(self, corpus_path: str, cluster_data_path: str):
        self.corpus_path = Path(corpus_path)
        self.cluster_data_path = Path(cluster_data_path)
        
        self.ground_truth_clusters = self._load_ground_truth_clusters()
        
        self._init_rag_methods()
        
        print(f"Loaded {len(self.ground_truth_clusters)} ground truth clusters")
    
    def _load_ground_truth_clusters(self) -> Dict[int, set]:
        
        clusters = {}
        
        with self.cluster_data_path.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cluster_id = int(row['object_id'])
                blog_id = int(row['blog_id'])
                
                if cluster_id not in clusters:
                    clusters[cluster_id] = set()
                clusters[cluster_id].add(blog_id)
        
        return clusters
    
    def _init_rag_methods(self):
        
        
        from dynamic_baselines import VanillaRAG, CSKGAugmentedRAG
        
        self.vanilla_rag = VanillaRAG(str(self.corpus_path))
        
        self.cskg_rag = CSKGAugmentedRAG(
            cluster_data_path=str(self.cluster_data_path),
            corpus_path=str(self.corpus_path)
        )
        
        print("Initialized RAG methods")
    
    def _extract_blog_ids_from_filename(self, filename: str, case_data: Dict[str, Any] = None) -> List[int]:
        
        if case_data and 'ground_truth_reference' in case_data:
            blog_ids = []
            for blog_ref in case_data['ground_truth_reference']:
                try:
                    blog_id = int(blog_ref.split('-')[1])
                    blog_ids.append(blog_id)
                except (ValueError, IndexError):
                    continue
            return blog_ids
        return []
    
    def _find_ground_truth_cluster(self, blog_ids: List[int]) -> set:
        
        ground_truth_blog_ids = set()
        
        for blog_id in blog_ids:
            for cluster_id, cluster_blogs in self.ground_truth_clusters.items():
                if blog_id in cluster_blogs:
                    ground_truth_blog_ids.update(cluster_blogs)
                    break
        
        return ground_truth_blog_ids
    
    def _calculate_precision_recall_f1(self, retrieved_blog_ids: set, ground_truth_blog_ids: set) -> Tuple[float, float, float]:
        
        if not retrieved_blog_ids:
            return 0.0, 0.0, 0.0
        
        if not ground_truth_blog_ids:
            return 0.0, 0.0, 0.0
        
        intersection = retrieved_blog_ids.intersection(ground_truth_blog_ids)
        
        precision = len(intersection) / len(retrieved_blog_ids) if retrieved_blog_ids else 0.0
        recall = len(intersection) / len(ground_truth_blog_ids) if ground_truth_blog_ids else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1_score
    
    def evaluate_method(self, method: str, test_cases: List[Dict[str, Any]]) -> Dict[str, float]:
        
        print(f"\nEvaluating {method.upper()} method...")
        
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        valid_cases = 0
        
        for i, case in enumerate(test_cases):
            case_id = case['case_id']
            blog_ids = case['blog_ids']
            
            print(f"  Processing case {i+1}/{len(test_cases)}: {case_id}")
            
            ground_truth_blog_ids = self._find_ground_truth_cluster(blog_ids)
            
            if not ground_truth_blog_ids:
                print(f"    Warning: No ground truth found for case {case_id}")
                continue
            
            try:
                if method == 'vanilla':
                    query = f"BLOG-{blog_ids[0]}" if blog_ids else "test query"
                    retrieved_docs = self.vanilla_rag.retrieve_documents(query, top_k=10, threshold=0.3)
                elif method == 'cskg':
                    query = f"BLOG-{blog_ids[0]}" if blog_ids else "test query"
                    retrieved_docs = self.cskg_rag.retrieve_documents(query, task_type='CSC', top_k=10)
                else:
                    continue
                
                retrieved_blog_ids = {doc['id'] for doc in retrieved_docs}
                
                precision, recall, f1 = self._calculate_precision_recall_f1(
                    retrieved_blog_ids, ground_truth_blog_ids
                )
                
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                valid_cases += 1
                
                print(f"    Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                print(f"    Error processing case {case_id}: {e}")
                continue
        
        avg_precision = total_precision / valid_cases if valid_cases > 0 else 0.0
        avg_recall = total_recall / valid_cases if valid_cases > 0 else 0.0
        avg_f1 = total_f1 / valid_cases if valid_cases > 0 else 0.0
        
        print(f"\n{method.upper()} Results:")
        print(f"  Valid cases: {valid_cases}/{len(test_cases)}")
        print(f"  Average Precision: {avg_precision:.4f}")
        print(f"  Average Recall: {avg_recall:.4f}")
        print(f"  Average F1-Score: {avg_f1:.4f}")
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': avg_f1,
            'valid_cases': valid_cases,
            'total_cases': len(test_cases)
        }
    
    def run_evaluation(self, test_cases_dir: str) -> Dict[str, Dict[str, float]]:
        
        test_cases_dir = Path(test_cases_dir)
        
        test_cases = []
        for json_file in test_cases_dir.glob("*.json"):
            if json_file.name == "generation_report.json":
                continue
                
            try:
                with json_file.open('r', encoding='utf-8') as f:
                    case_data = json.load(f)
                
                blog_ids = self._extract_blog_ids_from_filename(json_file.name, case_data)
                
                test_cases.append({
                    'case_id': json_file.stem,
                    'blog_ids': blog_ids,
                    'data': case_data
                })
                
            except Exception as e:
                print(f"Warning: Error loading {json_file}: {e}")
                continue
        
        print(f"Loaded {len(test_cases)} test cases from {test_cases_dir}")
        
        results = {}
        
        results['vanilla'] = self.evaluate_method('vanilla', test_cases)
        results['cskg'] = self.evaluate_method('cskg', test_cases)
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS COMPARISON")
        print(f"{'='*60}")
        print(f"{'Method':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print(f"{'-'*60}")
        
        for method, metrics in results.items():
            print(f"{method.upper():<15} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f}")
        
        print(f"{'-'*60}")
        
        output_path = Path(__file__).parent / "metrics_results.json"
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_path}")
        
        return results


def evaluate_file(file_path: str, model: str = "gpt4o-mini") -> Dict[str, Dict[str, float]]:
    
    data = json.loads(Path(file_path).read_text(encoding="utf-8"))
    cases = data.get("cases", [])

    evaluated_cases = []
    for case in tqdm(cases, desc=f"Processing {Path(file_path).name}", unit="case"):
        res = evaluate_case(case, model=model)
        evaluated_cases.append(res)

    summary = aggregate_results(evaluated_cases)

    out_dir = Path(file_path).parent / "evaluations"
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"{Path(file_path).stem}_parsed_eval_{timestamp}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"cases": evaluated_cases, "summary": summary}, f, indent=2, ensure_ascii=False)

    print(f"\n{Path(file_path).name} done. Summary:")
    for m, stats in summary.items():
        print(f"  {m:8}: P={stats['precision']:.3f}, R={stats['recall']:.3f}, F1={stats['f1']:.3f}")
    print(f"Results saved to {out_file}\n")

    return summary


def evaluate_folder(folder_path: str, model: str = "gpt4o-mini"):
    
    folder = Path(folder_path)
    result_files = list(folder.glob("*_result.json"))

    if not result_files:
        print(f"No *_result.json found in {folder}")
        return

    print(f"Found {len(result_files)} files in {folder}")

    all_summaries = {}
    for file_path in result_files:
        summary = evaluate_file(file_path, model=model)
        all_summaries[file_path.name] = summary

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = folder / f"overall_summary_{timestamp}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    print(f"\nOverall summary saved to {out_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Dynamic Baseline Results')
    parser.add_argument('--file', type=str, help='Single result file to evaluate')
    parser.add_argument('--folder', type=str, help='Folder containing result files to evaluate')
    parser.add_argument('--model', type=str, default='gpt4o-mini', help='LLM model for evaluation')
    parser.add_argument('--test_cases_dir', type=str, help='Directory containing test case JSON files for metrics calculation')
    parser.add_argument('--corpus_path', type=str, default='corpus/blog.jsonl', help='Path to corpus file')
    parser.add_argument('--cluster_data_path', type=str, default='datasets/blogcluster.csv', help='Path to cluster data file')
    
    args = parser.parse_args()
    
    if args.file:
        evaluate_file(args.file, model=args.model)
    elif args.folder:
        evaluate_folder(args.folder, model=args.model)
    elif args.test_cases_dir:
        calculator = MetricsCalculator(
            corpus_path=args.corpus_path,
            cluster_data_path=args.cluster_data_path
        )
        calculator.run_evaluation(args.test_cases_dir)
    else:
        print("Please specify --file, --folder, or --test_cases_dir")
        sys.exit(1)


if __name__ == "__main__":
    main()
