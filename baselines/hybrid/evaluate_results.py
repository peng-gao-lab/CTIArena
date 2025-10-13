#!/usr/bin/env python3

import json
import re
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import time
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from model_interface.ollama_inference import get_single_prediction

@dataclass
class EvaluationResult:

    file_name: str
    question: str
    expected_answer: str
    approaches: Dict[str, Dict[str, Any]]
    metrics: Dict[str, Dict[str, float]]

@dataclass
class OverallMetrics:

    linkage_accuracy: Dict[str, float]
    explanation_faithfulness: Dict[str, float]
    actionability_accuracy: Dict[str, float]
    factual_precision: Dict[str, float]
    factual_recall: Dict[str, float]
    factual_f1: Dict[str, float]
    temporal_coherence: Dict[str, float]

class CTIEvaluator:

    def __init__(self, model_name: str = "gpt_o3_mini", framework: str = "cwe"):

        self.model_name = model_name
        self.framework = framework
        self.approaches = ["closed_book", "vanilla_rag", "rag_expansion"]
        
        self.evaluation_results: List[EvaluationResult] = []
        
        print(f" Initializing CTI Evaluator with model: {model_name}")
    
    def extract_enumeration_ids(self, text: str) -> List[str]:

        patterns = [
            r'CWE-(\d+)',
            r'CVE-(\d{4}-\d{4,})',
            r'CAPEC-(\d+)',
            r'T\d{4}(?:\.\d{3})?'
        ]
        
        ids = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if pattern.startswith('CWE'):
                    ids.append(f"CWE-{match}")
                elif pattern.startswith('CVE'):
                    ids.append(f"CVE-{match}")
                elif pattern.startswith('CAPEC'):
                    ids.append(f"CAPEC-{match}")
                else:
                    ids.append(match)
        
        return list(set(ids))
    
    def compute_linkage_accuracy(self, predicted_text: str, ground_truth_text: str) -> float:

        predicted_ids = set(self.extract_enumeration_ids(predicted_text))
        ground_truth_ids = set(self.extract_enumeration_ids(ground_truth_text))
        
        if not ground_truth_ids:
            return 1.0 if not predicted_ids else 0.0
        
        correct_matches = len(predicted_ids.intersection(ground_truth_ids))
        total_ground_truth = len(ground_truth_ids)
        
        return correct_matches / total_ground_truth if total_ground_truth > 0 else 0.0
    
    def llm_judge_nli(self, premise: str, hypothesis: str, task_type: str) -> Dict[str, Any]:

        if task_type == "EF":
            prompt = f"""You are a cybersecurity expert evaluating whether a generated explanation is consistent with an authoritative definition.

Authoritative Definition (Premise): {premise}

Generated Explanation (Hypothesis): {hypothesis}

Please determine if the generated explanation is semantically consistent with the authoritative definition. Consider the following aspects:
1. Do the core concepts match?
2. Is the technical detail accurate?
3. Is the security impact consistent?

Respond in JSON format:
{{
    "entailment": "entailment/contradiction/neutral",
    "confidence": 0.0-1.0,
    "reasoning": "reason for judgment"
}}"""
        
        elif task_type == "AA":
            prompt = f"""You are a cybersecurity expert evaluating whether generated mitigations are consistent with authoritative recommendations.

Authoritative Recommendations (Premise): {premise}

Generated Mitigations (Hypothesis): {hypothesis}

Please determine if the generated mitigations are consistent with the authoritative recommendations in terms of feasibility and effectiveness. Consider the following aspects:
1. Technical feasibility
2. Security effectiveness
3. Implementation difficulty
4. Scope coverage

Respond in JSON format:
{{
    "entailment": "entailment/contradiction/neutral",
    "confidence": 0.0-1.0,
    "reasoning": "reason for judgment"
}}"""
        
        elif task_type == "FP":
            prompt = f"""You are a cybersecurity expert evaluating whether generated facts are consistent with a true account.

True Account (Premise): {premise}

Generated Facts (Hypothesis): {hypothesis}

Please determine if the generated facts are consistent with the true account. Consider the following aspects:
1. Factual accuracy
2. Technical correctness
3. Temporal consistency
4. Causal relationships

Respond in JSON format:
{{
    "entailment": "entailment/contradiction/neutral",
    "confidence": 0.0-1.0,
    "reasoning": "reason for judgment"
}}"""
        
        elif task_type == "TC":
            prompt = f"""You are a cybersecurity expert evaluating whether a generated narrative maintains temporal consistency.

Reference Timeline (Premise): {premise}

Generated Narrative (Hypothesis): {hypothesis}

Please determine if the generated narrative maintains consistency in its temporal order and causal relationships. Consider the following aspects:
1. Logical time sequence
2. Rationality of causal relationships
3. Event coherence
4. Consistency

Respond in JSON format:
{{
    "entailment": "entailment/contradiction/neutral",
    "confidence": 0.0-1.0,
    "reasoning": "reason for judgment"
}}"""
        
        try:
            response = get_single_prediction(self.model_name, prompt)
            
            try:
                result = json.loads(response)
                return result
            except json.JSONDecodeError:
                response_lower = response.lower()
                if "entailment" in response_lower:
                    entailment = "entailment" if "entailment" in response_lower else "contradiction" if "contradiction" in response_lower else "neutral"
                else:
                    entailment = "neutral"
                
                confidence_match = re.search(r'confidence["\']?\s*:\s*([0-9.]+)', response, re.IGNORECASE)
                confidence = float(confidence_match.group(1)) if confidence_match else 0.5
                
                return {
                    "entailment": entailment,
                    "confidence": confidence,
                    "reasoning": response
                }
                
        except Exception as e:
            print(f"  LLM judgment failed: {e}")
            return {
                "entailment": "neutral",
                "confidence": 0.0,
                "reasoning": f"Judgment failed: {e}"
            }
    
    def compute_explanation_faithfulness(self, generated_explanation: str, ground_truth_definition: str) -> float:

        result = self.llm_judge_nli(ground_truth_definition, generated_explanation, "EF")
        
        if result["entailment"] == "entailment":
            return result["confidence"]
        elif result["entailment"] == "contradiction":
            return 1.0 - result["confidence"]
        else:
            return 0.5
    
    def compute_actionability_accuracy(self, generated_mitigations: str, ground_truth_mitigations: str) -> float:

        result = self.llm_judge_nli(ground_truth_mitigations, generated_mitigations, "AA")
        
        if result["entailment"] == "entailment":
            return result["confidence"]
        elif result["entailment"] == "contradiction":
            return 1.0 - result["confidence"]
        else:
            return 0.5
    
    def compute_factual_metrics(self, generated_facts: str, ground_truth_facts: str) -> Tuple[float, float, float]:

        generated_fact_list = [fact.strip() for fact in generated_facts.split('\n') if fact.strip()]
        ground_truth_fact_list = [fact.strip() for fact in ground_truth_facts.split('\n') if fact.strip()]
        
        if not ground_truth_fact_list:
            return (1.0, 1.0, 1.0) if not generated_fact_list else (0.0, 1.0, 0.0)
        
        tp = 0
        fp = 0
        
        for gen_fact in generated_fact_list:
            best_score = 0.0
            for gt_fact in ground_truth_fact_list:
                result = self.llm_judge_nli(gt_fact, gen_fact, "FP")
                if result["entailment"] == "entailment":
                    best_score = max(best_score, result["confidence"])
            
            if best_score >= 0.7:
                tp += 1
            else:
                fp += 1
        
        fn = 0
        for gt_fact in ground_truth_fact_list:
            best_score = 0.0
            for gen_fact in generated_fact_list:
                result = self.llm_judge_nli(gt_fact, gen_fact, "FP")
                if result["entailment"] == "entailment":
                    best_score = max(best_score, result["confidence"])
            
            if best_score < 0.7:
                fn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def compute_temporal_coherence(self, generated_narrative: str, reference_timeline: str) -> float:

        result = self.llm_judge_nli(reference_timeline, generated_narrative, "TC")
        
        if result["entailment"] == "entailment":
            return result["confidence"]
        elif result["entailment"] == "contradiction":
            return 1.0 - result["confidence"]
        else:
            return 0.5
    
    def evaluate_single_case(self, case_data: Dict[str, Any]) -> EvaluationResult:

        file_name = case_data.get("file", "")
        question = case_data.get("question", "")
        expected_answer = case_data.get("expected_answer", "")
        results = case_data.get("results", {})
        
        print(f"   Evaluating test case: {file_name}")
        
        case_metrics = {}
        
        for approach in self.approaches:
            if approach not in results:
                continue
            
            approach_result = results[approach]
            if self.framework == 'cwe':
                prediction = approach_result.get("cwe_prediction", "")
            elif self.framework == 'mitre':
                prediction = approach_result.get("mitre_technique", "")
                print("1")
            explanation = approach_result.get("explanation", "")
            mitigations = approach_result.get("mitigations", "")
            
            print(f"    Evaluating approach: {approach},{prediction}")
            
            metrics = {}
            
            metrics["la"] = self.compute_linkage_accuracy(prediction, expected_answer)
            
            
            if explanation and expected_answer:
                
                metrics["ef"] = 0.0  
                metrics["aa"] = 0.0  
                metrics["fp"] = 0.0  
                metrics["fr"] = 0.0  
                metrics["f1"] = 0.0  
                metrics["tc"] = 0.0  
            
            case_metrics[approach] = metrics
            
        
        return EvaluationResult(
            file_name=file_name,
            question=question,
            expected_answer=expected_answer,
            approaches=results,
            metrics=case_metrics
        )
    
    def _get_batch_id_from_results_file(self, results_file: Path) -> str:

        filename = results_file.stem
        parts = filename.split('_')
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return "unknown_batch"
    
    def _get_evaluation_output_dir(self, results_file: Path) -> Path:

        batch_id = self._get_batch_id_from_results_file(results_file)
        output_dir = Path("output") / batch_id
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def evaluate_results_file(self, results_file: Path) -> List[EvaluationResult]:

        print(f" Loading results file: {results_file}")
        
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(" Incorrect results file format, should be a list")
            return []
        
        print(f" Found {len(data)} test cases")
        
        evaluation_results = []
        for i, case_data in enumerate(data):
            print(f"\n{'='*60}")
            print(f" Test case {i+1}/{len(data)}")
            print(f"{'='*60}")
            
            try:
                result = self.evaluate_single_case(case_data)
                evaluation_results.append(result)
            except Exception as e:
                print(f" Failed to evaluate test case: {e}")
                continue
        
        return evaluation_results
    
    def compute_overall_metrics(self, evaluation_results: List[EvaluationResult]) -> OverallMetrics:

        print(f"\n Computing overall metrics...")
        
        metrics = {
            "linkage_accuracy": {},
            "explanation_faithfulness": {},
            "actionability_accuracy": {},
            "factual_precision": {},
            "factual_recall": {},
            "factual_f1": {},
            "temporal_coherence": {}
        }
        
        for approach in self.approaches:
            la_scores = []
            ef_scores = []
            aa_scores = []
            fp_scores = []
            fr_scores = []
            f1_scores = []
            tc_scores = []
            
            for result in evaluation_results:
                if approach in result.metrics:
                    case_metrics = result.metrics[approach]
                    la_scores.append(case_metrics.get("la", 0.0))
                    ef_scores.append(case_metrics.get("ef", 0.0))
                    aa_scores.append(case_metrics.get("aa", 0.0))
                    fp_scores.append(case_metrics.get("fp", 0.0))
                    fr_scores.append(case_metrics.get("fr", 0.0))
                    f1_scores.append(case_metrics.get("f1", 0.0))
                    tc_scores.append(case_metrics.get("tc", 0.0))
            
            metrics["linkage_accuracy"][approach] = np.mean(la_scores) if la_scores else 0.0
            metrics["explanation_faithfulness"][approach] = np.mean(ef_scores) if ef_scores else 0.0
            metrics["actionability_accuracy"][approach] = np.mean(aa_scores) if aa_scores else 0.0
            metrics["factual_precision"][approach] = np.mean(fp_scores) if fp_scores else 0.0
            metrics["factual_recall"][approach] = np.mean(fr_scores) if fr_scores else 0.0
            metrics["factual_f1"][approach] = np.mean(f1_scores) if f1_scores else 0.0
            metrics["temporal_coherence"][approach] = np.mean(tc_scores) if tc_scores else 0.0
        
        return OverallMetrics(
            linkage_accuracy=metrics["linkage_accuracy"],
            explanation_faithfulness=metrics["explanation_faithfulness"],
            actionability_accuracy=metrics["actionability_accuracy"],
            factual_precision=metrics["factual_precision"],
            factual_recall=metrics["factual_recall"],
            factual_f1=metrics["factual_f1"],
            temporal_coherence=metrics["temporal_coherence"]
        )
    
    def generate_evaluation_report(self, evaluation_results: List[EvaluationResult], 
                                 overall_metrics: OverallMetrics, 
                                 results_file: Path, 
                                 output_file: Optional[Path] = None) -> Path:

        if output_file is None:
            output_dir = self._get_evaluation_output_dir(results_file)
            
            filename = results_file.stem
            parts = filename.split('_')
            model_name = parts[-2] if len(parts) >= 3 else "unknown_model"
            
            batch_id = self._get_batch_id_from_results_file(results_file)
            output_filename = f"{batch_id}_{model_name}_evaluation.json"
            output_file = output_dir / output_filename
        
        print(f" Generating evaluation report: {output_file}")
        
        report = {
            "evaluation_info": {
                "timestamp": datetime.now().isoformat(),
                "model_used": self.model_name,
                "judge_model": self.model_name,
                "total_cases": len(evaluation_results),
                "approaches": self.approaches,
                "source_results_file": str(results_file)
            },
            "overall_metrics": {
                "linkage_accuracy": overall_metrics.linkage_accuracy,
                "explanation_faithfulness": overall_metrics.explanation_faithfulness,
                "actionability_accuracy": overall_metrics.actionability_accuracy,
                "factual_precision": overall_metrics.factual_precision,
                "factual_recall": overall_metrics.factual_recall,
                "factual_f1": overall_metrics.factual_f1,
                "temporal_coherence": overall_metrics.temporal_coherence
            },
            "detailed_results": []
        }
        
        for result in evaluation_results:
            case_report = {
                "file_name": result.file_name,
                "question": result.question,
                "expected_answer": result.expected_answer,
                "metrics": result.metrics
            }
            report["detailed_results"].append(case_report)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f" Evaluation report saved to: {output_file}")
        return output_file
    
    def print_summary(self, overall_metrics: OverallMetrics) -> None:

        print(f"\n{'='*80}")
        print(f" Evaluation Summary")
        print(f"{'='*80}")
        
        print(f"\n Linkage Accuracy (LA):")
        for approach, score in overall_metrics.linkage_accuracy.items():
            print(f"  {approach:15}: {score:.3f}")
        
        print(f"\n Explanation Faithfulness (EF):")
        for approach, score in overall_metrics.explanation_faithfulness.items():
            print(f"  {approach:15}: {score:.3f}")
        
        print(f"\n  Actionability Accuracy (AA):")
        for approach, score in overall_metrics.actionability_accuracy.items():
            print(f"  {approach:15}: {score:.3f}")
        
        print(f"\n Factual Metrics:")
        print(f"  {'Approach':15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        for approach in self.approaches:
            if approach in overall_metrics.factual_precision:
                fp = overall_metrics.factual_precision[approach]
                fr = overall_metrics.factual_recall[approach]
                f1 = overall_metrics.factual_f1[approach]
                print(f"  {approach:15} {fp:>10.3f} {fr:>10.3f} {f1:>10.3f}")
        
        print(f"\n Temporal-Coherence Accuracy (TC):")
        for approach, score in overall_metrics.temporal_coherence.items():
            print(f"  {approach:15}: {score:.3f}")

def main():

    parser = argparse.ArgumentParser(description="CTI Arena Hybrid Baselines Automated Evaluation")
    parser.add_argument("--results", type=str, required=True,
                       help="Path to the results file (e.g., batch_0_gpt4o_results.json)")
    parser.add_argument("--model", type=str, default="gpt_o3_mini",
                       help="Name of the model to use for LLM judgments")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to the output evaluation report file (optional, defaults to the same directory structure as the baseline results)")
    parser.add_argument("--framework", type=str, default='cwe',
                       help="Path to the output evaluation report file (optional, defaults to the same directory structure as the baseline results)")
  
    args = parser.parse_args()
    
    results_file = Path(args.results)
    if not results_file.exists():
        print(f" Results file not found: {results_file}")
        return
    
    print(f" Starting CTI Arena Hybrid Baselines automated evaluation")
    print(f" Results file: {results_file}")
    print(f" Judge model: {args.model}")
    
    evaluator = CTIEvaluator(model_name=args.model, framework=args.framework)
    
    start_time = time.time()
    evaluation_results = evaluator.evaluate_results_file(results_file)
    evaluation_time = time.time() - start_time
    
    if not evaluation_results:
        print(" No valid evaluation results")
        return
    
    overall_metrics = evaluator.compute_overall_metrics(evaluation_results)
    
    output_file = None
    if args.output:
        output_file = Path(args.output)
    
    saved_file = evaluator.generate_evaluation_report(evaluation_results, overall_metrics, results_file, output_file)
    
    evaluator.print_summary(overall_metrics)
    
    print(f"\n  Total evaluation time: {evaluation_time:.2f} seconds")
    print(f" Evaluation report saved to: {saved_file}")
    print(f" Evaluation complete!")

if __name__ == "__main__":
    main()