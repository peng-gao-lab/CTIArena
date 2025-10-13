#!/usr/bin/env python3

import json
import csv
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import shutil

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from model_interface.ollama_inference import get_single_prediction

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class ResultsLoader:
    
    
    def __init__(self, cskg_results_path: str = None, vanilla_rag_results_path: str = None):
        self.cskg_results_path = Path(cskg_results_path) if cskg_results_path else None
        self.vanilla_rag_results_path = Path(vanilla_rag_results_path) if vanilla_rag_results_path else None
        
        self.cskg_results = self._load_results(self.cskg_results_path) if self.cskg_results_path else {}
        self.vanilla_rag_results = self._load_results(self.vanilla_rag_results_path) if self.vanilla_rag_results_path else {}
        
        print(f"Loaded CSKG results: {len(self.cskg_results)} entries")
        print(f"Loaded Vanilla RAG results: {len(self.vanilla_rag_results)} entries")
    
    def _load_results(self, results_path: Path) -> Dict[int, Dict[str, Any]]:
        if not results_path or not results_path.exists():
            return {}
        
        results = {}
        with open(results_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cluster_id = int(row['cluster_id'])
                results[cluster_id] = {
                    'representative_blog_id': int(row['representative_blog_id']),
                    'predicted_blogs': [int(x) for x in row['predicted_blogs'].split(';')],
                    'ground_truth_blogs': [int(x) for x in row['ground_truth_blogs'].split(';')],
                    'predicted_count': int(row['predicted_count']),
                    'ground_truth_count': int(row['ground_truth_count'])
                }
        
        return results


class VanillaRAG:
    
    
    def __init__(self, corpus_path: str):
        self.corpus_path = Path(corpus_path)
        self.corpus_data: List[Dict[str, Any]] = []
        print(f"Initialized VanillaRAG (CSV-based retrieval only)")
    
    def load_corpus(self) -> bool:
        if self.corpus_data:
            print(f"Corpus already loaded ({len(self.corpus_data)} entries)")
            return True
        
        if not self.corpus_path.exists():
            print(f"Error: Corpus file not found: {self.corpus_path}")
            return False
        
        print(f"Loading corpus from {self.corpus_path}...")
        
        self.corpus_data = []
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        entry = json.loads(line)
                        self.corpus_data.append(entry)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON on line {line_num}: {e}")
                        continue
        
        print(f"Loaded {len(self.corpus_data)} entries from corpus")
        return True
    
    def retrieve_documents(self, query: str, top_k: int = 5, threshold: float = 0.5, 
                          predicted_blogs: List[int] = None) -> List[Dict[str, Any]]:
        if not self.load_corpus():
            print("Error: Failed to load corpus")
            return []
        
        if predicted_blogs is None:
            print("Warning: No predicted_blogs provided. Skipping CSV-based retrieval by design.")
            return []
        
        print(f"  Using CSV clustering results: {len(predicted_blogs)} predicted blogs")
        retrieved_docs = []
        for blog_id in predicted_blogs:
            doc = next((d for d in self.corpus_data if d.get('id') == blog_id), None)
            if doc:
                doc_copy = doc.copy()
                doc_copy['similarity_score'] = 1.0
                retrieved_docs.append(doc_copy)
            else:
                print(f"  Warning: Blog ID {blog_id} not found in corpus")
        
        print(f"  Retrieved {len(retrieved_docs)} documents from CSV clustering results")
        return retrieved_docs


class CSKGAugmentedRAG:
    
    
    def __init__(self, cluster_data_path: str, corpus_path: str):
        self.cluster_data_path = Path(cluster_data_path)
        self.corpus_path = Path(corpus_path)
        
        self.cluster_data = self._load_cluster_csv()
        self.corpus_data = {}
        self._load_corpus()
    
    def _load_cluster_csv(self) -> Dict[int, Dict[str, Any]]:
        cluster_data = {}
        try:
            import csv
            with self.cluster_data_path.open('r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    object_id = int(row.get('object_id', 0))
                    if object_id not in cluster_data:
                        cluster_data[object_id] = {
                            'blogs': [],
                            'title': row.get('Title', ''),
                            'type': row.get('type', '')
                        }
                    
                    cluster_data[object_id]['blogs'].append({
                        'blog_id': int(row.get('blog_id', 0)),
                        'title': row.get('Title', ''),
                        'platform': row.get('platform', ''),
                        'link': row.get('link', ''),
                        'type': row.get('type', ''),
                        'publish_date': row.get('publish_date', '')
                    })
        except Exception as e:
            print(f"Warning: Error loading cluster CSV: {e}")
            cluster_data = {}
        
        return cluster_data
    
    def _load_corpus(self):
        with self.corpus_path.open('r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    blog_id = data.get('id')
                    if blog_id is not None:
                        self.corpus_data[blog_id] = data
    
    def retrieve_documents(self, query: str, task_type: str = 'CSC', top_k: int = 5, 
                          predicted_blogs: List[int] = None) -> List[Dict[str, Any]]:
        if predicted_blogs is None:
            print("Warning: No predicted_blogs provided. As required, skipping CSV-based retrieval and returning empty results.")
            return []
        
        print(f"  Using CSV CSKG clustering results: {len(predicted_blogs)} predicted blogs")
        retrieved_docs = []
        for blog_id in predicted_blogs:
            if blog_id in self.corpus_data:
                doc = self.corpus_data[blog_id].copy()
                doc['similarity_score'] = 1.0
                retrieved_docs.append(doc)
            else:
                print(f"  Warning: Blog ID {blog_id} not found in corpus")
        
        print(f"  Retrieved {len(retrieved_docs)} documents from CSV CSKG clustering results")
        return retrieved_docs


class LLMAnswerGenerator:
    
    
    def __init__(self, model: str = "o4-mini", clustering_results_loader=None):
        self.model = model
        self.client = None
        self.clustering_results_loader = clustering_results_loader
        
        if OPENAI_AVAILABLE:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                print("Warning: OPENAI_API_KEY not found in environment variables")
        else:
            print("Warning: OpenAI library not available")
    
    def generate_answer(self, query: str, retrieved_docs: List[Dict[str, Any]], 
                       method: str = "vanilla", task_type: str = None, question: str = None, 
                       cluster_id: int = None) -> Dict[str, Any]:
        
        if not self.client:
            return {
                "answer": "LLM not available. Please configure OpenAI API key.",
                "method": method,
                "task_type": task_type,
                "retrieved_docs_count": len(retrieved_docs)
            }
        
        clustering_info = None
        if self.clustering_results_loader and cluster_id is not None:
            if method == "cskg":
                clustering_info = self.clustering_results_loader.cskg_results.get(cluster_id)
            elif method == "vanilla":
                clustering_info = self.clustering_results_loader.vanilla_rag_results.get(cluster_id)

        context = self._build_context(retrieved_docs, clustering_info)
        prompt = self._build_prompt(query, context, method, task_type, question, clustering_info)
        
        try:
            full_prompt = f"You are a cybersecurity expert analyzing threat intelligence reports.\n\n{prompt}"
            answer = get_single_prediction(self.model, full_prompt)
            
            return {
                "answer": answer,
                "method": method,
                "task_type": task_type,
                "retrieved_docs_count": len(retrieved_docs),
                "model": self.model,
                "context_length": len(context),
                "prompt": prompt
            }
            
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "method": method,
                "task_type": task_type,
                "retrieved_docs_count": len(retrieved_docs),
                "error": str(e)
            }
    
    def _build_context(self, retrieved_docs: List[Dict[str, Any]], clustering_info: Dict[str, Any] = None) -> str:
        context_parts = []

        if clustering_info:
            context_parts.append("=== CLUSTERING PERFORMANCE INFORMATION ===")
            context_parts.append(f"Predicted Blogs: {clustering_info['predicted_blogs']}")
            context_parts.append(f"Ground Truth Blogs: {clustering_info['ground_truth_blogs']}")
            context_parts.append("=" * 50)
            context_parts.append("")
        
        context_parts.append("=== RETRIEVED DOCUMENTS ===")
        for i, doc in enumerate(retrieved_docs, 1):
            doc_info = f"Document {i}:\n"
            doc_info += f"Title: {doc.get('title', 'N/A')}\n"
            doc_info += f"Platform: {doc.get('metadata', {}).get('platform', 'N/A')}\n"
            
            if 'similarity_score' in doc:
                doc_info += f"Similarity Score: {doc['similarity_score']:.4f}\n"
            
            if 'cluster_id' in doc:
                doc_info += f"Cluster ID: {doc['cluster_id']}\n"
                doc_info += f"Cluster Type: {doc['cluster_type']}\n"
            
            content = doc.get('clean_text', '') or doc.get('exe_sum', '')
            if len(content) > 2000:
                content = content[:2000] + "..."
            
            doc_info += f"Content: {content}\n"
            doc_info += "-" * 50 + "\n"
            
            context_parts.append(doc_info)
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str, method: str, task_type: str = None, question: str = None, clustering_info: Dict[str, Any] = None) -> str:
        if task_type == "CSC":
            return self._build_csc_prompt(query, context, method, question, clustering_info)
        elif task_type == "TAP":
            return self._build_tap_prompt(query, context, method, question, clustering_info)
        elif task_type == "MLA":
            return self._build_mla_prompt(query, context, method, question, clustering_info)
        else:
            return self._build_generic_prompt(query, context, method, question, clustering_info)
    
    def _build_csc_prompt(self, query: str, context: str, method: str, question: str = None, clustering_info: Dict[str, Any] = None) -> str:
        task_description = question if question else "Extract campaign intelligence from cross-report analysis"
        
        clustering_context = ""
        if clustering_info:
            clustering_context = f"""

## CLUSTERING PERFORMANCE CONTEXT
The documents were retrieved using {method.upper()} method with the following clustering performance:
- Retrieved {clustering_info['predicted_count']} documents
- Found {len([x for x in clustering_info['predicted_blogs'] if x in clustering_info['ground_truth_blogs']])} out of {clustering_info['ground_truth_count']} relevant documents

Consider this clustering performance when analyzing the retrieved documents and constructing your answer.
"""
        
        return f"""You are a Campaign Storyline Construction (CSC) assistant. Analyze security reports to extract campaign intelligence.

**Use only the provided information. Do not invent facts.**

---

## INPUTS

Latest Blog:
{query}

Related Historical Blogs:
{context}{clustering_context}

---

## QUESTION
- {task_description}
- Do not refer to the blogs by their IDs in the answer.

---

## EXAMPLE

Based on the question: "What is the time span of the EleKtra-Leak campaign based on the earliest and latest report dates?"

Expected answer format:
- Earliest report date: December 2020
- Latest report date: October 6, 2023

Answer:"""

    def _build_tap_prompt(self, query: str, context: str, method: str, question: str = None, clustering_info: Dict[str, Any] = None) -> str:
        task_description = question if question else "Extract threat actor intelligence from cross-report analysis"
        
        clustering_context = ""
        if clustering_info:
            clustering_context = f"""

## CLUSTERING PERFORMANCE CONTEXT
The documents were retrieved using {method.upper()} method with the following clustering performance:
- Retrieved {clustering_info['predicted_count']} documents
- Found {len([x for x in clustering_info['predicted_blogs'] if x in clustering_info['ground_truth_blogs']])} out of {clustering_info['ground_truth_count']} relevant documents

Consider this clustering performance when analyzing the retrieved documents and constructing your answer.
"""
        
        return f"""You are a Threat Actor Profiling (TAP) assistant. Analyze security reports to extract threat actor intelligence.

**Use only the provided information. Do not invent facts.**

---

## INPUTS

Latest Blog:
{query}

Related Historical Blogs:
{context}{clustering_context}

---

## QUESTION
- {task_description}
- Do not refer to the blogs by their IDs in the answer.

---

## EXAMPLE

Based on the question: "What is the canonical threat actor name, resolving any aliases mentioned across these reports?"

Expected answer format:
- Canonical Threat Actor Name: APT29
- Aliases Resolved: Cozy Bear, Nobelium
- Primary Tool/Malware: Cobalt Strike, Mimikatz

Answer:"""

    def _build_mla_prompt(self, query: str, context: str, method: str, question: str = None, clustering_info: Dict[str, Any] = None) -> str:
        task_description = question if question else "Extract malware lineage intelligence from cross-report analysis"
        
        clustering_context = ""
        if clustering_info:
            clustering_context = f"""

## CLUSTERING PERFORMANCE CONTEXT
The documents were retrieved using {method.upper()} method with the following clustering performance:
- Retrieved {clustering_info['predicted_count']} documents
- Found {len([x for x in clustering_info['predicted_blogs'] if x in clustering_info['ground_truth_blogs']])} out of {clustering_info['ground_truth_count']} relevant documents

Consider this clustering performance when analyzing the retrieved documents and constructing your answer.
"""
        
        return f"""You are a Malware Lineage Analysis (MLA) assistant. Analyze security reports to extract malware lineage intelligence.

**Use only the provided information. Do not invent facts.**

---

## INPUTS

Latest Blog:
{query}

Related Historical Blogs:
{context}{clustering_context}

---

## QUESTION
- {task_description}
- Do not refer to the blogs by their IDs in the answer.

---

## EXAMPLE

Based on the question: "What are all the distinct malware variant names mentioned across these reports?"

Expected answer format:
- Emotet v1.0
- Emotet v2.0
- Emotet v3.0
- Emotet v4.0
- Most Notable New Capability: Added worm-like propagation in v3.0

Answer:"""

    def _build_generic_prompt(self, query: str, context: str, method: str, question: str = None, clustering_info: Dict[str, Any] = None) -> str:
        task_description = question if question else "Extract threat intelligence from cross-report analysis"
        
        clustering_context = ""
        if clustering_info:
            clustering_context = f"""

## CLUSTERING PERFORMANCE CONTEXT
The documents were retrieved using {method.upper()} method with the following clustering performance:
- Retrieved {clustering_info['predicted_count']} documents
- Found {len([x for x in clustering_info['predicted_blogs'] if x in clustering_info['ground_truth_blogs']])} out of {clustering_info['ground_truth_count']} relevant documents

Consider this clustering performance when analyzing the retrieved documents and constructing your answer.
"""
        
        return f"""You are analyzing cyber threat intelligence reports. Analyze the provided reports to extract threat intelligence.

**Use only the provided information. Do not invent facts.**

---

## INPUTS

Latest Blog:
{query}

Related Historical Blogs:
{context}{clustering_context}

---

## QUESTION
- {task_description}
- Do not refer to the blogs by their IDs in the answer.

---

## EXAMPLE

Based on the question: "What malware families, threat groups, or vulnerability IDs appear consistently across these reports?"

Expected answer format:
- Malware Families: Emotet, TrickBot
- Threat Actor Groups: APT28, Lazarus Group
- Vulnerability IDs: CVE-2021-34527, CVE-2020-1472

Answer:"""


class DynamicBaselinesEvaluator:
    
    
    def __init__(self, input_dir: str, output_dir: str, corpus_path: str = None, resume: bool = False, 
                 restart: bool = False, task_type: str = 'CSC', cskg_results_path: str = None, 
                 vanilla_rag_results_path: str = None, llm_model: str = None):
        
        project_root = Path(__file__).parent.parent.parent
        in_path = Path(input_dir)
        self.input_dir = in_path if in_path.is_absolute() else project_root / in_path
        
        if corpus_path:
            cp = Path(corpus_path)
            self.corpus_path = cp if cp.is_absolute() else project_root / cp
        else:
            self.corpus_path = project_root / "corpus" / "blog.jsonl"
        
        self.resume = resume
        self.restart = restart
        self.llm_model = llm_model
        self.task_type = task_type
        
        self.output_dir = self._create_output_structure(output_dir)
        
        self.clustering_results_loader = None
        if cskg_results_path or vanilla_rag_results_path:
            resolved_cskg_path = None
            resolved_vanilla_path = None
            
            if cskg_results_path:
                cskg_path = Path(cskg_results_path)
                resolved_cskg_path = str(cskg_path if cskg_path.is_absolute() else project_root / cskg_path)
            
            if vanilla_rag_results_path:
                vanilla_path = Path(vanilla_rag_results_path)
                resolved_vanilla_path = str(vanilla_path if vanilla_path.is_absolute() else project_root / vanilla_path)
            
            self.clustering_results_loader = ResultsLoader(
                cskg_results_path=resolved_cskg_path,
                vanilla_rag_results_path=resolved_vanilla_path
            )
        
        self._init_baselines()
        
        self.progress_file = self.output_dir / f"{self.llm_model}_progress.json"
        self.result_file = self.output_dir / f"{self.llm_model}_result.json"
        
        self.progress = self._load_progress()
        self.results = self._load_results()
        self._corpus_index = None

    def _ensure_corpus_index(self):
        if self._corpus_index is not None:
            return
        self._corpus_index = {}
        try:
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        blog_id = data.get('id')
                        if blog_id is not None:
                            self._corpus_index[int(blog_id)] = data
                    except Exception:
                        continue
        except Exception as e:
            print(f"  Failed to load corpus index from {self.corpus_path}: {e}")

    def _resolve_blog_text(self, blog_id_str: str) -> Optional[str]:
        if not isinstance(blog_id_str, str) or not blog_id_str.startswith('BLOG-'):
            return None
        try:
            blog_id = int(blog_id_str.split('-')[1])
        except Exception:
            return None
        self._ensure_corpus_index()
        data = self._corpus_index.get(blog_id)
        if not data:
            return None
        return data.get('clean_text') or data.get('title')
    
    def _find_cluster_id_from_blog_ids(self, blog_ids: List[str]) -> Optional[int]:
        if not blog_ids:
            return None
        
        cluster_csv_path = Path(__file__).parent.parent.parent / "datasets" / "blogcluster.csv"
        
        try:
            import csv
            with open(cluster_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                cluster_data = list(reader)
            
            numeric_blog_ids = []
            for blog_id_str in blog_ids:
                if isinstance(blog_id_str, str) and blog_id_str.startswith('BLOG-'):
                    try:
                        numeric_id = int(blog_id_str.split('-')[1])
                        numeric_blog_ids.append(numeric_id)
                    except (ValueError, IndexError):
                        continue
            
            if not numeric_blog_ids:
                return None
            
            for row in cluster_data:
                try:
                    row_blog_id = int(row.get('blog_id', 0))
                    if row_blog_id in numeric_blog_ids:
                        cluster_id = int(row.get('object_id', 0))
                        print(f"Found cluster ID {cluster_id} for blog IDs {numeric_blog_ids}")
                        return cluster_id
                except (ValueError, TypeError):
                    continue
            
            print(f"  No cluster ID found for blog IDs {numeric_blog_ids}")
            return None
            
        except Exception as e:
            print(f"  Error reading BlogCluster.csv: {e}")
            return None
    
    def _get_predicted_blogs_for_cluster(self, cluster_id: int, method: str) -> Optional[List[int]]:
        if not self.clustering_results_loader or cluster_id is None:
            return None
        
        if method == "cskg":
            cluster_info = self.clustering_results_loader.cskg_results.get(cluster_id)
        elif method == "vanilla":
            cluster_info = self.clustering_results_loader.vanilla_rag_results.get(cluster_id)
        else:
            cluster_info = None
            
        if cluster_info:
            return cluster_info.get('predicted_blogs', [])
        return None
    
    def _create_output_structure(self, base_output_dir: str) -> Path:
        input_parts = self.input_dir.parts
        
        task_type = None
        
        for part in input_parts:
            if part.upper() in ['CSC', 'MLA', 'TAP']:
                task_type = part.upper()
                break
        
        if not task_type:
            raise ValueError(f"Could not extract task type from input path: {self.input_dir}")
        
        output_dir = Path(base_output_dir) / task_type
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f" Output directory: {output_dir}")
        return output_dir
    
    def _load_progress(self) -> Dict[str, Any]:
        if self.restart:
            if self.output_dir.exists():
                shutil.rmtree(self.output_dir)
                print(f"Restart mode: Removed entire output directory: {self.output_dir}")
                self.output_dir.mkdir(parents=True, exist_ok=True)
                print(f"Restart mode: Recreated output directory: {self.output_dir}")
        
        auto_resume = not self.resume and not self.restart and self.progress_file.exists()
        
        if self.progress_file.exists() and (self.resume or auto_resume) and not self.restart:
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                if auto_resume:
                    print(f"Auto-resume: Found existing progress, continuing from {progress['completed_cases']}/{progress['total_cases']} cases")
                else:
                    print(f"Loaded progress: {progress['completed_cases']}/{progress['total_cases']} cases completed")
                return progress
            except Exception as e:
                print(f"  Error loading progress file: {e}")
                if auto_resume:
                    print("Falling back to fresh start due to corrupted progress file")
        
        return {
            'input_dir': str(self.input_dir),
            'output_dir': str(self.output_dir),
            'start_time': datetime.now().isoformat(),
            'total_cases': 0,
            'completed_cases': 0,
            'failed_cases': 0,
            'completed_case_ids': [],
            'failed_case_ids': [],
            'current_case': None
        }
    
    def _load_results(self) -> Dict[str, Any]:
        if self.result_file.exists() and self.resume:
            try:
                with open(self.result_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                print(f" Loaded existing results: {len(results.get('cases', []))} cases")
                return results
            except Exception as e:
                print(f"  Error loading results file: {e}")
        
        return {
            'evaluation_info': {
                'input_dir': str(self.input_dir),
                'output_dir': str(self.output_dir),
                'corpus_path': str(self.corpus_path),
                'start_time': datetime.now().isoformat(),
                'total_cases': 0,
                'successful_cases': 0,
                'failed_cases': 0
            },
            'cases': []
        }
    
    def _save_progress(self):
        self.progress['last_updated'] = datetime.now().isoformat()
        Path(self.progress_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, indent=2, ensure_ascii=False)
    
    def _save_results(self):
        self.results['evaluation_info']['last_updated'] = datetime.now().isoformat()
        with open(self.result_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
    
    def _init_baselines(self):
        project_root = Path(__file__).parent.parent.parent
        
        self.vanilla_rag = VanillaRAG(str(self.corpus_path))
        
        cluster_data_path = project_root / "datasets" / "blogcluster.csv"
        self.cskg_rag = CSKGAugmentedRAG(str(cluster_data_path), str(self.corpus_path))
        
        self.llm_generator = LLMAnswerGenerator(self.llm_model, self.clustering_results_loader)
        
        print(" Initialized all baseline methods")
    
    def _get_blog_ids_from_filename(self, filename: str) -> List[str]:
        return []
    
    def evaluate_case(self, case_file: Path) -> Dict[str, Any]:
        case_id = case_file.stem
        print(f"\nEvaluating case: {case_id}")
        
        try:
            with open(case_file, 'r', encoding='utf-8') as f:
                case_data = json.load(f)
        except Exception as e:
            print(f" Error loading case {case_id}: {e}")
            return {'case_id': case_id, 'error': str(e)}
    
        query = None
        source_blogs = []
        
        blog_ids = self._get_blog_ids_from_filename(case_file.name)
        
        if not blog_ids and 'ground_truth_reference' in case_data:
            blog_ids = case_data['ground_truth_reference']
        
        if not query and blog_ids:
            query = blog_ids[0]
        
        case_result = {
            'case_id': case_id,
            'query': query,
            'source_blogs': source_blogs,
            'blog_ids': blog_ids,
            'golden_answer': case_data.get('answer', ''),
            'question': case_data.get('question', ''),
            'methods': {}
        }
        
        methods_to_evaluate = ['vanilla', 'cskg']
        
        for method in methods_to_evaluate:
            print(f"   Evaluating {method} method...")
            try:
                method_result = self._evaluate_method(method, case_result)
                case_result['methods'][method] = method_result
            except Exception as e:
                print(f"   Error in {method} method: {e}")
                case_result['methods'][method] = {'error': str(e)}
        
        return case_result
    
    def _evaluate_method(self, method: str, case_result: Dict[str, Any]) -> Dict[str, Any]:
        method_result = {
            'method': method,
            'retrieved_documents': [],
            'llm_answer': None,
            'evaluation_metrics': {},
            'prompt': None
        }
        
        if method == 'vanilla':
            blog_ids = case_result.get('blog_ids') or []
            source_query = blog_ids[0] if blog_ids else None
            query = self._resolve_blog_text(source_query) if source_query else None
            
            cluster_id = self._find_cluster_id_from_blog_ids(case_result.get('blog_ids', []))
            predicted_blogs = self._get_predicted_blogs_for_cluster(cluster_id, "vanilla")
            
            if predicted_blogs:
                retrieved_docs = self.vanilla_rag.retrieve_documents(query, top_k=5, threshold=0.3, predicted_blogs=predicted_blogs)
            else:
                retrieved_docs = self.vanilla_rag.retrieve_documents(query, top_k=5, threshold=0.3)
            filtered_docs = retrieved_docs
            if isinstance(query, str) and query.startswith('BLOG-'):
                try:
                    src_id = int(query.split('-')[1])
                    filtered_docs = [d for d in retrieved_docs if d.get('id') != src_id]
                except Exception:
                    pass
            method_result['retrieved_documents'] = filtered_docs
            
            question = case_result.get('question', '')
            cluster_id = self._find_cluster_id_from_blog_ids(case_result.get('blog_ids', []))
            llm_result = self.llm_generator.generate_answer(query, filtered_docs, method, self.task_type, question, cluster_id)
            method_result['llm_answer'] = llm_result
            
            if 'prompt' in llm_result:
                method_result['prompt'] = llm_result['prompt']
            
        elif method == 'cskg':
            blog_ids = case_result.get('blog_ids') or []
            source_query = blog_ids[0] if blog_ids else None
            resolved = self._resolve_blog_text(source_query) if source_query else None
            query = case_result.get('query') or resolved or source_query \
                or 'Analyze the provided threat intelligence reports and construct a campaign storyline'
            
            cluster_id = self._find_cluster_id_from_blog_ids(case_result.get('blog_ids', []))
            predicted_blogs = self._get_predicted_blogs_for_cluster(cluster_id, "cskg")
            
            if predicted_blogs:
                retrieved_docs = self.cskg_rag.retrieve_documents(query, task_type=self.task_type, top_k=5, predicted_blogs=predicted_blogs)
            else:
                retrieved_docs = self.cskg_rag.retrieve_documents(query, task_type=self.task_type, top_k=5)
            filtered_docs = retrieved_docs
            if source_query and isinstance(source_query, str) and source_query.startswith('BLOG-'):
                try:
                    src_id = int(source_query.split('-')[1])
                    filtered_docs = [d for d in retrieved_docs if d.get('id') != src_id]
                except Exception:
                    pass
            method_result['retrieved_documents'] = filtered_docs
            
            question = case_result.get('question', '')
            cluster_id = self._find_cluster_id_from_blog_ids(case_result.get('blog_ids', []))
            llm_result = self.llm_generator.generate_answer(query, filtered_docs, method, self.task_type, question, cluster_id)
            method_result['llm_answer'] = llm_result
            
            if 'prompt' in llm_result:
                method_result['prompt'] = llm_result['prompt']
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return method_result
    
    def run_evaluation(self):
        print(f" Starting dataset evaluation...")
        print(f" Input directory: {self.input_dir}")
        print(f" Output directory: {self.output_dir}")
        print(f"Resume mode: {self.resume}")
        print(f"Restart mode: {self.restart}")
        
        test_files = list(self.input_dir.glob("*.json"))
        test_files = [f for f in test_files if f.name != "generation_report.json"]
        
        print(f" Found {len(test_files)} test cases")
        
        self.progress['total_cases'] = len(test_files)
        self.results['evaluation_info']['total_cases'] = len(test_files)
        
        if self.resume:
            completed_ids = set(self.progress['completed_case_ids'])
            test_files = [f for f in test_files if f.stem not in completed_ids]
            print(f"Resuming: {len(test_files)} cases remaining")
        
        for i, test_file in enumerate(test_files, 1):
            case_id = test_file.stem
            
            self.progress['current_case'] = case_id
            self._save_progress()
            
            print(f"\nEvaluating case {i}/{len(test_files)}: {case_id}")
            
            try:
                case_result = self.evaluate_case(test_file)
                
                self.results['cases'].append(case_result)
                self.results['evaluation_info']['successful_cases'] += 1
                
                self.progress['completed_cases'] += 1
                self.progress['completed_case_ids'].append(case_id)
                self.progress['current_case'] = None
                
                self._save_progress()
                self._save_results()
                
                print(f" Completed case {case_id}")
                
            except Exception as e:
                print(f" Error evaluating {case_id}: {e}")
                
                self.progress['failed_cases'] += 1
                self.progress['failed_case_ids'].append(case_id)
                self.progress['current_case'] = None
                self.results['evaluation_info']['failed_cases'] += 1
                
                self._save_progress()
                self._save_results()
        
        self.progress['end_time'] = datetime.now().isoformat()
        self.results['evaluation_info']['end_time'] = datetime.now().isoformat()
        self._save_progress()
        self._save_results()
        
        self._print_summary()
    
    def _print_summary(self):
        print(f"\n Evaluation Summary:")
        print(f"  Total cases: {self.results['evaluation_info']['total_cases']}")
        print(f"  Successful: {self.results['evaluation_info']['successful_cases']}")
        print(f"  Failed: {self.results['evaluation_info']['failed_cases']}")
        print(f"  Progress file: {self.progress_file}")
        print(f"  Results file: {self.result_file}")
        
        if self.progress['failed_case_ids']:
            print(f"  Failed cases: {', '.join(self.progress['failed_case_ids'])}")


def main():
    parser = argparse.ArgumentParser(description='Dynamic Baseline Evaluation')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing test cases')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for evaluation results')
    parser.add_argument('--corpus_path', type=str, 
                       default='corpus/blog.jsonl',
                       help='Path to corpus file')
    parser.add_argument('--resume', action='store_true',
                       help='Resume evaluation from previous progress')
    parser.add_argument('--llm_model', type=str, required=True,
                       help='LLM model to use for answer generation')
    parser.add_argument('--restart', action='store_true',
                       help='Restart evaluation from beginning (clears existing progress and results)')
    parser.add_argument('--task_type', type=str, default='CSC',
                       choices=['CSC', 'TAP', 'MLA'],
                       help='Task type for CSKG method (CSC=Campaign, TAP=Threat Actor, MLA=Malware)')
    parser.add_argument('--cskg-results', type=str,
                       default='baselines/dynamic/index/cskg_results.csv',
                       help='Path to CSKG clustering results CSV file')
    parser.add_argument('--vanilla-rag-results', type=str,
                       default='baselines/dynamic/index/vanilla_results.csv',
                       help='Path to Vanilla RAG clustering results CSV file')
    
    args = parser.parse_args()
    
    if args.resume and args.restart:
        print(" Error: Cannot use both --resume and --restart flags simultaneously")
        print("   Use --resume to continue from previous progress")
        print("   Use --restart to start fresh (clears existing progress)")
        sys.exit(1)
    
    evaluator = DynamicBaselinesEvaluator(
        input_dir=args.input_dir, 
        output_dir=args.output_dir, 
        corpus_path=args.corpus_path, 
        resume=args.resume, 
        restart=args.restart, 
        task_type=args.task_type, 
        cskg_results_path=args.cskg_results, 
        vanilla_rag_results_path=args.vanilla_rag_results, 
        llm_model=args.llm_model
    )
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
