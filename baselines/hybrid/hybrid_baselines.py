import json 
import os
import sys
import time
import re
import argparse
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from model_interface.ollama_inference import get_single_prediction

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import openai
    from openai import OpenAI
except ImportError:
    raise ImportError("OpenAI library not installed. Run: pip install openai")

try:
    import faiss
except ImportError:
    raise ImportError("FAISS library not installed. Run: pip install faiss-cpu")

@dataclass
class RAGResult:

    cwe_id: str
    title: str
    description: str
    similarity_score: float
    content: str
    atomic_behavior: Optional[str] = None
    behavior_index: Optional[int] = None

@dataclass
class BaselineResult:

    approach: str
    cwe_prediction: str
    explanation: str
    mitigations: str
    closed_book_llm_answer: Optional[str] = None
    vanilla_rag_llm_answer: Optional[str] = None
    rag_expansion_llm_answer: Optional[str] = None
    retrieval_results: Optional[List[RAGResult]] = None
    processing_time: float = 0.0
    decomposed_behaviors: Optional[List[str]] = None

class HybridBaselines:

    def __init__(self, corpus_dir: str = "corpus", embedding_model: str = "text-embedding-3-large", 
                 llm_model: str = "o3-mini", framework: str = 'cve'):

        self.corpus_dir = Path(corpus_dir)
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.framework = framework
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = OpenAI(api_key=api_key)
        
        self.corpus_data: List[Dict[str, Any]] = []
        self.corpus_embeddings: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        
        self.empty_description_cwe_ids: set = set()
        
        self.cache_dir = Path("cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f" Initialized HybridBaselines with model: {llm_model}")
        print(f" Cache directory: {self.cache_dir}")
    
    def _get_cache_key(self, corpus_file: str) -> str:

        content = f"{corpus_file}_{self.embedding_model}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_fallback_description(self, entry_index: int) -> str:

        entry = self.corpus_data[entry_index]
        
        try:
            if "_full_content" in entry:
                return entry["_full_content"]
            
            if "contents" in entry:
                try:
                    contents_data = json.loads(entry["contents"]) if isinstance(entry["contents"], str) else entry["contents"]
                    if isinstance(contents_data, dict):
                        if "Description" in contents_data and contents_data["Description"]:
                            return contents_data["Description"]
                        elif "Summary" in contents_data and contents_data["Summary"]:
                            return contents_data["Summary"]
                        elif "@Name" in contents_data and contents_data["@Name"]:
                            return contents_data["@Name"]
                        elif "Extended_Description" in contents_data and contents_data["Extended_Description"]:
                            return contents_data["Extended_Description"]
                        elif "Potential_Mitigations" in contents_data and contents_data["Potential_Mitigations"]:
                            return contents_data["Potential_Mitigations"]
                except json.JSONDecodeError:
                    pass
            
            if "description" in entry and entry["description"]:
                return entry["description"]
            
            if "title" in entry and entry["title"]:
                return entry["title"]
            
            if "id" in entry and entry["id"]:
                return f"CWE-{entry['id']}"
                
        except Exception:
            pass
        
        return "Unknown CWE Entry"
    
    def _get_batch_id(self, dataset_path: str) -> str:

        dataset_dir = Path(dataset_path)
        full_name = dataset_dir.name
        batch_id = full_name.split('_')[0] + '_' + full_name.split('_')[1] if '_' in full_name else full_name
        return batch_id
    
    def _get_output_dir(self, dataset_path: str) -> Path:

        batch_id = self._get_batch_id(dataset_path)
        output_dir = Path("output") / self.framework / batch_id
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def _get_progress_file_path(self, dataset_path: str) -> Path:

        output_dir = self._get_output_dir(dataset_path)
        batch_id = self._get_batch_id(dataset_path)
        progress_file = output_dir / f"{batch_id}_{self.llm_model}_progress.json"
        return progress_file
    
    def _save_progress(self, dataset_path: str, progress_data: Dict[str, Any]) -> bool:

        try:
            progress_file = self._get_progress_file_path(dataset_path)
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)
            print(f" Progress saved to {progress_file}")
            return True
        except Exception as e:
            print(f"  Error saving progress: {e}")
            return False
    
    def _load_progress(self, dataset_path: str) -> Optional[Dict[str, Any]]:

        try:
            progress_file = self._get_progress_file_path(dataset_path)
            if progress_file.exists():
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                print(f" Progress loaded from {progress_file}")
                return progress_data
            else:
                print(f" No progress file found at {progress_file}")
                return None
        except Exception as e:
            print(f"  Error loading progress: {e}")
            return None
    
    def _cleanup_progress(self, dataset_path: str) -> bool:

        try:
            progress_file = self._get_progress_file_path(dataset_path)
            if progress_file.exists():
                progress_file.unlink()
                print(f"  Progress file cleaned up: {progress_file}")
            return True
        except Exception as e:
            print(f"  Error cleaning up progress: {e}")
            return False
    
    def _get_cache_paths(self, framework: str) -> Tuple[Path, Path, Path]:

        cache_key = self._get_cache_key(str(self.corpus_dir / f"{framework}.jsonl"))
        cache_dir = self.cache_dir / framework
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        corpus_path = cache_dir / f"{cache_key}_corpus.pkl"
        embedding_path = cache_dir / f"{cache_key}_embedding.pkl"
        index_path = cache_dir / f"{cache_key}_index.pkl"
        
        return corpus_path, embedding_path, index_path
    
    def _load_from_cache(self, framework: str) -> bool:

        corpus_path, embedding_path, index_path = self._get_cache_paths(framework)
        
        if not all([corpus_path.exists(), embedding_path.exists(), index_path.exists()]):
            print(f" Cache files not found for {framework}")
            return False
        
        try:
            print(f" Loading {framework.upper()} from cache...")
            
            with open(corpus_path, 'rb') as f:
                self.corpus_data = pickle.load(f)
            print(f"   Loaded corpus data: {len(self.corpus_data)} entries")
            
            with open(embedding_path, 'rb') as f:
                self.corpus_embeddings = pickle.load(f)
            
            if not isinstance(self.corpus_embeddings, np.ndarray):
                print(f"    Embeddings are not numpy array, type: {type(self.corpus_embeddings)}")
                return False
            
            print(f"   Loaded embeddings with shape: {self.corpus_embeddings.shape}")
            
            with open(index_path, 'rb') as f:
                self.faiss_index = pickle.load(f)
            print(f"   Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            
            cache_key = self._get_cache_key(str(self.corpus_dir / f"{framework}.jsonl"))
            cache_dir = self.cache_dir / framework
            empty_ids_path = cache_dir / f"{cache_key}_empty_ids.pkl"
            
            if empty_ids_path.exists():
                with open(empty_ids_path, 'rb') as f:
                    self.empty_description_cwe_ids = pickle.load(f)
                print(f"   Loaded empty description IDs: {len(self.empty_description_cwe_ids)} entries")
            else:
                print(f"    No empty description IDs cache found, initializing empty set")
                self.empty_description_cwe_ids = set()
            
            print(f" Successfully loaded {framework.upper()} from cache")
            return True
            
        except Exception as e:
            print(f"  Error loading from cache: {e}")
            print(f"   Cache files: {corpus_path.exists()}, {embedding_path.exists()}, {index_path.exists()}")
            try:
                if corpus_path.exists():
                    corpus_path.unlink()
                if embedding_path.exists():
                    embedding_path.unlink()
                if index_path.exists():
                    index_path.unlink()
                print(f"    Cleaned up corrupted cache files")
            except Exception as cleanup_error:
                print(f"    Error cleaning up cache: {cleanup_error}")
            return False
    
    def _save_to_cache(self, framework: str) -> bool:

        if not self.corpus_data or self.corpus_embeddings is None or self.faiss_index is None:
            print(f"  No data to cache for {framework}")
            return False
        
        corpus_path, embedding_path, index_path = self._get_cache_paths(framework)
        
        try:
            print(f" Saving {framework.upper()} to cache...")
            
            with open(corpus_path, 'wb') as f:
                pickle.dump(self.corpus_data, f)
            print(f"   Saved corpus data: {len(self.corpus_data)} entries")
            
            if not isinstance(self.corpus_embeddings, np.ndarray):
                print(f"    Converting embeddings to numpy array")
                self.corpus_embeddings = np.array(self.corpus_embeddings, dtype=np.float32)
            
            with open(embedding_path, 'wb') as f:
                pickle.dump(self.corpus_embeddings, f)
            print(f"   Saved embeddings with shape: {self.corpus_embeddings.shape}")
            
            with open(index_path, 'wb') as f:
                pickle.dump(self.faiss_index, f)
            print(f"   Saved FAISS index with {self.faiss_index.ntotal} vectors")
            
            cache_key = self._get_cache_key(str(self.corpus_dir / f"{framework}.jsonl"))
            cache_dir = self.cache_dir / framework
            empty_ids_path = cache_dir / f"{cache_key}_empty_ids.pkl"
            with open(empty_ids_path, 'wb') as f:
                pickle.dump(self.empty_description_cwe_ids, f)
            print(f"   Saved empty description IDs: {len(self.empty_description_cwe_ids)} entries")
            
            print(f" Successfully saved {framework.upper()} cache to {self.cache_dir / framework}")
            return True
            
        except Exception as e:
            print(f"  Error saving to cache: {e}")
            return False
    
    def load_corpus(self, framework: str = "cwe", force_reload: bool = False) -> bool:

        if self.corpus_data and not force_reload:
            print(f" Corpus already loaded ({len(self.corpus_data)} entries)")
            return True
        
        if not force_reload and self._load_from_cache(framework):
            return True
        
        corpus_file = self.corpus_dir / f"{framework}.jsonl"
        if not corpus_file.exists():
            print(f" Corpus file not found: {corpus_file}")
            return False
        
        print(f" Loading {framework.upper()} corpus from {corpus_file}...")
        
        self.corpus_data = []
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        entry = json.loads(line)
                        self.corpus_data.append(entry)
                    except json.JSONDecodeError as e:
                        print(f"  Warning: Invalid JSON on line {line_num}: {e}")
                        continue
        
        print(f" Loaded {len(self.corpus_data)} entries from {framework.upper()} corpus")
        
        print(f"Extracting descriptive content for {framework.upper()}...")
        content_texts = []
        
        for i, entry in enumerate(self.corpus_data):
            if framework == "cwe":
                content = entry.get('contents', {})
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except json.JSONDecodeError:
                        pass
                
                description = ""
                if isinstance(content, dict):
                    description = content.get('Description', '')
                
                if not description or not description.strip():
                    cwe_id = entry.get('cwe_id', entry.get('id', ''))
                    if cwe_id:
                        self.empty_description_cwe_ids.add(cwe_id)
                        print(f"    Empty description for CWE-{cwe_id}, skipping embedding")
                    content_texts.append(None)
                else:
                    content_texts.append(description)
                
            elif framework == "cve":
                content = entry.get('contents', {})
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except json.JSONDecodeError:
                        pass
                
                description = ""
                if isinstance(content, dict):
                    descriptions = content.get('descriptions', [])
                    if descriptions and len(descriptions) > 0:
                        description = descriptions[0].get('value', '')
                
                if not description or not description.strip():
                    cve_id = entry.get('cve_id', entry.get('id', ''))
                    if cve_id:
                        self.empty_description_cwe_ids.add(cve_id)
                        print(f"    Empty description for CVE-{cve_id}, skipping embedding")
                    content_texts.append(None)
                else:
                    content_texts.append(description)
                
            elif framework in ["mitre", "capec"]:
                content = entry.get('contents', {})
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except json.JSONDecodeError:
                        pass
                
                description = ""
                if isinstance(content, dict):
                    description = content.get('description', '')
                
                if not description or not description.strip():
                    entry_id = entry.get('id', '')
                    if entry_id:
                        self.empty_description_cwe_ids.add(entry_id)
                        print(f"    Empty description for {framework.upper()}-{entry_id}, skipping embedding")
                    content_texts.append(None)
                else:
                    content_texts.append(description)
            
            else:
                title = entry.get('title', '')
                if not title or not title.strip():
                    entry_id = entry.get('id', '')
                    if entry_id:
                        self.empty_description_cwe_ids.add(entry_id)
                        print(f"    Empty title for entry {entry_id}, skipping embedding")
                    content_texts.append(None)
                else:
                    content_texts.append(title)
        
        print(f" Creating embeddings for {len(content_texts)} entries...")
        embeddings = []
        
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(content_texts):
            if text is None:
                print(f"    Skipping entry {i} due to empty description")
                continue
            elif text and text.strip() and len(text.strip()) > 0:
                valid_texts.append(text.strip())
                valid_indices.append(i)
            else:
                fallback_text = self._get_fallback_description(i)
                valid_texts.append(fallback_text)
                valid_indices.append(i)
                print(f"    Empty description for entry {i}, using fallback: {fallback_text[:50]}...")
        
        print(f"   Processing {len(valid_texts)} valid texts (filtered from {len(content_texts)} total)")
        empty_ids_preview = sorted(list(self.empty_description_cwe_ids))[:10]
        empty_ids_display = f"{empty_ids_preview}{'...' if len(self.empty_description_cwe_ids) > 10 else ''}"
        print(f"   Recorded {len(self.empty_description_cwe_ids)} entries with empty descriptions: {empty_ids_display}")
        
        batch_size = 100
        for i in range(0, len(valid_texts), batch_size):
            batch_texts = valid_texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch_texts
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                print(f"   Processed batch {i//batch_size + 1}/{(len(valid_texts) + batch_size - 1)//batch_size}")
                
            except Exception as e:
                print(f"  Error creating embeddings for batch {i//batch_size + 1}: {e}")
                batch_embeddings = [[0.0] * 3072] * len(batch_texts)
                embeddings.extend(batch_embeddings)
        
        self.corpus_embeddings = np.array(embeddings, dtype=np.float32)

        if self.corpus_embeddings.size == 0:
            print("    No valid texts to embed; creating empty FAISS index.")
            dimension = 3072
            self.corpus_embeddings = np.zeros((0, dimension), dtype=np.float32)
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self._save_to_cache(framework)
            return True

        faiss.normalize_L2(self.corpus_embeddings)
        
        dimension = self.corpus_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(self.corpus_embeddings)
        
        print(f" Created FAISS index with {self.faiss_index.ntotal} vectors (dimension: {dimension})")
        
        self._save_to_cache(framework)
        
        return True
    
    def get_query_embedding(self, query: str) -> Optional[np.ndarray]:

        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=[query]
            )
            
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            faiss.normalize_L2(embedding.reshape(1, -1))
            return embedding
            
        except Exception as e:
            print(f"  Error creating query embedding: {e}")
            return None
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 10, 
                      threshold: float = 0.5) -> List[RAGResult]:

        if self.faiss_index is None or self.corpus_data is None:
            print("  Corpus not loaded. Call load_corpus() first.")
            return []
        
        similarities, indices = self.faiss_index.search(
            query_embedding.reshape(1, -1), top_k
        )
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if sim >= threshold:
                entry = self.corpus_data[idx]
                
                if self.framework == "mitre":
                    cwe_id = entry.get('mitre_id', entry.get('id', ''))
                elif self.framework == "cve":
                    cwe_id = entry.get('cve_id', entry.get('id', ''))
                else:
                    cwe_id = entry.get('cwe_id', entry.get('id', ''))
                title = entry.get('title', '')
                
                description = self._get_fallback_description(idx)
                
                result = RAGResult(
                    cwe_id=cwe_id,
                    title=title,
                    description=description,
                    similarity_score=float(sim),
                    content=str(entry.get('contents', ''))
                )
                results.append(result)
        
        return results
    
    def _enrich_candidates_with_corpus(self, candidates: List[RAGResult]) -> List[RAGResult]:

        enriched_candidates = []
        
        for candidate in candidates:
            entry_from_corpus = None
            for entry in self.corpus_data:
                if self.framework == "mitre":
                    if (entry.get('mitre_id', entry.get('id', '')) == candidate.cwe_id or
                        entry.get('id', '') == candidate.cwe_id):
                        entry_from_corpus = entry
                        break
                elif self.framework == "cve":
                    if (entry.get('cve_id', entry.get('id', '')) == candidate.cwe_id or
                        entry.get('id', '') == candidate.cwe_id):
                        entry_from_corpus = entry
                        break
                else:
                    if (entry.get('cwe_id', entry.get('id', '')) == candidate.cwe_id or 
                        entry.get('id', '') == candidate.cwe_id):
                        entry_from_corpus = entry
                        break
            
            if entry_from_corpus:
                enriched_description = self._get_fallback_description(
                    self.corpus_data.index(entry_from_corpus)
                )
                
                enriched_candidate = RAGResult(
                    cwe_id=candidate.cwe_id,
                    title=candidate.title,
                    description=enriched_description,
                    similarity_score=candidate.similarity_score,
                    content=candidate.content,
                    atomic_behavior=candidate.atomic_behavior,
                    behavior_index=candidate.behavior_index
                )
                enriched_candidates.append(enriched_candidate)
            else:
                enriched_candidates.append(candidate)
        
        return enriched_candidates
    
    def _decompose_vulnerability(self, vulnerability_text: str) -> List[str]:

        print(f"     Decomposing vulnerability into atomic behaviors...")
        
        prompt = f"""You are a cybersecurity expert. Analyze the following vulnerability description and identify its core atomic behaviors.

Vulnerability: "{vulnerability_text}"

Instructions:
- Extract 2-4 key atomic behaviors that describe the vulnerability itself
- Each behavior should be a specific, actionable security issue
- Focus on the vulnerability mechanics, not mitigations or solutions
- Use clear, concise language describing what the vulnerability does
- Avoid splitting into too many granular behaviors

Example format:
["behavior 1", "behavior 2", "behavior 3"]

Return only the JSON array:"""

        try:
            print(f"       Calling LLM for vulnerability decomposition...")
            response = self.call_llm_api(prompt)
            
            print(f"      LLM response: {repr(response)}")
            
            if not response:
                print(f"        LLM decomposition failed (no response), using original vulnerability")
                return [vulnerability_text]
            
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                try:
                    atomic_behaviors = json.loads(json_match.group())
                    if isinstance(atomic_behaviors, list) and len(atomic_behaviors) > 0:
                        cleaned_behaviors = []
                        for behavior in atomic_behaviors:
                            if isinstance(behavior, str) and behavior.strip():
                                cleaned_behaviors.append(behavior.strip())
                        
                        if cleaned_behaviors:
                            print(f"       Decomposed into {len(cleaned_behaviors)} atomic behaviors")
                            return cleaned_behaviors
                        else:
                            print(f"        No valid behaviors found after cleaning")
                except json.JSONDecodeError as e:
                    print(f"        JSON parsing error in decomposition: {e}")
                    print(f"       Raw response: {response}")
            else:
                print(f"        No JSON array found in response")
                print(f"       Raw response: {response}")
            
            print(f"        Decomposition parsing failed, using original vulnerability")
            return [vulnerability_text]
            
        except Exception as e:
            print(f"        Error during vulnerability decomposition: {e}")
            return [vulnerability_text]
    
    def _llm_judge_cwe_mapping(self, full_vulnerability: str, atomic_behavior: str, 
                               candidates: List[RAGResult]) -> List[RAGResult]:

        label = "ATT&CK" if self.framework == "mitre" else ("CVE" if self.framework == "cve" else "CWE")
        print(f"       Using LLM to judge {label} mapping for atomic behavior: {atomic_behavior[:80]}...")
        
        if not candidates:
            return []
        
        
        candidates_text = ""
        for i, candidate in enumerate(candidates, 1):
            id_label = candidate.cwe_id if self.framework == "mitre" else f"CWE-{candidate.cwe_id}"
            candidates_text += f"{i}. {id_label}: {candidate.title}\n"
            candidates_text += f"   Description: {candidate.description[:500]}...\n"
            candidates_text += f"   Similarity Score: {candidate.similarity_score:.3f}\n\n"

        prompt = f"""You are a cybersecurity expert. Please analyze whether the following atomic behavior has a mapping relationship with the given {label} entries.

Atomic Behavior: "{atomic_behavior}"
Full Vulnerability Context: "{full_vulnerability}"

{label} Candidates (with similarity scores):

{candidates_text}

Please analyze each {label} candidate and determine if it has a mapping relationship with the atomic behavior. Consider:
1. Semantic similarity between the atomic behavior and CWE description
2. Whether the CWE describes the same type of weakness/vulnerability
3. The similarity score (higher scores indicate better matches)

IMPORTANT: You can select AT MOST ONE candidate that best matches the atomic behavior. If no candidates are suitable, select none. If multiple candidates are suitable, choose the one with the highest similarity score.

For each candidate, respond with:
- "YES" if it is the BEST match for the atomic behavior (at most one should be YES)
- "NO" for all other candidates

Format your response as a JSON array with the same number of elements as candidates:
["YES", "NO", "NO", ...] or ["NO", "NO", "NO", ...] if no match

Return only the JSON array:"""

        try:
            response = self.call_llm_api(prompt)
            
            result = response.strip() if response else ""
            print(f"      Raw LLM response: {repr(result)}")
            
            if not result:
                print("        LLM returned empty response, returning best candidate based on similarity")
                best_candidate = max(candidates, key=lambda x: x.similarity_score)
                return [best_candidate]
            
            json_match = re.search(r'\[.*?\]', result, re.DOTALL)
            if json_match:
                try:
                    judgments = json.loads(json_match.group())
                    if isinstance(judgments, list) and len(judgments) == len(candidates):
                        mapped_candidates = []
                        for i, (candidate, judgment) in enumerate(zip(candidates, judgments)):
                            if judgment.upper() == "YES":
                                mapped_candidates.append(candidate)
                                if self.framework == "mitre":
                                    print(f"         LLM approved {candidate.cwe_id}: {candidate.title}")
                                elif self.framework == "cve":
                                    print(f"         LLM approved CVE-{candidate.cwe_id}: {candidate.title}")
                                else:
                                    print(f"         LLM approved CWE-{candidate.cwe_id}: {candidate.title}")
                            else:
                                if self.framework == "mitre":
                                    print(f"         LLM rejected {candidate.cwe_id}: {candidate.title}")
                                elif self.framework == "cve":
                                    print(f"         LLM rejected CVE-{candidate.cwe_id}: {candidate.title}")
                                else:
                                    print(f"         LLM rejected CWE-{candidate.cwe_id}: {candidate.title}")
                        
                        if len(mapped_candidates) > 1:
                            print(f"          LLM selected {len(mapped_candidates)} candidates, keeping only the one with highest similarity")
                            best_candidate = max(mapped_candidates, key=lambda x: x.similarity_score)
                            mapped_candidates = [best_candidate]
                        elif len(mapped_candidates) == 0:
                            print("          LLM rejected all candidates for this atomic behavior")
                            return []
                        
                        return mapped_candidates
                except json.JSONDecodeError as e:
                    print(f"          JSON decode error: {e}")
            
            print("          LLM response parsing failed, returning best candidate based on similarity")
            best_candidate = max(candidates, key=lambda x: x.similarity_score)
            return [best_candidate]
            
        except Exception as e:
            print(f"         Error calling LLM: {e}")
            best_candidate = max(candidates, key=lambda x: x.similarity_score)
            return [best_candidate]
    
    def _select_best_cwe_with_llm(self, cwe_candidates: List[RAGResult], vulnerability: str) -> RAGResult:

        if not cwe_candidates:
            return None
        
        prompt = f"""You are a cybersecurity expert. Select which CWE best matches the COMPLETE vulnerability description below.

Vulnerability Description: "{vulnerability}"

CWE Candidates to compare:

Respond as a JSON array with exactly {len(cwe_candidates)} elements in order where only the best candidate is "YES": ["YES", "NO", ...]. Return only the JSON array."""
        
        try:
            response = self.call_llm_api(prompt)
            if not response:
                return max(cwe_candidates, key=lambda x: x.similarity_score)
            
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                try:
                    judgments = json.loads(json_match.group())
                    if isinstance(judgments, list) and len(judgments) == len(cwe_candidates):
                        for cand, j in zip(cwe_candidates, judgments):
                            if isinstance(j, str) and j.strip().upper() == "YES":
                                return cand
                        return max(cwe_candidates, key=lambda x: x.similarity_score)
                except json.JSONDecodeError:
                    return max(cwe_candidates, key=lambda x: x.similarity_score)
            return max(cwe_candidates, key=lambda x: x.similarity_score)
        except Exception:
            return max(cwe_candidates, key=lambda x: x.similarity_score)
    
    def call_llm_api(self, prompt: str, system_prompt: str = None, 
                     max_tokens: int = 2000) -> Optional[str]:

        try:
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            response = get_single_prediction(self.llm_model, full_prompt)
            
            return response.strip() if response else None
            
        except Exception as e:
            print(f"  LLM API call failed: {e}")
            return None
    
    def should_skip_test_case(self, expected_answer: str) -> bool:

        if not self.empty_description_cwe_ids or not expected_answer:
            return False
        
        cwe_matches = re.findall(r'CWE-(\d+)', expected_answer, re.IGNORECASE)
        
        for cwe_id in cwe_matches:
            if cwe_id in self.empty_description_cwe_ids:
                print(f"    Skipping test case: answer contains CWE-{cwe_id} which has empty description")
                return True
        
        return False
    
    def closed_book_generation(self, question: str) -> BaselineResult:

        print(f"Running Closed-book Generation...")
        start_time = time.time()
        
        if self.framework == "mitre":
            prompt = f"""You are a cybersecurity expert. Based on the following security question, please provide:

1) The MITRE ATT&CK technique ID and name (e.g., T1059.001 - PowerShell)
2) An explanation of how the behavior maps to that technique
3) Detection recommendations aligned to x_mitre_detection (logs, sensors, NDR)

Question: {question}

Please format your response as:
1) T
2) [Detailed mapping explanation]
3) [Detection guidance]"""
        else:
            prompt = f"""You are a cybersecurity expert. Based on the following security question, please provide:

1) The CWE category that maps to this vulnerability
2) An explanation of how this vulnerability maps to the CWE  
3) What are the potential mitigations (code or configuration changes) to address this issue

Question: {question}

Please format your response as:
1) CWE-XXX: [CWE Name]
2) [Detailed explanation of the mapping]
3) [List of potential mitigations]"""

        response = self.call_llm_api(prompt)
        processing_time = time.time() - start_time
        
        if not response:
            return BaselineResult(
                approach="closed_book",
                cwe_prediction="CWE-Unknown",
                explanation="Failed to generate response",
                mitigations="No mitigations available",
                processing_time=processing_time
            )
        
        lines = response.split('\n')
        cwe_prediction = "CWE-Unknown"
        explanation = ""
        mitigations = ""
        
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith('1)'):
                current_section = "cwe"
                if self.framework == "mitre":
                    tech_match = re.search(r'T\d{4}(?:\.\d{3})?', line, re.IGNORECASE)
                    if tech_match:
                        cwe_prediction = tech_match.group().upper()
                else:
                    cwe_match = re.search(r'CWE-\d+', line)
                    if cwe_match:
                        cwe_prediction = cwe_match.group()
            elif line.startswith('2)'):
                current_section = "explanation"
                explanation = line[2:].strip()
            elif line.startswith('3)'):
                current_section = "mitigations"
                mitigations = line[2:].strip()
            elif current_section == "explanation" and line:
                explanation += " " + line
            elif current_section == "mitigations" and line:
                mitigations += " " + line
        
        if self.framework == "mitre" and (not cwe_prediction or cwe_prediction == "CWE-Unknown"):
            any_tech = re.search(r'T\d{4}(?:\.\d{3})?', response, re.IGNORECASE)
            if any_tech:
                cwe_prediction = any_tech.group().upper()
        
        return BaselineResult(
            approach="closed_book",
            cwe_prediction=cwe_prediction,
            explanation=explanation.strip(),
            mitigations=mitigations.strip(),
            processing_time=processing_time,
            closed_book_llm_answer=response
        )
    
    def vanilla_rag(self, question: str, top_k: int = 10, threshold: float = 0.5) -> BaselineResult:

        print(f"Running Vanilla RAG...")
        start_time = time.time()
        
        vulnerability_text = question
        quote_match = re.search(r'"([^"]+)"', question)
        if quote_match:
            vulnerability_text = quote_match.group(1)
        
        print(f"   Query text: {vulnerability_text}")
        
        query_embedding = self.get_query_embedding(vulnerability_text)
        if query_embedding is None:
            return BaselineResult(
                approach="vanilla_rag",
                cwe_prediction="CWE-Unknown",
                explanation="Failed to create query embedding",
                mitigations="No mitigations available",
                processing_time=time.time() - start_time
            )
        
        retrieval_results = self.search_similar(query_embedding, top_k, threshold)
        print(f"   Found {len(retrieval_results)} relevant entries (similarity > {threshold})")
        
        evidence = ""
        for i, result in enumerate(retrieval_results):
            id_label = result.cwe_id if self.framework == "mitre" else f"CWE-{result.cwe_id}"
            evidence += f"{id_label}: {result.title}\n"
            evidence += f"Description: {result.description[:500]}...\n"
            evidence += f"Similarity: {result.similarity_score:.3f}\n\n"
        
        if self.framework == "mitre":
            prompt = f"""You are a cybersecurity expert. Based on the following security question and retrieved ATT&CK evidence, please provide:

1) The MITRE ATT&CK technique ID and name (e.g., T1059.001 - PowerShell)
2) An explanation of how the behavior maps to that technique
3) Detection recommendations aligned to x_mitre_detection (logs, sensors, NDR)

Question: {question}

Retrieved ATT&CK Evidence:
{evidence}

Please format your response as:
1) T
2) [Detailed mapping explanation]
3) [Detection guidance]"""
        else:
            prompt = f"""You are a cybersecurity expert. Based on the following security question and retrieved CWE evidence, please provide:

1) The CWE category that maps to this vulnerability
2) An explanation of how this vulnerability maps to the CWE
3) What are the potential mitigations (code or configuration changes) to address this issue

Question: {question}

Retrieved CWE Evidence:
{evidence}

Please format your response as:
1) CWE-XXX: [CWE Name]
2) [Detailed explanation of the mapping]
3) [List of potential mitigations]"""

        response = self.call_llm_api(prompt)
        processing_time = time.time() - start_time
        
        if not response:
            return BaselineResult(
                approach="vanilla_rag",
                cwe_prediction="CWE-Unknown",
                explanation="Failed to generate response",
                mitigations="No mitigations available",
                retrieval_results=retrieval_results,
                processing_time=processing_time
            )
        
        lines = response.split('\n')
        cwe_prediction = "CWE-Unknown"
        explanation = ""
        mitigations = ""
        
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith('1)'):
                current_section = "cwe"
                if self.framework == "mitre":
                    tech_match = re.search(r'T\d{4}(?:\.\d{3})?', line, re.IGNORECASE)
                    if tech_match:
                        cwe_prediction = tech_match.group().upper()
                else:
                    cwe_match = re.search(r'CWE-\d+', line)
                    if cwe_match:
                        cwe_prediction = cwe_match.group()
            elif line.startswith('2)'):
                current_section = "explanation"
                explanation = line[2:].strip()
            elif line.startswith('3)'):
                current_section = "mitigations"
                mitigations = line[2:].strip()
            elif current_section == "explanation" and line:
                explanation += " " + line
            elif current_section == "mitigations" and line:
                mitigations += " " + line
        
        return BaselineResult(
            approach="vanilla_rag",
            cwe_prediction=cwe_prediction,
            explanation=explanation.strip(),
            mitigations=mitigations.strip(),
            retrieval_results=retrieval_results,
            processing_time=processing_time,
            vanilla_rag_llm_answer=response
        )
    
    def rag_with_query_expansion(self, question: str, top_k: int = 10, 
                                threshold: float = 0.5) -> BaselineResult:

        print(f" Running RAG with Query Expansion...")
        start_time = time.time()
        
        behavior_text = question
        quote_match = re.search(r'"([^"]+)"', question)
        if quote_match:
            behavior_text = quote_match.group(1)
        
        print(f"   Extracted behavior: {behavior_text}")
        
        atomic_behaviors = self._decompose_vulnerability(behavior_text)
        print(f"   Atomic behaviors: {atomic_behaviors}")
        
        all_selected_cwes = []
        
        for i, atomic_behavior in enumerate(atomic_behaviors):
            print(f"    Processing atomic behavior {i+1}: {atomic_behavior[:80]}...")
            
            behavior_embedding = self.get_query_embedding(atomic_behavior)
            if behavior_embedding is None:
                print(f"        Failed to create embedding for behavior: {atomic_behavior}")
                continue
            
            behavior_results = self.search_similar(behavior_embedding, top_k, threshold)
            print(f"       Found {len(behavior_results)} candidates for this atomic behavior")
            
            if not behavior_results:
                print(f"        No candidates found for atomic behavior {i+1}")
                continue
            
            enriched_candidates = self._enrich_candidates_with_corpus(behavior_results)
            
            selected_candidates = self._llm_judge_cwe_mapping(behavior_text, atomic_behavior, enriched_candidates)
            
            for result in selected_candidates:
                result.content = f"[Atomic Behavior: {atomic_behavior}] " + result.content
                result.atomic_behavior = atomic_behavior
                result.behavior_index = i + 1
            
            all_selected_cwes.extend(selected_candidates)
        
        print(f"   Total selected CWEs from all atomic behaviors: {len(all_selected_cwes)}")
        
        if all_selected_cwes:
            evidence = ""
            for i, result in enumerate(all_selected_cwes):
                id_label = result.cwe_id if self.framework == "mitre" else f"CWE-{result.cwe_id}"
                evidence += f"{id_label}: {result.title}\n"
                evidence += f"Description: {result.description[:500]}...\n"
                evidence += f"Similarity: {result.similarity_score:.3f}\n"
                if hasattr(result, 'atomic_behavior'):
                    evidence += f"Matched Atomic Behavior: {result.atomic_behavior}\n"
                evidence += "\n"
            
            if self.framework == "mitre":
                prompt = f"""You are a cybersecurity expert. Based on the following security question and the selected ATT&CK evidence from atomic behavior analysis, please provide a confident answer:

Question: {question}

Original Behavior: {behavior_text}
Decomposed Atomic Behaviors: {atomic_behaviors}

Selected ATT&CK Evidence (from atomic behavior matching):
{evidence}

Instructions:
- You can answer based on the provided ATT&CK evidence, or use your internal knowledge if you're confident
- Be confident in your response - if you're not sure, say so
- Provide the most accurate MITRE ATT&CK technique mapping

Please format your response as:
1) T
2) [Detailed explanation of the mapping and confidence level]
3) [Detection guidance aligned to x_mitre_detection]"""
            else:
                prompt = f"""You are a cybersecurity expert. Based on the following security question and the selected CWE evidence from atomic behavior analysis, please provide a confident answer:

Question: {question}

Original Behavior: {behavior_text}
Decomposed Atomic Behaviors: {atomic_behaviors}

Selected CWE Evidence (from atomic behavior matching):
{evidence}

Instructions:
- You can answer based on the provided CWE evidence, or use your internal knowledge if you're confident
- You don't have to select from the provided CWEs if you have a better answer
- Be confident in your response - if you're not sure, say so
- Provide the most accurate CWE mapping for this vulnerability

Please format your response as:
1) CWE-XXX: [CWE Name] (or "CWE-Unknown" if uncertain)
2) [Detailed explanation of the mapping and confidence level]
3) [List of potential mitigations]"""

        else:
            prompt = f"""You are a cybersecurity expert. Based on the following security question, please provide your best answer using your internal knowledge:

Question: {question}

Original Behavior: {behavior_text}
Decomposed Atomic Behaviors: {atomic_behaviors}

Note: No relevant CWE evidence was found through atomic behavior matching. Please use your internal knowledge to provide the best possible answer.

Please format your response as:
1) CWE-XXX: [CWE Name] (or "CWE-Unknown" if uncertain)
2) [Detailed explanation of the mapping and confidence level]
3) [List of potential mitigations]"""

        response = self.call_llm_api(prompt)
        processing_time = time.time() - start_time
        
        if not response:
            return BaselineResult(
                approach="rag_with_query_expansion",
                cwe_prediction="CWE-Unknown",
                explanation="Failed to generate response",
                mitigations="No mitigations available",
                retrieval_results=all_selected_cwes,
                decomposed_behaviors=atomic_behaviors,
                processing_time=processing_time
            )
        
        lines = response.split('\n')
        cwe_prediction = "CWE-Unknown"
        explanation = ""
        mitigations = ""
        
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith('1)'):
                current_section = "cwe"
                if self.framework == "mitre":
                    tech_match = re.search(r'T\d{4}(?:\.\d{3})?', line, re.IGNORECASE)
                    if tech_match:
                        cwe_prediction = tech_match.group().upper()
                else:
                    cwe_match = re.search(r'CWE-\d+', line)
                    if cwe_match:
                        cwe_prediction = cwe_match.group()
            elif line.startswith('2)'):
                current_section = "explanation"
                explanation = line[2:].strip()
            elif line.startswith('3)'):
                current_section = "mitigations"
                mitigations = line[2:].strip()
            elif current_section == "explanation" and line:
                explanation += " " + line
            elif current_section == "mitigations" and line:
                mitigations += " " + line
        
        if self.framework == "mitre" and (not cwe_prediction or cwe_prediction == "CWE-Unknown"):
            any_tech = re.search(r'T\d{4}(?:\.\d{3})?', response, re.IGNORECASE)
            if any_tech:
                cwe_prediction = any_tech.group().upper()
        
        return BaselineResult(
            approach="rag_with_query_expansion",
            cwe_prediction=cwe_prediction,
            explanation=explanation.strip(),
            mitigations=mitigations.strip(),
            retrieval_results=all_selected_cwes,
            decomposed_behaviors=atomic_behaviors,
            processing_time=processing_time,
            rag_expansion_llm_answer=response
        )

def main():

    parser = argparse.ArgumentParser(description="Run hybrid baseline approaches")
    parser.add_argument("--dataset", type=str, 
                       default="datasets/hybrid/vca/batch_0_20250905_090044",
                       help="Path to dataset directory")
    parser.add_argument("--corpus", type=str, default="corpus",
                       help="Path to corpus directory")
    parser.add_argument("--output", type=str, default="baseline_results.json",
                       help="Output file for results")
    parser.add_argument("--model", type=str, default="gpt_o3_mini",
                       help="LLM model short name (e.g., gpt_o3_mini, llama3-70b, qwen3-8b)")
    parser.add_argument("--k", type=int, default=10,
                       help="Top-k results for RAG")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Similarity threshold for RAG")
    parser.add_argument("--approaches", nargs="+", 
                       default=["closed_book", "vanilla_rag", "rag_expansion"],
                       choices=["closed_book", "vanilla_rag", "rag_expansion"],
                       help="Which approaches to run")
    parser.add_argument("--force-reload", action="store_true",
                       help="Force reload corpus and recompute embeddings (ignore cache)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from previous progress if available")
    parser.add_argument("--no-cleanup", action="store_true",
                       help="Don't clean up progress file after completion")
    parser.add_argument("--framework", type=str, default="cwe",
                       choices=["cwe", "mitre", "capec", "cve"],
                       help="Framework to use for corpus and answer generation (default: cwe)")
    
    args = parser.parse_args()
    
    print(f" Starting Hybrid Baseline Evaluation")
    print(f" Dataset: {args.dataset}")
    print(f"Corpus: {args.corpus}")
    print(f" Model: {args.model}")
    print(f" Approaches: {args.approaches}")
    print(f" Top-k: {args.k}, Threshold: {args.threshold}")
    print(f" Framework: {args.framework.upper()}")
    
    baselines = HybridBaselines(corpus_dir=args.corpus, llm_model=args.model, framework=args.framework)
    
    if not baselines.load_corpus(framework=args.framework, force_reload=args.force_reload):
        print(" Failed to load corpus")
        return
    
    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        print(f" Dataset directory not found: {dataset_dir}")
        return
    
    dataset_files = list(dataset_dir.glob("*.json"))
    if not dataset_files:
        print(f" No JSON files found in dataset directory")
        return
    
    print(f" Found {len(dataset_files)} test cases")
    
    all_results = []
    processed_files = set()
    start_index = 0
    
    if args.resume:
        progress_data = baselines._load_progress(args.dataset)
        if progress_data:
            if (progress_data.get("config", {}).get("approaches") == args.approaches and
                progress_data.get("config", {}).get("k") == args.k and
                progress_data.get("config", {}).get("threshold") == args.threshold and
                progress_data.get("config", {}).get("framework") == args.framework):
                
                all_results = progress_data.get("results", [])
                processed_files = set(progress_data.get("processed_files", []))
                
                for i, file_path in enumerate(dataset_files):
                    if file_path.name not in processed_files:
                        start_index = i
                        break
                else:
                    start_index = len(dataset_files)
                
                print(f"Resuming from file {start_index + 1}/{len(dataset_files)}")
                print(f" Already processed: {len(processed_files)} files")
            else:
                print("  Configuration mismatch, starting fresh")
                progress_data = None
        else:
            print(" No previous progress found, starting fresh")
    else:
        print("Starting fresh evaluation")
    
    try:
        for i in range(start_index, len(dataset_files)):
            file_path = dataset_files[i]
            print(f"\n{'='*60}")
            print(f" Test Case {i+1}/{len(dataset_files)}: {file_path.name}")
            print(f"{'='*60}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                test_case = json.load(f)
            
            question = test_case.get("question", "")
            expected_answer = test_case.get("answer", "")
            
            print(f"Question: {question[:100]}...")
            
            if baselines.should_skip_test_case(expected_answer):
                print(f"  Skipping test case {file_path.name} due to empty description CWE IDs in answer")
                case_results = {
                    "file": file_path.name,
                    "question": question,
                    "expected_answer": expected_answer,
                    "skipped": True,
                    "skip_reason": "Answer contains CWE IDs with empty descriptions",
                    "results": {}
                }
                all_results.append(case_results)
                processed_files.add(file_path.name)
                
                progress_data = {
                    "config": {
                        "approaches": args.approaches,
                        "k": args.k,
                        "threshold": args.threshold,
                        "corpus": args.corpus,
                        "framework": args.framework
                    },
                    "results": all_results,
                    "processed_files": list(processed_files),
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_files": len(dataset_files),
                    "current_file": i + 1
                }
                baselines._save_progress(args.dataset, progress_data)
                continue
            
            case_results = {
                "file": file_path.name,
                "question": question,
                "expected_answer": expected_answer,
                "results": {}
            }
            
            for approach in args.approaches:
                print(f"\nRunning {approach}...")
                
                try:
                    if approach == "closed_book":
                        result = baselines.closed_book_generation(question)
                    elif approach == "vanilla_rag":
                        result = baselines.vanilla_rag(question, args.k, args.threshold)
                    elif approach == "rag_expansion":
                        result = baselines.rag_with_query_expansion(question, args.k, args.threshold)
                    
                    result_dict = {
                        "approach": result.approach,
                        "cwe_prediction": result.cwe_prediction,
                        "explanation": result.explanation,
                        "mitigations": result.mitigations,
                        "processing_time": result.processing_time
                    }
                    
                    if result.retrieval_results:
                        if args.framework == "mitre":
                            result_dict["retrieval_results"] = [
                                {
                                    "technique_id": r.cwe_id,
                                    "title": r.title,
                                    "similarity_score": r.similarity_score
                                } for r in result.retrieval_results
                            ]
                        else:
                            result_dict["retrieval_results"] = [
                                {
                                    "cwe_id": r.cwe_id,
                                    "title": r.title,
                                    "similarity_score": r.similarity_score
                                } for r in result.retrieval_results
                            ]
                    
                    if result.decomposed_behaviors:
                        result_dict["decomposed_behaviors"] = result.decomposed_behaviors
                    if args.framework == "mitre":
                        result_dict["mitre_technique"] = result_dict.pop("cwe_prediction")
                    if result.closed_book_llm_answer:
                        result_dict["closed_book_llm_answer"] = result.closed_book_llm_answer
                    if result.vanilla_rag_llm_answer:
                        result_dict["vanilla_rag_llm_answer"] = result.vanilla_rag_llm_answer
                    if result.rag_expansion_llm_answer:
                        result_dict["rag_expansion_llm_answer"] = result.rag_expansion_llm_answer

                    case_results["results"][approach] = result_dict
                    
                    shown_pred = result_dict.get("mitre_technique") if args.framework == "mitre" else result.cwe_prediction
                    print(f" {approach}: {shown_pred} (took {result.processing_time:.2f}s)")

                except Exception as e:
                    print(f" Error in {approach}: {e}")
                    case_results["results"][approach] = {
                        "approach": approach,
                        "error": str(e),
                        "processing_time": 0
                    }
            
            all_results.append(case_results)
            processed_files.add(file_path.name)
            
            progress_data = {
                "config": {
                    "approaches": args.approaches,
                    "k": args.k,
                    "threshold": args.threshold,
                    "corpus": args.corpus,
                    "framework": args.framework
                },
                "results": all_results,
                "processed_files": list(processed_files),
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_files": len(dataset_files),
                "current_file": i + 1
            }
            baselines._save_progress(args.dataset, progress_data)
            
    except KeyboardInterrupt:
        print(f"\n  Interrupted by user. Progress saved.")
        print(f" Processed {len(processed_files)}/{len(dataset_files)} files")
        print(f"Use --resume to continue from where you left off")
        return
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        print(f" Processed {len(processed_files)}/{len(dataset_files)} files")
        print(f"Use --resume to continue from where you left off")
        return
    
    output_dir = baselines._get_output_dir(args.dataset)
    batch_id = baselines._get_batch_id(args.dataset)
    
    if args.output == "baseline_results.json":
        output_filename = f"{batch_id}_{args.model}_results.json"
    else:
        output_path = Path(args.output)
        if output_path.suffix:
            output_filename = f"{output_path.stem}_{args.model}{output_path.suffix}"
        else:
            output_filename = f"{args.output}_{args.model}"
    
    output_file = output_dir / output_filename
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluation complete! Results saved to {output_file}")
    
    if not args.no_cleanup:
        baselines._cleanup_progress(args.dataset)
    
    print(f"\n Summary:")
    skipped_count = sum(1 for case in all_results if case.get("skipped", False))
    processed_count = len(all_results) - skipped_count
    print(f"  Total test cases: {len(all_results)}")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped (empty description CWE IDs): {skipped_count}")
    
    for approach in args.approaches:
        processing_times = [
            case["results"][approach]["processing_time"] 
            for case in all_results 
            if approach in case["results"] and not case.get("skipped", False)
        ]
        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
        print(f"  {approach}: {len(processing_times)} cases, avg time: {avg_time:.2f}s")

if __name__ == "__main__":
    main()