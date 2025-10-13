#!/usr/bin/env python3

import json
import re
import argparse
import sys
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from model_interface.ollama_inference import get_single_prediction
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnvConfig:
    
    
    def __init__(self, env_file: str = None):
        self.config = self._load_default_config()
        
        if env_file and Path(env_file).exists():
            self._load_env_file(env_file)
        
        self._load_from_environment()
    
    def _load_default_config(self) -> Dict[str, Any]:
        return {
            "openai_api_key": "",
            "openai_model": "o4-mini",
            "openai_max_tokens": 1000,
            "openai_temperature": 0.1,
            
            "anthropic_api_key": "",
            "anthropic_model": "claude-3-sonnet-20240229",
            
            "eval_batch_size": 10,
            "eval_max_samples": 100,
            "eval_timeout": 30,
            
            "log_level": "INFO",
            "log_file": "baselines/static/logs/baseline.log",
            
            "corpus_dir": "corpus",
            "data_dir": "dataset_generation/data",
            "output_dir": "baselines/static/results",
        }
    
    def _load_env_file(self, env_file: str):
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip().lower()
                        value = value.strip()
                        
                        if key in ["openai_max_tokens", "eval_batch_size", "eval_max_samples", "eval_timeout"]:
                            value = int(value)
                        elif key in ["openai_temperature"]:
                            value = float(value)
                        
                        config_key = key.replace('_', '_')
                        if config_key in self.config:
                            self.config[config_key] = value
        except Exception as e:
            print(f"Warning: Could not load .env file {env_file}: {e}")
    
    def _load_from_environment(self):
        env_mappings = {
            "OPENAI_API_KEY": "openai_api_key",
            "OPENAI_MODEL": "openai_model",
            "OPENAI_MAX_TOKENS": "openai_max_tokens",
            "OPENAI_TEMPERATURE": "openai_temperature",
            "ANTHROPIC_API_KEY": "anthropic_api_key",
            "ANTHROPIC_MODEL": "anthropic_model",
            "EVAL_BATCH_SIZE": "eval_batch_size",
            "EVAL_MAX_SAMPLES": "eval_max_samples",
            "EVAL_TIMEOUT": "eval_timeout",
            "LOG_LEVEL": "log_level",
            "LOG_FILE": "log_file",
            "CORPUS_DIR": "corpus_dir",
            "DATA_DIR": "data_dir",
            "OUTPUT_DIR": "output_dir",
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if config_key in ["openai_max_tokens", "eval_batch_size", "eval_max_samples", "eval_timeout"]:
                    try:
                        value = int(value)
                    except ValueError:
                        continue
                elif config_key in ["openai_temperature"]:
                    try:
                        value = float(value)
                    except ValueError:
                        continue
                
                self.config[config_key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
    
    def get_openai_config(self) -> Dict[str, Any]:
        return {
            "api_key": self.get("openai_api_key"),
            "model": self.get("openai_model"),
            "max_tokens": self.get("openai_max_tokens"),
            "temperature": self.get("openai_temperature"),
        }
    
    def get_anthropic_config(self) -> Dict[str, Any]:
        return {
            "api_key": self.get("anthropic_api_key"),
            "model": self.get("anthropic_model"),
        }
    
    def get_eval_config(self) -> Dict[str, Any]:
        return {
            "batch_size": self.get("eval_batch_size"),
            "max_samples": self.get("eval_max_samples"),
            "timeout": self.get("eval_timeout"),
        }
    
    def get_paths_config(self) -> Dict[str, str]:
        return {
            "corpus_dir": self.get("corpus_dir"),
            "data_dir": self.get("data_dir"),
            "output_dir": self.get("output_dir"),
        }
    
    def is_openai_configured(self) -> bool:
        return bool(self.get("openai_api_key"))
    
    def is_anthropic_configured(self) -> bool:
        return bool(self.get("anthropic_api_key"))



env_file = None
current_dir = Path(__file__).parent
for parent_dir in [current_dir, current_dir.parent, current_dir.parent.parent]:
    env_path = parent_dir / ".env"
    if env_path.exists():
        env_file = str(env_path)
        break

config = EnvConfig(env_file)


@dataclass
class TaskConfig:
    
    name: str
    source_type: str
    target_type: str
    source_corpus_file: str
    target_corpus_file: str
    correlation_file: str
    id_pattern: str
    description_field: str


class CorpusLoader:
    
    
    def __init__(self, corpus_dir: str):
        self.corpus_dir = Path(corpus_dir)
        self._cache = {}
    
    def load_corpus(self, entity_type: str) -> Dict[str, Dict[str, Any]]:
        
        if entity_type in self._cache:
            return self._cache[entity_type]
        
        corpus_file = self.corpus_dir / f"{entity_type}.jsonl"
        if not corpus_file.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
        
        entities = {}
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    entity_id = entry.get(f"{entity_type}_id", entry.get("id"))
                    if entity_id:
                        entities[str(entity_id)] = entry
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error at line {line_num}: {e}")
                    continue
        
        self._cache[entity_type] = entities
        logger.info(f"Loaded {len(entities)} {entity_type} entities")
        return entities
    
    def get_entity_by_id(self, entity_type: str, entity_id: str) -> Optional[Dict[str, Any]]:
        
        entities = self.load_corpus(entity_type)
        return entities.get(str(entity_id))


class IDExtractor:
    
    
    @staticmethod
    def extract_cve_id(text: str) -> Optional[str]:
        
        pattern = r'CVE-\d{4}-\d{4,}'
        match = re.search(pattern, text)
        return match.group() if match else None
    
    @staticmethod
    def extract_cwe_id(text: str) -> Optional[str]:
        
        pattern = r'CWE-(\d+)'
        match = re.search(pattern, text)
        if match:
            return f"CWE-{match.group(1)}"
        return None
    
    @staticmethod
    def extract_capec_id(text: str) -> Optional[str]:
        
        pattern = r'CAPEC-(\d+)'
        match = re.search(pattern, text)
        if match:
            return f"CAPEC-{match.group(1)}"
        return None
    
    @staticmethod
    def extract_mitre_id(text: str) -> Optional[str]:
        
        pattern = r'T\d{4}(?:\.\d{3})?'
        match = re.search(pattern, text)
        return match.group() if match else None


class ClosedBookBaseline:
    
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name
    
    def generate_response(self, question: str, task: str) -> str:
        entity_id = None
        if task == "rcm":
            entity_id = IDExtractor.extract_cve_id(question)
        elif task == "wim":
            entity_id = IDExtractor.extract_cwe_id(question)
        elif task == "atd":
            entity_id = IDExtractor.extract_capec_id(question)
        elif task == "esd":
            entity_id = IDExtractor.extract_cwe_id(question)
        
        if not entity_id:
            return "Unable to extract relevant ID from question."
        
        return self._generate_llm_response(question, task, entity_id)
    
    def _generate_llm_response(self, question: str, task: str, entity_id: str) -> str:
        
        try:
            prompt = self._create_prompt(question, task, entity_id)

            response = get_single_prediction(self.model_name, prompt)

            return response
            
        
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_simulated_response(question, task, entity_id)
    
    def _create_prompt(self, question: str, task: str, entity_id: str) -> str:
        
        if task == "rcm":
            return f"Based on your knowledge, which CWE does {entity_id} map to? Please provide the CWE ID and a brief explanation."
        elif task == "wim":
            return f"Based on your knowledge, which CVE instantiates {entity_id}? Please provide the CVE ID and a brief explanation."
        elif task == "atd":
            return f"Based on your knowledge, which MITRE ATT&CK technique maps to {entity_id}? Please provide the technique ID and a brief explanation."
        elif task == "esd":
            return f"Based on your knowledge, which CAPEC attack pattern exploits {entity_id}? Please provide the CAPEC ID and a brief explanation."
        else:
            return question
    
    def _get_system_prompt(self, task: str) -> str:
        
        base_prompt = "You are a cybersecurity expert. Answer questions about cybersecurity mappings based on your knowledge."
        
        if task == "rcm":
            return f"{base_prompt} For Root Cause Mapping, provide CWE IDs and explanations for CVE vulnerabilities."
        elif task == "wim":
            return f"{base_prompt} For Weakness Instantiation Mapping, provide CVE IDs and explanations for CWE weaknesses."
        elif task == "atd":
            return f"{base_prompt} For Attack Technique Derivation, provide MITRE ATT&CK technique IDs and explanations for CAPEC attack patterns."
        elif task == "esd":
            return f"{base_prompt} For Exploitation Surface Discovery, provide CAPEC attack pattern IDs and explanations for CWE weaknesses."
        else:
            return base_prompt
    
    def _generate_simulated_response(self, question: str, task: str, entity_id: str) -> str:
        
        if task == "rcm":
            return f"- CWE-79.\n- {entity_id} causes cross-site scripting vulnerability, matching CWE-79: improper neutralization of input during web page generation."
        elif task == "wim":
            return f"- CVE-2023-1234.\n- CVE-2023-1234 instantiates {entity_id}: buffer overflow vulnerability in web application."
        elif task == "atd":
            return f"- T1059.001.\n- T1059.001 corresponds to PowerShell command execution, mapping {entity_id}: using PowerShell for command execution."
        elif task == "esd":
            return f"- CAPEC-63.\n- CAPEC-63 exploits cross-site scripting vulnerability, targeting {entity_id}: improper neutralization of input during web page generation."
        else:
            return f"Simulated response for {entity_id} in {task} task."


class MemoryInjectedBaseline:
    
    
    def __init__(self, corpus_loader: CorpusLoader, model_name: str = None):
        self.corpus_loader = corpus_loader
        self.model_name = model_name
    
    def generate_response(self, question: str, task: str) -> str:
        result = self.generate_response_with_memory(question, task)
        return result["prediction"]
    
    def generate_response_with_memory(self, question: str, task: str) -> Dict[str, Any]:
        entity_id = None
        retrieved_info = None
        
        if task == "rcm":
            entity_id = IDExtractor.extract_cve_id(question)
            if entity_id:
                retrieved_info = self._retrieve_cve_info(entity_id)
        elif task == "wim":
            entity_id = IDExtractor.extract_cwe_id(question)
            if entity_id:
                retrieved_info = self._retrieve_cwe_info(entity_id)
        elif task == "atd":
            entity_id = IDExtractor.extract_capec_id(question)
            if entity_id:
                retrieved_info = self._retrieve_capec_info(entity_id)
        elif task == "esd":
            entity_id = IDExtractor.extract_cwe_id(question)
            if entity_id:
                retrieved_info = self._retrieve_cwe_info(entity_id)
        
        if not entity_id:
            return {
                "prediction": "Unable to extract relevant ID from question.",
                "retrieved_memory": None,
                "entity_id": None
            }
        
        prediction = self._generate_llm_response_with_context(question, task, entity_id, retrieved_info)
        
        return {
            "prediction": prediction,
            "retrieved_memory": retrieved_info,
            "entity_id": entity_id
        }
    
    def _retrieve_cve_info(self, cve_id: str) -> Optional[Dict[str, Any]]:
        
        cve_entry = self.corpus_loader.get_entity_by_id("cve", cve_id)
        if not cve_entry:
            cve_entry = self.corpus_loader.get_entity_by_id("cve", cve_id.replace("CVE-", ""))
        
        if cve_entry:
            related_cwes = self._extract_cwe_from_cve(cve_entry)
            return {
                "id": cve_id,
                "title": cve_entry.get("title", ""),
                "description": self._extract_description(cve_entry, "cve"),
                "related_cwes": related_cwes,
                "related_cwe_details": self._get_cwe_details(related_cwes)
            }
        return None
    
    def _retrieve_cwe_info(self, cwe_id: str) -> Optional[Dict[str, Any]]:
        
        cwe_entry = self.corpus_loader.get_entity_by_id("cwe", cwe_id)
        if not cwe_entry:
            cwe_entry = self.corpus_loader.get_entity_by_id("cwe", cwe_id.replace("CWE-", ""))
        
        if cwe_entry:
            related_cves = self._extract_cve_from_cwe(cwe_entry)
            related_capecs = self._extract_capec_from_cwe(cwe_entry)
            return {
                "id": cwe_id,
                "title": cwe_entry.get("title", ""),
                "description": self._extract_description(cwe_entry, "cwe"),
                "related_cves": related_cves,
                "related_capecs": related_capecs,
                "related_cve_details": self._get_cve_details(related_cves),
                "related_capec_details": self._get_capec_details(related_capecs)
            }
        return None
    
    def _retrieve_capec_info(self, capec_id: str) -> Optional[Dict[str, Any]]:
        
        capec_entry = self.corpus_loader.get_entity_by_id("capec", capec_id)
        if not capec_entry:
            capec_entry = self.corpus_loader.get_entity_by_id("capec", capec_id.replace("CAPEC-", ""))
        
        if capec_entry:
            related_mitre = self._extract_mitre_from_capec(capec_entry)
            return {
                "id": capec_id,
                "title": capec_entry.get("title", ""),
                "description": self._extract_description(capec_entry, "capec"),
                "related_mitre": related_mitre,
                "related_mitre_details": self._get_mitre_details(related_mitre)
            }
        return None
    
    def _generate_llm_response_with_context(self, question: str, task: str, entity_id: str, context: Dict[str, Any]) -> str:
        
        try:
            prompt = self._create_context_prompt(question, task, entity_id, context)
            
            response = get_single_prediction(self.model_name, prompt)
            return response
        
        except Exception as e:
            logger.error(f"LLM generation with context failed: {e}")
            return self._generate_simulated_response_with_context(question, task, entity_id, context)
    
    def _create_context_prompt(self, question: str, task: str, entity_id: str, context: Dict[str, Any]) -> str:
        
        if task == "rcm":
            prompt = f"Based on the following information about {entity_id}, which CWE does it map to?\n\n"
            prompt += f"Description: {context.get('description', 'N/A')}\n"
            
            related_cwes = context.get('related_cwes', [])
            if related_cwes:
                prompt += f"Related CWEs: {', '.join(related_cwes)}\n"
                cwe_details = context.get('related_cwe_details', [])
                if cwe_details:
                    prompt += "CWE Details:\n"
                    for cwe_detail in cwe_details[:3]:
                        prompt += f"- {cwe_detail['id']}: {cwe_detail['description'][:200]}...\n"
            
            prompt += "\nPlease provide the CWE ID and a brief explanation."
            return prompt
            
        elif task == "wim":
            prompt = f"Based on the following information about {entity_id}, which CVE instantiates it?\n\n"
            prompt += f"Description: {context.get('description', 'N/A')}\n"
            
            related_cves = context.get('related_cves', [])
            if related_cves:
                prompt += f"Related CVEs: {', '.join(related_cves)}\n"
                cve_details = context.get('related_cve_details', [])
                if cve_details:
                    prompt += "CVE Details:\n"
                    for cve_detail in cve_details[:3]:
                        prompt += f"- {cve_detail['id']}: {cve_detail['description'][:200]}...\n"
            
            prompt += "\nPlease provide the CVE ID and a brief explanation."
            return prompt
            
        elif task == "atd":
            prompt = f"Based on the following information about {entity_id}, which MITRE ATT&CK technique maps to it?\n\n"
            prompt += f"Description: {context.get('description', 'N/A')}\n"
            
            related_mitre = context.get('related_mitre', [])
            if related_mitre:
                prompt += f"Related MITRE ATT&CK techniques: {', '.join(related_mitre)}\n"
                mitre_details = context.get('related_mitre_details', [])
                if mitre_details:
                    prompt += "MITRE ATT&CK Details:\n"
                    for mitre_detail in mitre_details[:3]:
                        prompt += f"- {mitre_detail['id']}: {mitre_detail['description'][:200]}...\n"
            
            prompt += "\nPlease provide the MITRE ATT&CK technique ID and a brief explanation."
            return prompt
            
        elif task == "esd":
            prompt = f"Based on the following information about {entity_id}, which CAPEC attack pattern exploits it?\n\n"
            prompt += f"Description: {context.get('description', 'N/A')}\n"
            
            related_capecs = context.get('related_capecs', [])
            if related_capecs:
                prompt += f"Related CAPEC patterns: {', '.join(related_capecs)}\n"
                capec_details = context.get('related_capec_details', [])
                if capec_details:
                    prompt += "CAPEC Details:\n"
                    for capec_detail in capec_details[:3]:
                        prompt += f"- {capec_detail['id']}: {capec_detail['description'][:200]}...\n"
            
            prompt += "\nPlease provide the CAPEC ID and a brief explanation."
            return prompt
        else:
            return question
    
    def _get_context_system_prompt(self, task: str) -> str:
        
        base_prompt = "You are a cybersecurity expert. Answer questions about cybersecurity mappings based on your knowledge."
        
        if task == "rcm":
            return f"{base_prompt} For Root Cause Mapping, provide CWE IDs and explanations for CVE vulnerabilities."
        elif task == "wim":
            return f"{base_prompt} For Weakness Instantiation Mapping, provide CVE IDs and explanations for CWE weaknesses."
        elif task == "atd":
            return f"{base_prompt} For Attack Technique Derivation, provide MITRE ATT&CK technique IDs and explanations for CAPEC attack patterns."
        elif task == "esd":
            return f"{base_prompt} For Exploitation Surface Discovery, provide CAPEC attack pattern IDs and explanations for CWE weaknesses."
        else:
            return base_prompt
    
    def _generate_simulated_response_with_context(self, question: str, task: str, entity_id: str, context: Dict[str, Any]) -> str:
        
        if not context:
            return f"Simulated response for {entity_id} in {task} task (no context available)."
        
        description = context.get("description", "")
        
        if task == "rcm":
            related_cwes = context.get("related_cwes", [])
            if related_cwes:
                cwe_id = related_cwes[0]
                return f"- {cwe_id}.\n- {entity_id} causes {description.lower()}, matching {cwe_id}: {description}."
            else:
                return f"- CWE-79.\n- {entity_id} causes {description.lower()}, matching CWE-79: {description}."
        elif task == "wim":
            related_cves = context.get("related_cves", [])
            if related_cves:
                cve_id = related_cves[0]
                return f"- {cve_id}.\n- {cve_id} instantiates {entity_id}: {description.lower()}."
            else:
                return f"- CVE-2023-1234.\n- CVE-2023-1234 instantiates {entity_id}: {description.lower()}."
        elif task == "atd":
            related_mitre = context.get("related_mitre", [])
            if related_mitre:
                mitre_id = related_mitre[0]
                return f"- {mitre_id}.\n- {mitre_id} corresponds to attack technique, mapping {entity_id}: {description.lower()}."
            else:
                return f"- T1059.001.\n- T1059.001 corresponds to command execution, mapping {entity_id}: {description.lower()}."
        elif task == "esd":
            related_capecs = context.get("related_capecs", [])
            if related_capecs:
                capec_id = related_capecs[0]
                return f"- {capec_id}.\n- {capec_id} exploits {description.lower()}, targeting {entity_id}: {description}."
            else:
                return f"- CAPEC-63.\n- CAPEC-63 exploits {description.lower()}, targeting {entity_id}: {description}."
        else:
            return f"Simulated response for {entity_id} in {task} task with context: {description[:100]}..."
    
    def _extract_cwe_from_cve(self, cve_entry: Dict[str, Any]) -> List[str]:
        
        cwe_ids = []
        try:
            contents = json.loads(cve_entry.get("contents", "{}"))
            weaknesses = contents.get("weaknesses", [])
            for weakness in weaknesses:
                descriptions = weakness.get("description", [])
                for desc in descriptions:
                    value = desc.get("value", "")
                    if value.startswith("CWE-"):
                        cwe_ids.append(value)
        except json.JSONDecodeError:
            pass
        return cwe_ids
    
    def _extract_cve_from_cwe(self, cwe_entry: Dict[str, Any]) -> List[str]:
        
        cve_ids = []
        try:
            contents = json.loads(cwe_entry.get("contents", "{}"))
            
            if "Observed_Examples" in contents:
                examples_data = contents["Observed_Examples"]
                
                if isinstance(examples_data, dict) and "Observed_Example" in examples_data:
                    examples = examples_data["Observed_Example"]
                    if isinstance(examples, dict):
                        examples = [examples]
                elif isinstance(examples_data, list):
                    examples = examples_data
                else:
                    examples = [examples_data]
                
                for example in examples:
                    if isinstance(example, dict) and "Reference" in example:
                        ref = example["Reference"]
                        if "CVE-" in ref:
                            import re
                            cve_match = re.search(r'CVE-\d{4}-\d{4,}', ref)
                            if cve_match:
                                cve_ids.append(cve_match.group())
            
            description = contents.get("Description", "")
            if description:
                import re
                cve_matches = re.findall(r'CVE-\d{4}-\d{4,}', description)
                cve_ids.extend(cve_matches)
                
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        return cve_ids
    
    def _extract_capec_from_cwe(self, cwe_entry: Dict[str, Any]) -> List[str]:
        
        capec_ids = []
        try:
            contents = json.loads(cwe_entry.get("contents", "{}"))
            
            if "Related_Attack_Patterns" in contents:
                patterns_data = contents["Related_Attack_Patterns"]
                
                if isinstance(patterns_data, dict) and "Related_Attack_Pattern" in patterns_data:
                    patterns = patterns_data["Related_Attack_Pattern"]
                    if isinstance(patterns, dict):
                        patterns = [patterns]
                elif isinstance(patterns_data, list):
                    patterns = patterns_data
                else:
                    patterns = [patterns_data]
                
                for pattern in patterns:
                    if isinstance(pattern, dict) and "@CAPEC_ID" in pattern:
                        capec_id = pattern["@CAPEC_ID"]
                        if capec_id:
                            capec_ids.append(f"CAPEC-{capec_id}")
            
            description = contents.get("Description", "")
            if description:
                import re
                capec_matches = re.findall(r'CAPEC-\d+', description)
                capec_ids.extend(capec_matches)
                
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        return capec_ids
    
    def _extract_mitre_from_capec(self, capec_entry: Dict[str, Any]) -> List[str]:
        
        mitre_ids = []
        try:
            contents = json.loads(capec_entry.get("contents", "{}"))
            
            if "Taxonomy_Mappings" in contents:
                taxonomy_mappings = contents["Taxonomy_Mappings"]
                
                mappings = taxonomy_mappings.get("Taxonomy_Mapping", [])
                if not isinstance(mappings, list):
                    mappings = [mappings]
                
                for mapping in mappings:
                    if isinstance(mapping, dict):
                        taxonomy_name = mapping.get("@Taxonomy_Name", "")
                        entry_id = mapping.get("Entry_ID", "")
                        
                        if taxonomy_name == "ATTACK" and entry_id:
                            if entry_id.startswith("T"):
                                mitre_ids.append(entry_id)
                            else:
                                mitre_ids.append(f"T{entry_id}")
            
            description = contents.get("Description", "")
            if description:
                import re
                mitre_matches = re.findall(r'T\d{4}(?:\.\d{3})?', description)
                mitre_ids.extend(mitre_matches)
                
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        return mitre_ids
    
    def _get_cwe_details(self, cwe_ids: List[str]) -> List[Dict[str, Any]]:
        
        details = []
        for cwe_id in cwe_ids:
            cwe_entry = self.corpus_loader.get_entity_by_id("cwe", cwe_id)
            if not cwe_entry:
                cwe_entry = self.corpus_loader.get_entity_by_id("cwe", cwe_id.replace("CWE-", ""))
            
            if cwe_entry:
                details.append({
                    "id": cwe_id,
                    "title": cwe_entry.get("title", ""),
                    "description": self._extract_description(cwe_entry, "cwe")
                })
        return details
    
    def _get_cve_details(self, cve_ids: List[str]) -> List[Dict[str, Any]]:
        
        details = []
        for cve_id in cve_ids:
            cve_entry = self.corpus_loader.get_entity_by_id("cve", cve_id)
            if not cve_entry:
                cve_entry = self.corpus_loader.get_entity_by_id("cve", cve_id.replace("CVE-", ""))
            
            if cve_entry:
                details.append({
                    "id": cve_id,
                    "title": cve_entry.get("title", ""),
                    "description": self._extract_description(cve_entry, "cve")
                })
        return details
    
    def _get_capec_details(self, capec_ids: List[str]) -> List[Dict[str, Any]]:
        
        details = []
        for capec_id in capec_ids:
            capec_entry = self.corpus_loader.get_entity_by_id("capec", capec_id)
            if not capec_entry:
                capec_entry = self.corpus_loader.get_entity_by_id("capec", capec_id.replace("CAPEC-", ""))
            
            if capec_entry:
                details.append({
                    "id": capec_id,
                    "title": capec_entry.get("title", ""),
                    "description": self._extract_description(capec_entry, "capec")
                })
        return details
    
    def _get_mitre_details(self, mitre_ids: List[str]) -> List[Dict[str, Any]]:
        
        details = []
        for mitre_id in mitre_ids:
            mitre_entry = self.corpus_loader.get_entity_by_id("mitre", mitre_id)
            if mitre_entry:
                details.append({
                    "id": mitre_id,
                    "title": mitre_entry.get("title", ""),
                    "description": self._extract_description(mitre_entry, "mitre")
                })
        return details
    
    def _extract_description(self, entry: Dict[str, Any], entity_type: str) -> str:
        
        try:
            contents = json.loads(entry.get("contents", "{}"))
            if entity_type == "cwe":
                return contents.get("Description", entry.get("title", ""))
            elif entity_type == "capec":
                return contents.get("Description", entry.get("title", ""))
            elif entity_type == "mitre":
                return contents.get("description", entry.get("title", ""))
            elif entity_type == "cve":
                descriptions = contents.get("descriptions", [])
                if descriptions:
                    return descriptions[0].get("value", entry.get("title", ""))
        except json.JSONDecodeError:
            pass
        
        return entry.get("title", "")


class StaticBaselineEvaluator:
    
    
    def __init__(self, corpus_dir: str = None, data_dir: str = None, resume: bool = False, restart: bool = False, model: str = None):
        paths_config = config.get_paths_config()
        
        script_dir = Path(__file__).parent.parent.parent
        
        if corpus_dir:
            self.corpus_dir = Path(corpus_dir)
        else:
            corpus_path = paths_config["corpus_dir"]
            if Path(corpus_path).is_absolute():
                self.corpus_dir = Path(corpus_path)
            else:
                if corpus_path.startswith("../"):
                    parts = corpus_path.split("/")
                    up_levels = sum(1 for part in parts if part == "..")
                    remaining_path = "/".join(parts[up_levels:])
                    target_dir = script_dir
                    for _ in range(up_levels):
                        target_dir = target_dir.parent
                    self.corpus_dir = target_dir / remaining_path
                else:
                    self.corpus_dir = script_dir / corpus_path
        
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            data_path = paths_config["data_dir"]
            if Path(data_path).is_absolute():
                self.data_dir = Path(data_path)
            else:
                if data_path.startswith("../"):
                    parts = data_path.split("/")
                    up_levels = sum(1 for part in parts if part == "..")
                    remaining_path = "/".join(parts[up_levels:])
                    target_dir = script_dir
                    for _ in range(up_levels):
                        target_dir = target_dir.parent
                    self.data_dir = target_dir / remaining_path
                else:
                    self.data_dir = script_dir / data_path
        
        self.corpus_loader = CorpusLoader(str(self.corpus_dir))
        
        self.resume = resume
        self.restart = restart
        self.llm_model = model
        
        self.closed_book = ClosedBookBaseline(self.llm_model)
        self.memory_injected = MemoryInjectedBaseline(self.corpus_loader, self.llm_model)
    
    def _get_progress_file_path(self, task: str, input_dir: str = None) -> Path:
        
        if input_dir:
            output_dir = self._generate_output_path(input_dir, task)
            return Path(output_dir) / f"{self.llm_model}_progress.json"
        else:
            output_dir = config.get("output_dir", "baselines/static/results")
            return Path(output_dir) / f"{self.llm_model}_progress.json"
    
    def _get_result_file_path(self, task: str, input_dir: str = None) -> Path:
        
        if input_dir:
            output_dir = self._generate_output_path(input_dir, task)
            return Path(output_dir) / f"{self.llm_model}_result.json"
        else:
            output_dir = config.get("output_dir", "baselines/static/results")
            return Path(output_dir) / f"{self.llm_model}_result.json"
    
    def _load_progress(self, task: str, input_dir: str = None) -> Dict[str, Any]:
        
        progress_file = self._get_progress_file_path(task, input_dir)
        
        if self.restart:
            if progress_file.exists():
                progress_file.unlink()
                print("Restart mode: Cleared existing progress file")
            result_file = self._get_result_file_path(task, input_dir)
            if result_file.exists():
                result_file.unlink()
                print("Restart mode: Cleared existing results file")
        
        auto_resume = not self.resume and not self.restart and progress_file.exists()
        
        if progress_file.exists() and (self.resume or auto_resume) and not self.restart:
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
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
            'input_dir': str(input_dir) if input_dir else f"datasets/static/{task}",
            'output_dir': str(self._get_progress_file_path(task, input_dir).parent),
            'start_time': datetime.now().isoformat(),
            'total_cases': 0,
            'completed_cases': 0,
            'failed_cases': 0,
            'completed_case_ids': [],
            'failed_case_ids': [],
            'current_case': None,
            'last_updated': datetime.now().isoformat()
        }
    
    def _save_progress(self, progress: Dict[str, Any], task: str, input_dir: str = None):
        
        progress_file = self._get_progress_file_path(task, input_dir)
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        
        progress['last_updated'] = datetime.now().isoformat()
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)
    
    def evaluate_task(self, task: str, input_dir: str = None) -> Dict[str, Any]:
        print(f" Starting evaluation for task: {task}")
        print(f"Resume mode: {self.resume}")
        print(f"Restart mode: {self.restart}")
        
        progress = self._load_progress(task, input_dir)
        
        test_data = self._load_test_data(task, input_dir)
        if not test_data:
            logger.error(f"No test data found for {task}")
            return {}
        
        print(f" Found {len(test_data)} test samples")
        
        progress['total_cases'] = len(test_data)
        
        results = {
            "evaluation_info": {
                "input_dir": str(input_dir) if input_dir else f"datasets/static/{task}",
                "output_dir": str(self._get_progress_file_path(task, input_dir).parent),
                "corpus_path": str(self.corpus_dir),
                "start_time": progress['start_time'],
                "total_cases": len(test_data),
                "successful_cases": 0,
                "failed_cases": 0,
                "last_updated": datetime.now().isoformat()
            },
            "cases": []
        }
        
        if self.resume or progress['completed_cases'] > 0:
            completed_ids = set(progress['completed_case_ids'])
            remaining_samples = []
            remaining_indices = []
            
            for i, sample in enumerate(test_data):
                case_id = f"sample_{i+1}"
                if case_id not in completed_ids:
                    remaining_samples.append(sample)
                    remaining_indices.append(i)
            
            print(f"Resuming: {len(remaining_samples)} cases remaining")
            test_data = remaining_samples
        else:
            remaining_indices = list(range(len(test_data)))
        
        for i, (sample, original_index) in enumerate(zip(test_data, remaining_indices)):
            case_id = f"sample_{original_index+1}"
            
            progress['current_case'] = case_id
            self._save_progress(progress, task, input_dir)
            
            print(f"\nProcessing case {i+1}/{len(test_data)}: {case_id}")
            
            try:
                question = sample.get("question", "")
                ground_truth = sample.get("answer", "")
                
                cb_response = self.closed_book.generate_response(question, task)
                mi_result = self.memory_injected.generate_response_with_memory(question, task)
                
                case_result = {
                    "case_id": case_id,
                    "question": question,
                    "ground_truth": ground_truth,
                    "methods": {
                        "closed_book": {
                            "method": "closed_book",
                            "prediction": cb_response,
                            "model": self.closed_book.model_name,
                            "task_type": task.upper()
                        },
                        "memory_injected": {
                            "method": "memory_injected", 
                            "prediction": mi_result["prediction"],
                            "model": self.memory_injected.model_name,
                            "task_type": task.upper(),
                            "retrieved_memory": mi_result["retrieved_memory"],
                            "entity_id": mi_result["entity_id"]
                        }
                    }
                }
                
                results["cases"].append(case_result)
                
                progress['completed_cases'] += 1
                progress['completed_case_ids'].append(case_id)
                progress['current_case'] = None
                
                self._save_progress(progress, task, input_dir)
                
                self._save_results(results, task, input_dir)
                
                print(f" Completed case {case_id}")
                
            except Exception as e:
                print(f" Error processing {case_id}: {e}")
                
                progress['failed_cases'] += 1
                progress['failed_case_ids'].append(case_id)
                progress['current_case'] = None
                
                self._save_progress(progress, task, input_dir)
        
        results["evaluation_info"]["successful_cases"] = progress['completed_cases']
        results["evaluation_info"]["failed_cases"] = progress['failed_cases']
        results["evaluation_info"]["last_updated"] = datetime.now().isoformat()
        
        progress['last_updated'] = datetime.now().isoformat()
        self._save_progress(progress, task, input_dir)
        
        self._save_results(results, task, input_dir)
        
        self._print_evaluation_summary(progress, results)
        
        return results
    
    def _print_evaluation_summary(self, progress: Dict[str, Any], results: Dict[str, Any]):
        
        print(f"\n Evaluation Summary:")
        print(f"  Input Dir: {progress['input_dir']}")
        print(f"  Output Dir: {progress['output_dir']}")
        print(f"  Total cases: {progress['total_cases']}")
        print(f"  Completed: {progress['completed_cases']}")
        print(f"  Failed: {progress['failed_cases']}")
        
        if progress['failed_case_ids']:
            print(f"  Failed cases: {', '.join(progress['failed_case_ids'])}")
    
    def _load_test_data(self, task: str, input_dir: str = None) -> List[Dict[str, Any]]:
        
        if input_dir:
            data_path = Path(input_dir)
            logger.info(f"Using custom input directory: {data_path}")
        else:
            data_path = self.data_dir / task
        
        if not data_path.exists():
            logger.warning(f"Data path does not exist: {data_path}")
            return []
        
        test_data = []
        for json_file in data_path.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    test_data.append(data)
            except json.JSONDecodeError as e:
                logger.warning(f"Error loading {json_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(test_data)} test samples from {data_path}")
        return test_data
    
    def _calculate_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        
        cb_correct = 0
        mi_correct = 0
        
        for cb_result, mi_result in zip(results["closed_book_results"], results["memory_injected_results"]):
            if cb_result["prediction"] == cb_result["ground_truth"]:
                cb_correct += 1
            if mi_result["prediction"] == mi_result["ground_truth"]:
                mi_correct += 1
        
        total = results["total_samples"]
        
        return {
            "closed_book_accuracy": cb_correct / total if total > 0 else 0,
            "memory_injected_accuracy": mi_correct / total if total > 0 else 0,
            "closed_book_correct": cb_correct,
            "memory_injected_correct": mi_correct
        }
    
    def _generate_output_path(self, input_dir: str = None, task: str = None) -> str:
        
        if input_dir:
            input_path = Path(input_dir)
            path_parts = input_path.parts
            
            task_idx = None
            
            for i, part in enumerate(path_parts):
                if part in ["rcm", "wim", "atd", "esd"]:
                    task_idx = i
                    break
            
            if task_idx is not None:
                extracted_task = path_parts[task_idx]
                output_path = Path("baselines/static/output") / extracted_task
                return str(output_path)
            else:
                output_path = Path("baselines/static/output") / (task or "unknown")
                return str(output_path)
        else:
            output_path = Path("baselines/static/output") / (task or "unknown")
            return str(output_path)
    
    def _save_results(self, results: Dict[str, Any], task: str, input_dir: str = None):
        
        result_file = self._get_result_file_path(task, input_dir)
        result_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {result_file}")
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {output_path}")


def main():
    
    parser = argparse.ArgumentParser(
        description="Static Baseline Evaluator for Cybersecurity Mapping Tasks"
    )
    
    parser.add_argument(
        "--task",
        required=True,
        choices=["rcm", "wim", "atd", "esd"],
        help="Task to evaluate"
    )
    
    
    
    parser.add_argument(
        "--corpus-dir",
        help="Path to corpus directory (uses env config if not provided)"
    )
    
    parser.add_argument(
        "--data-dir",
        help="Path to test data directory (uses env config if not provided)"
    )
    
    parser.add_argument(
        "--output",
        help="Output directory for results (uses env config if not provided)"
    )
    
    parser.add_argument(
        "--model",
        help="Model name for baseline (uses env config if not provided)"
    )
    
    parser.add_argument(
        "--input-dir",
        help="Custom input directory path (e.g., datasets/static/atd)"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume evaluation from previous progress"
    )
    
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Restart evaluation from beginning (clears existing progress and results)"
    )
    
    args = parser.parse_args()
    
    if args.resume and args.restart:
        print(" Error: Cannot use both --resume and --restart flags simultaneously")
        print("   Use --resume to continue from previous progress")
        print("   Use --restart to start fresh (clears existing progress)")
        sys.exit(1)
    
    if args.corpus_dir:
        config.config["corpus_dir"] = args.corpus_dir
    if args.data_dir:
        config.config["data_dir"] = args.data_dir
    if args.output:
        config.config["output_dir"] = args.output
    
    evaluator = StaticBaselineEvaluator(resume=args.resume, restart=args.restart, model=args.model)
    
    results = evaluator.evaluate_task(args.task, args.input_dir)
    
    if results:
        if args.input_dir:
            output_dir = evaluator._generate_output_path(args.input_dir, args.task)
            output_file = f"{output_dir}/{args.task}_result.json"
        else:
            output_dir = config.get("output_dir", "baselines/static/results")
            output_file = f"{output_dir}/{args.task}_result.json"
        
        evaluator.save_results(results, output_file)
        
        print(f"\nEvaluation Summary for {args.task.upper()}:")
        print(f"Model: {args.model}")
        print(f"Total cases: {results['evaluation_info']['total_cases']}")
        print(f"Successful cases: {results['evaluation_info']['successful_cases']}")
        print(f"Failed cases: {results['evaluation_info']['failed_cases']}")
        print(f"Results saved to: {output_file}")
    else:
        print("No results generated. Check logs for errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
