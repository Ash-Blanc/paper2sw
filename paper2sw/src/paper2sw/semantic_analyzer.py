from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set
from collections import defaultdict

from .logging_config import get_logger


@dataclass
class ModelArchitecture:
    """Represents the extracted architecture information from a paper."""
    model_family: str
    num_layers: int | None
    hidden_size: int | None
    mlp_expansion: int | None
    attention_heads: int | None
    key_components: List[str]
    mentioned_layers: List[str]
    parameter_constraints: Dict[str, str]


@dataclass
class SuperWeightCandidate:
    """Represents a candidate super-weight location."""
    layer: int
    component_type: str  # e.g., 'mlp.down_proj', 'attention.q_proj'
    row: int | None
    col: int | None
    confidence: float
    evidence: List[str]


class SemanticAnalyzer:
    """Analyzes technical papers to extract architectural information and predict super-weight locations."""
    
    def __init__(self) -> None:
        """Initialize the semantic analyzer."""
        self.logger = get_logger()
        self.model_patterns = {
            "llama": [r"llama", r"llama2", r"llama-2", r"llama3", r"llama-3"],
            "mistral": [r"mistral"],
            "gemma": [r"gemma"],
            "phi": [r"phi[ -]?(1|2|3)"],
            "mixtral": [r"mixtral"],
            "olmo": [r"olmo"],
            "gpt": [r"gpt[ -]?(2|3|4)"],
            "bert": [r"bert"],
            "t5": [r"t5"],
            "bart": [r"bart"],
            "opt": [r"opt"],
            "bloom": [r"bloom"],
            "falcon": [r"falcon"],
            "mpt": [r"mpt"],
            "starcoder": [r"starcoder", r"star[ -]*coder"],
        }
        
        # Key architectural components we're looking for
        self.architecture_keywords = [
            "layer", "transformer", "attention", "mlp", "feed.forward", 
            "feedforward", "ffn", "multi.head", "self.attention", "down.proj",
            "up.proj", "gate.proj", "q.proj", "k.proj", "v.proj", "o.proj"
        ]
        
        # Keywords that indicate super-weight locations
        self.superweight_indicators = [
            "down.proj", "down projection", "mlp.down_proj",
            "early layer", "first layer", "initial layer",
            "critical weight", "super.weight", "outlier parameter",
            "stop word", "logit", "activation"
        ]

    def _infer_model_family(self, text: str) -> str:
        """
        Infer the model family from text content.
        
        Args:
            text: Input text
            
        Returns:
            Model family name
        """
        if not isinstance(text, str):
            return "Unknown-Model"
            
        lowered = text.lower()
        for family, patterns in self.model_patterns.items():
            for pattern in patterns:
                if re.search(pattern, lowered):
                    # Capitalize first letter
                    return family.capitalize() + "-7B" if family in ["llama", "mistral", "olmo"] else family.capitalize()
        return "Unknown-Model"

    def _extract_numerical_values(self, text: str) -> Dict[str, List[int]]:
        """
        Extract numerical values that might represent architectural parameters.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping parameter names to lists of values
        """
        results: Dict[str, List[int]] = defaultdict(list)
        
        # Patterns for common architectural parameters
        patterns = {
            "layers": [
                r"(\d+)[ -]*(?:transformer[ -]*)?layers?",
                r"num[ -]*layers?[ =:]?(\d+)",
                r"depth[ =:]?(\d+)",
                r"uses (\d+) transformer layers"
            ],
            "hidden_size": [
                r"hidden[ -]*(?:size|dimension)[ =:]?(\d+)",
                r"d[ _]?model[ =:]?(\d+)",
                r"model[ -]*dimension[ =:]?(\d+)",
                r"hidden (?:size|dimension) of (\d+)"
            ],
            "mlp_expansion": [
                r"mlp[ -]*expansion[ =:]?(\d+)",
                r"feed[ -]*forward[ -]*expansion[ =:]?(\d+)",
                r"ffn[ -]*expansion[ =:]?(\d+)",
                r"expansion[ -]*factor[ =:]?(\d+)",
                r"MLP expansion of (\d+)",
                r"MLP expansion factor is (\d+)"
            ],
            "attention_heads": [
                r"attention[ -]*heads?[ =:]?(\d+)",
                r"num[ -]*heads?[ =:]?(\d+)",
                r"multi[ -]*head[ =:]?(\d+)",
                r"(\d+)[ -]*heads?",
                r"grouped[ -]*query[ -]*attention.*?(\d+)[ -]*heads?",
                r"with (\d+) heads",
                r"use (\d+) attention heads"
            ]
        }
        
        for param_name, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        value = int(match)
                        # Apply reasonable constraints to filter out false positives
                        if param_name == "layers" and 5 <= value <= 100:
                            results[param_name].append(value)
                        elif param_name == "hidden_size" and 64 <= value <= 16384:
                            results[param_name].append(value)
                        elif param_name == "mlp_expansion" and 2 <= value <= 16:
                            results[param_name].append(value)
                        elif param_name == "attention_heads" and 1 <= value <= 128:
                            results[param_name].append(value)
                    except ValueError:
                        continue
                        
        return dict(results)

    def _extract_architecture_components(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Extract mentioned architectural components and layers.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (key_components, mentioned_layers)
        """
        key_components: Set[str] = set()
        mentioned_layers: Set[str] = set()
        
        # Look for architectural components
        for keyword in self.architecture_keywords:
            if re.search(r"\b" + re.escape(keyword) + r"\b", text, re.IGNORECASE):
                key_components.add(keyword)
                
        # Look for layer mentions (especially early layers)
        layer_patterns = [
            r"(?:layer|block)[ -]*(\d+)",
            r"early[ -]*(?:layer|block)[s]?[ -]*(?:\d+(?:[ ,-]*\d+)*)?",
            r"first[ -]*(?:\d+)[ -]*(?:layer|block)[s]?",
        ]
        
        for pattern in layer_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                mentioned_layers.add(match if isinstance(match, str) else str(match))
                
        # Special handling for down_proj which might be written differently
        if re.search(r"down[ -]*proj", text, re.IGNORECASE):
            key_components.add("down.proj")
        if re.search(r"down[ -]*projection", text, re.IGNORECASE):
            key_components.add("down.proj")
                
        return list(key_components), list(mentioned_layers)

    def _identify_superweight_candidates(self, architecture: ModelArchitecture) -> List[SuperWeightCandidate]:
        """
        Identify candidate super-weight locations based on architecture.
        
        Args:
            architecture: Extracted model architecture
            
        Returns:
            List of super-weight candidates
        """
        candidates: List[SuperWeightCandidate] = []
        
        # Heuristic: Super-weights are often in early layers of MLP down_proj
        num_layers = architecture.num_layers or 32  # Default assumption
        max_layer = min(8, num_layers // 4)  # Focus on early layers
        
        # If we found evidence of down_proj or MLP components
        has_mlp = any("mlp" in comp.lower() or "feed" in comp.lower() for comp in architecture.key_components)
        has_down_proj = any("down" in comp.lower() for comp in architecture.key_components)
        
        # Check for explicit super-weight mentions
        has_superweight_mention = any("super" in ind.lower() or "outlier" in ind.lower() 
                                    for ind in self.superweight_indicators)
        
        if has_mlp or has_down_proj or has_superweight_mention:
            # Create candidates for early layers with different confidence levels
            for layer in range(max_layer):
                confidence = 0.8 if has_down_proj else 0.6
                if has_superweight_mention:
                    confidence = min(0.95, confidence + 0.2)
                    
                candidates.append(SuperWeightCandidate(
                    layer=layer,
                    component_type="mlp.down_proj",
                    row=None,  # We don't know exact positions without the model
                    col=None,
                    confidence=confidence,
                    evidence=[
                        f"MLP components mentioned in paper",
                        f"Early layer {layer} identified as critical"
                    ]
                ))
                
                # Add some variation by also considering adjacent layers with lower confidence
                if layer + 1 < max_layer:
                    candidates.append(SuperWeightCandidate(
                        layer=layer + 1,
                        component_type="mlp.down_proj",
                        row=None,
                        col=None,
                        confidence=confidence * 0.7,  # Lower confidence for adjacent layers
                        evidence=[
                            f"Adjacent to critical layer {layer}",
                            f"MLP components in layer {layer + 1}"
                        ]
                    ))
                
        # If no specific MLP components found, use general heuristics
        if not candidates:
            for layer in range(min(4, num_layers)):
                candidates.append(SuperWeightCandidate(
                    layer=layer,
                    component_type="unknown",
                    row=None,
                    col=None,
                    confidence=0.4,
                    evidence=[
                        f"General heuristic for early layer {layer}"
                    ]
                ))
                
        return candidates

    def analyze_paper(self, text: str) -> ModelArchitecture:
        """
        Analyze a paper to extract model architecture information.
        
        Args:
            text: Input paper text
            
        Returns:
            ModelArchitecture with extracted information
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
            
        self.logger.info("Analyzing paper for architectural information")
        
        # Infer model family
        model_family = self._infer_model_family(text)
        
        # Extract numerical values
        numerical_values = self._extract_numerical_values(text)
        
        # Extract architecture components
        key_components, mentioned_layers = self._extract_architecture_components(text)
        
        # Look for parameter constraints
        parameter_constraints: Dict[str, str] = {}
        if "down.proj" in " ".join(key_components).lower():
            parameter_constraints["down_proj"] = "super-weight candidate"
            
        architecture = ModelArchitecture(
            model_family=model_family,
            num_layers=numerical_values.get("layers", [None])[-1] if numerical_values.get("layers") else None,
            hidden_size=numerical_values.get("hidden_size", [None])[-1] if numerical_values.get("hidden_size") else None,
            mlp_expansion=numerical_values.get("mlp_expansion", [None])[-1] if numerical_values.get("mlp_expansion") else None,
            attention_heads=numerical_values.get("attention_heads", [None])[-1] if numerical_values.get("attention_heads") else None,
            key_components=key_components,
            mentioned_layers=mentioned_layers,
            parameter_constraints=parameter_constraints
        )
        
        self.logger.info(f"Extracted architecture: {architecture.model_family} with {len(key_components)} key components")
        return architecture

    def predict_superweight_candidates(self, text: str) -> List[SuperWeightCandidate]:
        """
        Predict super-weight candidates from paper text.
        
        Args:
            text: Input paper text
            
        Returns:
            List of super-weight candidates
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
            
        self.logger.info("Predicting super-weight candidates")
        
        # Analyze the paper
        architecture = self.analyze_paper(text)
        
        # Identify candidates
        candidates = self._identify_superweight_candidates(architecture)
        
        self.logger.info(f"Identified {len(candidates)} super-weight candidates")
        return candidates