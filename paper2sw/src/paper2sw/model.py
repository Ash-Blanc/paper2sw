from __future__ import annotations

import hashlib
import random
from typing import List

from .types import SuperWeightPrediction
from .logging_config import get_logger
from .semantic_analyzer import SemanticAnalyzer


class SemanticDiffusionModel:
    """A semantic model that predicts super-weights based on architectural analysis of papers."""
    
    def __init__(self, model_id: str, device: str = "cpu", precision: str = "bf16") -> None:
        """
        Initialize the semantic model.
        
        Args:
            model_id: Identifier for the model
            device: Device to run on (cpu, cuda, etc.)
            precision: Numerical precision (bf16, fp16, fp32)
            
        Raises:
            ValueError: If parameters are invalid
        """
        self.logger = get_logger()
        
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model_id must be a non-empty string")
            
        if not isinstance(device, str):
            raise ValueError("device must be a string")
            
        if not isinstance(precision, str):
            raise ValueError("precision must be a string")
            
        self.model_id = model_id
        self.device = device
        self.precision = precision
        self.analyzer = SemanticAnalyzer()

    def predict(self, text: str, top_k: int = 5, seed: int | None = None) -> List[SuperWeightPrediction]:
        """
        Generate predictions based on semantic analysis of the input text.
        
        Args:
            text: Input text to generate predictions from
            top_k: Number of predictions to generate
            seed: Random seed for reproducibility
            
        Returns:
            List of SuperWeightPrediction objects
            
        Raises:
            ValueError: If parameters are invalid
            TypeError: If text is not a string
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
            
        if not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
            
        if top_k <= 0:
            return []
            
        if seed is not None and not isinstance(seed, int):
            raise TypeError("seed must be an integer or None")
            
        # Set seed for reproducibility
        if seed is not None:
            random.seed(seed)
            
        # Analyze the paper to extract architecture information
        try:
            architecture = self.analyzer.analyze_paper(text)
            candidates = self.analyzer.predict_superweight_candidates(text)
        except Exception as e:
            self.logger.warning(f"Failed to analyze paper semantically: {e}")
            # Fallback to basic model family inference
            architecture = None
            candidates = []
            
        predictions: List[SuperWeightPrediction] = []
        
        # If we have semantic candidates, use them
        if candidates:
            self.logger.info(f"Using {len(candidates)} semantic candidates for predictions")
            
            # Convert candidates to predictions
            for i, candidate in enumerate(candidates[:top_k]):
                # Generate plausible matrix dimensions based on model family
                matrix_dim = self._get_matrix_dimension(architecture.model_family if architecture else "Unknown-Model")
                
                # Generate row/col indices with preference for likely super-weight positions
                # Super-weights are often in specific regions of the weight matrices
                row_index = random.randint(0, matrix_dim - 1)
                col_index = random.randint(0, matrix_dim - 1)
                
                # Adjust indices based on component type
                if candidate.component_type == "mlp.down_proj":
                    # For down_proj, super-weights often have specific patterns
                    # This is a heuristic - in real models, they might be in certain rows/cols
                    if random.random() < 0.7:  # 70% chance to use heuristic positions
                        # Focus on middle ranges where super-weights are often found
                        row_index = random.randint(matrix_dim // 4, 3 * matrix_dim // 4)
                        col_index = random.randint(matrix_dim // 4, 3 * matrix_dim // 4)
                
                # Generate value based on confidence and heuristics
                # Super-weights typically have larger absolute values
                base_value = random.uniform(-15.0, 15.0) * (0.5 + candidate.confidence)
                
                predictions.append(
                    SuperWeightPrediction(
                        model_family=architecture.model_family if architecture else "Unknown-Model",
                        layer=candidate.layer,
                        row=row_index,
                        col=col_index,
                        value=float(base_value),
                    )
                )
        else:
            # Fallback to heuristic-based generation if semantic analysis fails
            self.logger.info("Falling back to heuristic-based generation")
            predictions = self._generate_heuristic_predictions(text, top_k, seed)
                
        self.logger.info(f"Generated {len(predictions)} predictions")
        return predictions

    def _get_matrix_dimension(self, model_family: str) -> int:
        """
        Get plausible matrix dimension based on model family.
        
        Args:
            model_family: Model family name
            
        Returns:
            Matrix dimension
        """
        dimensions = {
            "Llama-7B": 4096,
            "Llama-13B": 5120,
            "Llama-30B": 6656,
            "Llama-65B": 8192,
            "Mistral-7B": 4096,
            "Mixtral-8x7B": 4096,
            "OLMo-7B": 4096,
            "Gemma-2B": 2048,
            "Gemma-7B": 3072,
            "Phi-1": 2048,
            "Phi-2": 2560,
            "Phi-3": 3072,
            "GPT-2": 768,
            "GPT-3": 12288,
            "BERT-Base": 768,
            "BERT-Large": 1024,
        }
        return dimensions.get(model_family, 4096)  # Default to 4096

    def _generate_heuristic_predictions(self, text: str, top_k: int, seed: int | None = None) -> List[SuperWeightPrediction]:
        """
        Generate predictions using heuristics when semantic analysis fails.
        
        Args:
            text: Input text
            top_k: Number of predictions to generate
            seed: Random seed
            
        Returns:
            List of SuperWeightPrediction objects
        """
        # Infer model family using basic heuristics
        model_family = self._infer_model_family(text)
        matrix_dim = self._get_matrix_dimension(model_family)
        
        # Focus on early layers where super-weights are commonly found
        max_layer = 12  # Heuristic: most super-weights in first 12 layers
        
        predictions: List[SuperWeightPrediction] = []
        used_coords: set[tuple[int, int, int]] = set()
        
        for _ in range(top_k):
            try:
                # Try to find unique coordinates
                for _attempt in range(10):
                    layer_index = random.randint(0, max_layer - 1)
                    row_index = random.randint(0, matrix_dim - 1)
                    col_index = random.randint(0, matrix_dim - 1)
                    if (layer_index, row_index, col_index) not in used_coords:
                        used_coords.add((layer_index, row_index, col_index))
                        break
                        
                # Super-weights typically have larger absolute values
                raw_value = random.uniform(-20.0, 20.0)
                
                predictions.append(
                    SuperWeightPrediction(
                        model_family=model_family,
                        layer=layer_index,
                        row=row_index,
                        col=col_index,
                        value=float(raw_value),
                    )
                )
            except Exception as e:
                self.logger.warning(f"Failed to generate heuristic prediction: {e}")
                continue
                
        return predictions

    def _infer_model_family(self, text: str) -> str:
        """
        Infer the model family from text content.
        
        Args:
            text: Input text
            
        Returns:
            Model family name
        """
        return self.analyzer._infer_model_family(text)