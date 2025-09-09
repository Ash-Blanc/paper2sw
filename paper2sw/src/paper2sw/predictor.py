from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional

from .io_utils import read_text_from_source, write_jsonl
from .types import SuperWeightPrediction
from .cache import CacheManager
from .selector import select_relevant
from .logging_config import get_logger


class Predictor:
    """Main predictor class that orchestrates the prediction process."""
    
    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        precision: str = "bf16",
        enable_cache: bool = True,
        selection_keep_ratio: float = 1.0,
        backend: str = "dummy",  # placeholder for future backends
        cache_dir: str | Path | None = None,
    ) -> None:
        """
        Initialize the predictor.
        
        Args:
            model_id: Identifier for the model
            device: Device to run on
            precision: Numerical precision
            enable_cache: Whether to enable caching
            selection_keep_ratio: Ratio of text to keep during selection
            backend: Backend identifier
            cache_dir: Directory for cache files
            
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
            
        if not isinstance(enable_cache, bool):
            raise ValueError("enable_cache must be a boolean")
            
        if not isinstance(selection_keep_ratio, (int, float)) or not 0.0 <= selection_keep_ratio <= 1.0:
            raise ValueError("selection_keep_ratio must be a float between 0.0 and 1.0")
            
        if not isinstance(backend, str):
            raise ValueError("backend must be a string")
            
        self.model_id = model_id
        try:
            # Try to import the semantic model first, fall back to dummy if needed
            try:
                from .model import SemanticDiffusionModel
                self.model = SemanticDiffusionModel(model_id=model_id, device=device, precision=precision)
            except ImportError:
                from .model import DummyDiffusionModel
                self.model = DummyDiffusionModel(model_id=model_id, device=device, precision=precision)
        except Exception as e:
            raise ValueError(f"Failed to initialize model: {e}")
            
        try:
            self.cache = CacheManager(cache_dir=cache_dir, enabled=enable_cache, version_salt=f"{model_id}:{precision}")
        except Exception as e:
            self.logger.warning(f"Failed to initialize cache: {e}")
            self.cache = CacheManager(cache_dir=cache_dir, enabled=False, version_salt=f"{model_id}:{precision}")
            
        self.selection_keep_ratio = float(selection_keep_ratio)
        self.backend = backend

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "paper2sw/paper2sw-diff-base",
        device: str = "cpu",
        precision: str = "bf16",
        enable_cache: bool = True,
        selection_keep_ratio: float = 1.0,
        backend: str = "dummy",
        cache_dir: str | Path | None = None,
    ) -> "Predictor":
        """
        Create a predictor from pretrained model settings.
        
        Args:
            model_id: Identifier for the model
            device: Device to run on
            precision: Numerical precision
            enable_cache: Whether to enable caching
            selection_keep_ratio: Ratio of text to keep during selection
            backend: Backend identifier
            cache_dir: Directory for cache files
            
        Returns:
            Predictor instance
        """
        return cls(
            model_id=model_id,
            device=device,
            precision=precision,
            enable_cache=enable_cache,
            selection_keep_ratio=selection_keep_ratio,
            backend=backend,
            cache_dir=cache_dir,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Predictor":
        """
        Create a predictor from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Predictor instance
            
        Raises:
            TypeError: If config is not a dictionary
        """
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary")
            
        model_id = str(config.get("model_id", "paper2sw/paper2sw-diff-base"))
        device = str(config.get("device", "cpu"))
        precision = str(config.get("precision", "bf16"))
        enable_cache = bool(config.get("enable_cache", True))
        selection_keep_ratio = float(config.get("selection_keep_ratio", 1.0))
        backend = str(config.get("backend", "dummy"))
        cache_dir = config.get("cache_dir")
        return cls(
            model_id=model_id,
            device=device,
            precision=precision,
            enable_cache=enable_cache,
            selection_keep_ratio=selection_keep_ratio,
            backend=backend,
            cache_dir=cache_dir,
        )

    def _maybe_select(self, text: str) -> str:
        """
        Apply text selection if needed.
        
        Args:
            text: Input text
            
        Returns:
            Selected text or original text
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
            
        if self.selection_keep_ratio >= 0.999:
            return text
            
        try:
            sel = select_relevant(text, keep_ratio=self.selection_keep_ratio)
            return sel.text
        except Exception as e:
            self.logger.warning(f"Failed to select text: {e}")
            return text

    def predict(
        self,
        paper: str | Path,
        top_k: int = 5,
        seed: int | None = None,
        use_cache: Optional[bool] = None,
    ) -> List[SuperWeightPrediction]:
        """
        Predict super-weights from a paper.
        
        Args:
            paper: URL or path to paper text
            top_k: Number of predictions to return
            seed: Random seed for reproducibility
            use_cache: Whether to use cache (None uses default)
            
        Returns:
            List of SuperWeightPrediction objects
            
        Raises:
            Exception: If prediction fails
        """
        try:
            text = read_text_from_source(paper)
        except Exception as e:
            raise IOError(f"Failed to read paper from {paper}: {e}")
            
        try:
            text = self._maybe_select(text)
        except Exception as e:
            self.logger.warning(f"Failed to select text: {e}")
            
        cache_enabled = use_cache if use_cache is not None else self.cache.enabled
        if cache_enabled:
            try:
                cached = self.cache.get(model_id=self.model_id, text=text, top_k=top_k, seed=seed)
                if cached is not None:
                    return cached
            except Exception as e:
                self.logger.warning(f"Failed to read from cache: {e}")
                
        try:
            preds = self.model.predict(text=text, top_k=top_k, seed=seed)
        except Exception as e:
            raise RuntimeError(f"Failed to generate predictions: {e}")
            
        if cache_enabled:
            try:
                self.cache.put(model_id=self.model_id, text=text, top_k=top_k, seed=seed, predictions=preds)
            except Exception as e:
                self.logger.warning(f"Failed to write to cache: {e}")
                
        return preds

    def predict_batch(
        self,
        papers: Iterable[str | Path],
        top_k: int = 5,
        seed: int | None = None,
        use_cache: Optional[bool] = None,
    ) -> List[List[SuperWeightPrediction]]:
        """
        Predict super-weights for multiple papers.
        
        Args:
            papers: Iterable of URLs or paths to paper texts
            top_k: Number of predictions to return per paper
            seed: Random seed for reproducibility
            use_cache: Whether to use cache (None uses default)
            
        Returns:
            List of lists of SuperWeightPrediction objects
        """
        if not hasattr(papers, '__iter__'):
            raise TypeError("papers must be iterable")
            
        results: List[List[SuperWeightPrediction]] = []
        for i, item in enumerate(papers):
            try:
                result = self.predict(paper=item, top_k=top_k, seed=seed, use_cache=use_cache)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to predict for paper {i} ({item}): {e}")
                results.append([])  # Return empty list for failed predictions
        return results

    def save_jsonl(self, predictions: List[SuperWeightPrediction], path: str | Path) -> None:
        """
        Save predictions to a JSONL file.
        
        Args:
            predictions: List of predictions to save
            path: Path to save to
        """
        try:
            write_jsonl(predictions, path)
        except Exception as e:
            raise IOError(f"Failed to save predictions to {path}: {e}")