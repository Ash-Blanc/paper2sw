from __future__ import annotations

import pytest
import tempfile
import os
from pathlib import Path
from paper2sw.cache import CacheManager
from paper2sw.types import SuperWeightPrediction


def test_cache_manager_creation():
    """Test creating a CacheManager object."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cache = CacheManager(cache_dir=cache_dir, enabled=True, version_salt="test")
        
        assert cache.enabled == True
        assert cache.version_salt == "test"
        assert cache.cache_dir == cache_dir
        assert cache_dir.exists()


def test_cache_manager_default_directory():
    """Test CacheManager with default directory."""
    cache = CacheManager(enabled=True, version_salt="test")
    
    assert cache.enabled == True
    assert cache.version_salt == "test"
    assert cache.cache_dir.exists()


def test_cache_manager_disabled():
    """Test CacheManager when disabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cache = CacheManager(cache_dir=cache_dir, enabled=False, version_salt="test")
        
        assert cache.enabled == False


def test_cache_manager_put_and_get():
    """Test putting and getting items from cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cache = CacheManager(cache_dir=cache_dir, enabled=True, version_salt="test")
        
        # Create test predictions
        predictions = [
            SuperWeightPrediction(
                model_family="Llama-7B",
                layer=2,
                row=3968,
                col=7003,
                value=-17.328
            ),
            SuperWeightPrediction(
                model_family="Llama-7B",
                layer=5,
                row=1204,
                col=8191,
                value=12.456
            )
        ]
        
        # Put predictions in cache
        cache.put(
            model_id="test-model",
            text="This is a test paper",
            top_k=2,
            seed=42,
            predictions=predictions
        )
        
        # Get predictions from cache
        cached_predictions = cache.get(
            model_id="test-model",
            text="This is a test paper",
            top_k=2,
            seed=42
        )
        
        assert cached_predictions is not None
        assert len(cached_predictions) == len(predictions)
        for original, cached in zip(predictions, cached_predictions):
            assert original.to_dict() == cached.to_dict()


def test_cache_manager_get_nonexistent():
    """Test getting non-existent items from cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cache = CacheManager(cache_dir=cache_dir, enabled=True, version_salt="test")
        
        # Try to get non-existent item
        cached_predictions = cache.get(
            model_id="nonexistent-model",
            text="Nonexistent paper",
            top_k=5,
            seed=None
        )
        
        assert cached_predictions is None


def test_cache_manager_disabled_get_put():
    """Test that get and put don't work when cache is disabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cache = CacheManager(cache_dir=cache_dir, enabled=False, version_salt="test")
        
        # Create test predictions
        predictions = [
            SuperWeightPrediction(
                model_family="Llama-7B",
                layer=2,
                row=3968,
                col=7003,
                value=-17.328
            )
        ]
        
        # Put should not raise an error but should do nothing
        cache.put(
            model_id="test-model",
            text="This is a test paper",
            top_k=1,
            seed=42,
            predictions=predictions
        )
        
        # Get should return None
        cached_predictions = cache.get(
            model_id="test-model",
            text="This is a test paper",
            top_k=1,
            seed=42
        )
        
        assert cached_predictions is None


def test_cache_manager_different_parameters():
    """Test that cache correctly distinguishes between different parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cache = CacheManager(cache_dir=cache_dir, enabled=True, version_salt="test")
        
        # Create different test predictions
        predictions1 = [
            SuperWeightPrediction(
                model_family="Llama-7B",
                layer=2,
                row=3968,
                col=7003,
                value=-17.328
            )
        ]
        
        predictions2 = [
            SuperWeightPrediction(
                model_family="Mistral-7B",
                layer=1,
                row=1024,
                col=2048,
                value=5.123
            )
        ]
        
        # Put predictions with different parameters
        cache.put(
            model_id="model1",
            text="Paper 1",
            top_k=1,
            seed=42,
            predictions=predictions1
        )
        
        cache.put(
            model_id="model2",
            text="Paper 2",
            top_k=1,
            seed=24,
            predictions=predictions2
        )
        
        # Get predictions with first set of parameters
        cached_predictions1 = cache.get(
            model_id="model1",
            text="Paper 1",
            top_k=1,
            seed=42
        )
        
        assert cached_predictions1 is not None
        assert len(cached_predictions1) == 1
        assert cached_predictions1[0].to_dict() == predictions1[0].to_dict()
        
        # Get predictions with second set of parameters
        cached_predictions2 = cache.get(
            model_id="model2",
            text="Paper 2",
            top_k=1,
            seed=24
        )
        
        assert cached_predictions2 is not None
        assert len(cached_predictions2) == 1
        assert cached_predictions2[0].to_dict() == predictions2[0].to_dict()