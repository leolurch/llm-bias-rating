"""
Embedding Adapters for the bias evaluation framework.

This module provides a unified interface for different embedding models,
allowing the evaluator to compare results across multiple embedding approaches.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
import logging
import os

# Handle sentence-transformers import
try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    torch = None

# Handle OpenAI import
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

# Handle dotenv import
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

logger = logging.getLogger(__name__)


class EmbeddingAdapter(ABC):
    """Abstract base class for embedding adapters."""

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into embeddings.

        Args:
            texts: List of text strings to encode

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the embedding model."""
        pass

    def cleanup(self):
        """Optional cleanup method for releasing resources."""
        pass


class QwenEmbeddingAdapter(EmbeddingAdapter):
    """Adapter for Qwen3-Embedding models using sentence-transformers."""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str = "auto"):
        """
        Initialize the Qwen embedding adapter.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load the model on ('auto', 'cuda', 'cpu', 'mps')
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers and torch are required for QwenEmbeddingAdapter. "
                "Install with: pip install sentence-transformers torch"
            )

        self.model_name = model_name
        self.device = self._get_device(device)

        logger.info(f"Loading Qwen embedding model {model_name} on {self.device}")

        # Load the sentence transformer model
        self.model = SentenceTransformer(
            model_name, device=self.device, trust_remote_code=True
        )

        logger.info(f"Qwen embedding model loaded successfully")

    def _get_device(self, device: str) -> str:
        """Determine the appropriate device for model loading."""
        if device == "auto":
            if torch and torch.cuda.is_available():
                return "cuda"
            elif torch and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using the Qwen embedding model."""
        if not texts:
            return np.array([])

        # Use sentence-transformers to encode
        embeddings = self.model.encode(texts, convert_to_numpy=True)

        # Ensure we return 2D array even for single text
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        return embeddings

    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the Qwen embedding model."""
        return {
            "adapter_type": "qwen-sentence-transformers",
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dim": self.model.get_sentence_embedding_dimension(),
            "max_seq_length": getattr(self.model, 'max_seq_length', 'unknown'),
            "framework": "sentence-transformers"
        }

    def cleanup(self):
        """Clean up model resources."""
        if hasattr(self, 'model'):
            del self.model
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Qwen embedding adapter cleaned up")


class OpenAIEmbeddingAdapter(EmbeddingAdapter):
    """Adapter for OpenAI embedding models via API."""

    def __init__(self, model_name: str = "text-embedding-3-large", api_key: str = None):
        """
        Initialize the OpenAI embedding adapter.

        Args:
            model_name: OpenAI embedding model name
            api_key: OpenAI API key (if None, will try to get from environment)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai is required for OpenAIEmbeddingAdapter. "
                "Install with: pip install openai"
            )

        self.model_name = model_name

        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')

        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter"
            )

        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)

        logger.info(f"OpenAI embedding adapter initialized with model {model_name}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using OpenAI embedding API."""
        if not texts:
            return np.array([])

        try:
            # Call OpenAI embedding API
            response = self.client.embeddings.create(
                input=texts,
                model=self.model_name
            )

            # Extract embeddings from response
            embeddings = []
            for data in response.data:
                embeddings.append(data.embedding)

            return np.array(embeddings)

        except Exception as e:
            logger.error(f"Error calling OpenAI embedding API: {e}")
            raise RuntimeError(f"Failed to get embeddings from OpenAI: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the OpenAI embedding model."""
        # Model dimensions based on OpenAI documentation
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }

        return {
            "adapter_type": "openai-api",
            "model_name": self.model_name,
            "device": "api",
            "embedding_dim": model_dimensions.get(self.model_name, "unknown"),
            "max_seq_length": 8192,  # OpenAI's current limit
            "framework": "openai-api"
        }


class DummyEmbeddingAdapter(EmbeddingAdapter):
    """Dummy embedding adapter for testing purposes."""

    def __init__(self, embedding_dim: int = 768):
        """Initialize dummy adapter with specified embedding dimension."""
        self.embedding_dim = embedding_dim
        np.random.seed(42)  # For reproducible dummy embeddings
        logger.info(f"Dummy embedding adapter initialized with {embedding_dim} dimensions")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate dummy random embeddings."""
        if not texts:
            return np.array([])

        # Generate random embeddings based on text length (for some consistency)
        embeddings = []
        for text in texts:
            # Use text hash for reproducible randomness per text
            np.random.seed(hash(text) % 2**31)
            embedding = np.random.randn(self.embedding_dim)
            # Normalize to unit vector (similar to most embedding models)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)

        return np.array(embeddings)

    def get_model_info(self) -> Dict[str, Any]:
        """Return dummy model information."""
        return {
            "adapter_type": "dummy",
            "model_name": "dummy-embeddings",
            "device": "cpu",
            "embedding_dim": self.embedding_dim,
            "max_seq_length": "unlimited",
            "framework": "numpy"
        }


def create_embedding_adapter(adapter_type: str, **kwargs) -> EmbeddingAdapter:
    """
    Factory function to create embedding adapters.

    Args:
        adapter_type: Type of adapter to create
        **kwargs: Additional arguments for the adapter

    Returns:
        EmbeddingAdapter instance
    """
    adapter_type = adapter_type.lower()

    if adapter_type == "qwen":
        return QwenEmbeddingAdapter(**kwargs)
    elif adapter_type == "openai":
        return OpenAIEmbeddingAdapter(**kwargs)
    elif adapter_type == "dummy":
        return DummyEmbeddingAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown embedding adapter type: {adapter_type}")


def create_multiple_adapters(adapter_configs: List[Dict[str, Any]]) -> List[EmbeddingAdapter]:
    """
    Create multiple embedding adapters from configuration list.

    Args:
        adapter_configs: List of dictionaries with 'type' and optional parameters

    Returns:
        List of EmbeddingAdapter instances

    Example:
        configs = [
            {"type": "qwen", "model_name": "Qwen/Qwen3-Embedding-0.6B"},
            {"type": "openai", "model_name": "text-embedding-3-small"},
            {"type": "dummy", "embedding_dim": 768}
        ]
        adapters = create_multiple_adapters(configs)
    """
    adapters = []

    for i, config in enumerate(adapter_configs):
        if "type" not in config:
            raise ValueError(f"Adapter config {i} missing 'type' field")

        adapter_type = config.pop("type")

        try:
            adapter = create_embedding_adapter(adapter_type, **config)
            adapters.append(adapter)
            logger.info(f"Created adapter {i}: {adapter.get_model_info()['model_name']}")
        except Exception as e:
            logger.error(f"Failed to create adapter {i} ({adapter_type}): {e}")
            raise

    return adapters
