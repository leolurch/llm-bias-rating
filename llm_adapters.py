"""
LLM Adapter classes for the evaluation framework.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import os

# Handle OpenAI import
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

# Handle x.ai SDK import
try:
    from xai_sdk import Client
    from xai_sdk.chat import user, system
    XAI_AVAILABLE = True
except ImportError:
    XAI_AVAILABLE = False
    Client = None
    user = None
    system = None

# Handle dotenv import
try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    load_dotenv = None

logger = logging.getLogger(__name__)


class LLMAdapter(ABC):
    """Abstract base class for LLM adapters."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion for the given prompt."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the model."""
        pass

    def cleanup(self):
        """Optional cleanup method for releasing resources."""
        pass


class OpenAIAdapter(LLMAdapter):
    """Adapter for OpenAI GPT models via API."""

    def __init__(
        self,
        model_name: str = "gpt-4o-2024-08-06",
        api_key: str = None,
        **kwargs,
    ):
        """
        Initialize the OpenAI adapter.

        Args:
            model_name: OpenAI model identifier (default: gpt-4o-2024-08-06)
            api_key: OpenAI API key (if None, will try to get from OPENAI_API_KEY env var)
            **kwargs: Additional arguments (device is ignored for API calls)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package not available. Install it with: pip install openai>=1.0.0"
            )

        self.model_name = model_name

        # Load .env file if available
        if DOTENV_AVAILABLE:
            load_dotenv()
            logger.debug("Loaded .env file for environment variables")
        else:
            logger.debug("python-dotenv not available, skipping .env file loading")

        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text completion using OpenAI Chat Completions API.

        Args:
            prompt: Input prompt text
            **kwargs: Generation parameters (ignored, using static temperature)

        Returns:
            Generated text completion
        """
        try:
            # Use chat completions API with static parameters
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.7
            )

            # Extract content from response
            result = response.choices[0].message.content
            if result is None or result.strip() == "":
                logger.warning(f"Model {self.model_name} produced empty response.")
                return f"[Model produced empty response]"

            logger.info(f"OpenAI API call successful")
            return result.strip()

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return f"Error: OpenAI API call failed - {str(e)}"

    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the OpenAI model."""
        return {
            "model_name": self.model_name,
            "model_type": "openai-api",
            "device": "api",
            "framework": "openai",
            "supports_chat": True,
            "api_based": True,
        }

    def cleanup(self):
        """Cleanup method for OpenAI adapter (no resources to release)."""
        # No cleanup needed for API-based adapter
        pass


class GrokAdapter(LLMAdapter):
    """Adapter for Grok models via x.ai SDK."""

    def __init__(
        self,
        model_name: str = "grok-3-mini",
        api_key: str = None,
        **kwargs,
    ):
        """
        Initialize the Grok adapter.

        Args:
            model_name: Grok model identifier (default: grok-3-mini)
            api_key: x.ai API key (if None, will try to get from XAI_API_KEY env var)
            **kwargs: Additional arguments (ignored for API calls)
        """
        if not XAI_AVAILABLE:
            raise ImportError(
                "x.ai SDK not available. Install it with: pip install xai-sdk"
            )

        self.model_name = model_name

        # Load .env file if available
        if DOTENV_AVAILABLE:
            load_dotenv()
            logger.debug("Loaded .env file for environment variables")
        else:
            logger.debug("python-dotenv not available, skipping .env file loading")

        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.getenv("XAI_API_KEY")

        if not api_key:
            raise ValueError(
                "x.ai API key required. Set XAI_API_KEY environment variable or pass api_key parameter."
            )

        # Initialize x.ai client
        self.client = Client(api_key=api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text completion using x.ai SDK.

        Args:
            prompt: Input prompt text
            **kwargs: Generation parameters (ignored, using static temperature)

        Returns:
            Generated text completion
        """
        try:
            # Create chat with static temperature
            chat = self.client.chat.create(model=self.model_name, temperature=0)
            chat.append(user(prompt))

            # Sample response
            response = chat.sample()
            result = response.content

            if result is None or result.strip() == "":
                logger.warning(f"Model {self.model_name} produced empty response.")
                return f"[Model produced empty response]"

            logger.info(f"x.ai SDK call successful")
            return result.strip()

        except Exception as e:
            logger.error(f"x.ai SDK call failed: {e}")
            return f"Error: x.ai SDK call failed - {str(e)}"

    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the Grok model."""
        return {
            "model_name": self.model_name,
            "model_type": "grok-api",
            "device": "api",
            "framework": "x.ai-sdk",
            "supports_chat": True,
            "api_based": True,
        }

    def cleanup(self):
        """Cleanup method for Grok adapter (no resources to release)."""
        # No cleanup needed for API-based adapter
        pass


class Qwen25_3BAdapter(LLMAdapter):
    """Adapter for Qwen2.5-3B model."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B", device: str = "auto"):
        """
        Initialize the Qwen2.5-3B adapter.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load the model on ('auto', 'cuda', 'cpu', 'mps')
        """
        self.model_name = model_name
        self.device = self._get_device(device)

        logger.info(f"Loading Qwen2.5-3B model on {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        # Set pad_token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Smart model loading optimized for 16GB MacBook Pro
        self.model, self.actual_device = self._load_model_optimally(model_name)
        self.model.eval()

        logger.info(f"Qwen2.5-3B model loaded successfully on {self.actual_device}")

    def _get_device(self, device: str) -> str:
        """Determine the appropriate device for model loading."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def _load_model_optimally(self, model_name: str):
        """Load model with optimal strategy for 16GB MacBook Pro."""

        # Strategy 1: Direct MPS loading (should work well for 3B model)
        if self.device == "mps":
            try:
                logger.info("Attempting direct MPS loading...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="mps",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                logger.info("✓ Successfully loaded on MPS")
                return model, "mps"
            except Exception as e:
                logger.warning(f"Direct MPS loading failed: {e}")

        # Strategy 2: CUDA direct loading
        elif self.device == "cuda":
            try:
                logger.info("Attempting direct CUDA loading...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="cuda",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                logger.info("✓ Successfully loaded on CUDA")
                return model, "cuda"
            except Exception as e:
                logger.warning(f"Direct CUDA loading failed: {e}")

        # Strategy 3: Auto device mapping (let transformers decide)
        try:
            logger.info("Attempting auto device mapping...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            actual_device = self._get_actual_device(model)
            logger.info(f"✓ Auto mapping successful, model on {actual_device}")
            return model, actual_device
        except Exception as e:
            logger.warning(f"Auto device mapping failed: {e}")

        # Strategy 4: CPU fallback
        try:
            logger.info("Falling back to CPU loading...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            logger.info("✓ CPU loading successful")
            return model, "cpu"
        except Exception as e:
            logger.error(f"All loading strategies failed: {e}")
            raise RuntimeError("Failed to load model with any strategy")

    def _get_actual_device(self, model):
        """Determine the actual device where the model is loaded."""
        try:
            if hasattr(model, "hf_device_map") and model.hf_device_map:
                first_device = list(model.hf_device_map.values())[0]
                return str(first_device)
            first_param = next(model.parameters())
            return str(first_param.device)
        except:
            return "unknown"

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """
        Generate text completion for the given prompt using chat template format.

        Args:
            prompt: Input prompt string
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_p: Top-p sampling parameter
            **kwargs: Additional generation parameters

        Returns:
            Generated text completion
        """
        try:
            # Format as chat messages for Qwen2.5
            messages = [{"role": "user", "content": prompt}]

            # Apply chat template and tokenize
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            # Move inputs to the correct device
            inputs = {k: v.to(self.actual_device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs,
                )

            # Decode only the new tokens (excluding the input prompt)
            input_length = inputs["input_ids"].shape[-1]
            new_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            return generated_text.strip()

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return f"Error: {str(e)}"

    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the model."""
        return {
            "model_name": self.model_name,
            "model_type": "Qwen2.5-3B",
            "parameters": "3.1B",
            "context_length": 131072,
            "device": self.actual_device,
            "architecture": "Transformer with RoPE, SwiGLU, RMSNorm",
        }

    def cleanup(self):
        """Clean up model resources."""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        logger.info("Qwen2.5-3B adapter cleaned up")


class BloomzAdapter(LLMAdapter):
    """Adapter for BigScience BLOOMZ-7B1 model."""
    def __init__(self, model_name: str = "bigscience/bloomz-7b1", device: str = "auto"):
        """
        Initialize the BLOOMZ-7B1 adapter.
        Args:
            model_name: HuggingFace model identifier
            device: Device to load the model on ('auto', 'cuda', 'cpu', 'mps')
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        logger.info(f"Loading BLOOMZ-7B1 model on {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set pad_token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Smart model loading optimized for large model
        self.model, self.actual_device = self._load_model_optimally(model_name)
        self.model.eval()
        logger.info(f"BLOOMZ-7B1 model loaded successfully on {self.actual_device}")

    def _get_device(self, device: str) -> str:
        """Determine the appropriate device for model loading."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def _load_model_optimally(self, model_name: str):
        """Load model with optimal strategy for available hardware."""
        try:
            if self.device == "cuda":
                # Use device_map="auto" for multi-GPU or large model handling
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                actual_device = self._get_actual_device(model)
                logger.info(f"CUDA loading successful on {actual_device}")
                return model, actual_device
            elif self.device == "mps":
                # MPS loading for Apple Silicon
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                model = model.to("mps")
                logger.info("MPS loading successful")
                return model, "mps"
            else:
                # CPU fallback
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                model = model.to("cpu")
                logger.info("CPU loading successful")
                return model, "cpu"
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise RuntimeError(f"Failed to load BLOOMZ model: {e}")

    def _get_actual_device(self, model):
        """Determine the actual device where the model is loaded."""
        try:
            if hasattr(model, "hf_device_map") and model.hf_device_map:
                first_device = list(model.hf_device_map.values())[0]
                return str(first_device)
            first_param = next(model.parameters())
            return str(first_param.device)
        except:
            return "unknown"

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """
        Generate text completion for the given prompt.
        Args:
            prompt: Input prompt string
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_p: Top-p sampling parameter
            **kwargs: Additional generation parameters
        Returns:
            Generated text completion
        """
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")

            # Move inputs to the correct device
            # Handle device string conversion for tensor operations
            target_device = self.actual_device if self.actual_device not in ["0", "1", "2", "3"] else f"cuda:{self.actual_device}"

            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(target_device)
            else:
                inputs = {k: v.to(target_device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs,
                )

            # Decode only the new tokens (excluding the input prompt)
            if isinstance(inputs, torch.Tensor):
                input_length = inputs.shape[-1]
            else:
                input_length = inputs["input_ids"].shape[-1]

            new_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            return generated_text.strip()

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return f"Error: {str(e)}"

    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the model."""
        return {
            "model_name": self.model_name,
            "model_type": "BLOOMZ-7B1",
            "parameters": "7.1B",
            "context_length": 2048,
            "device": self.actual_device,
            "architecture": "BLOOM with instruction tuning",
        }

    def cleanup(self):
        """Clean up model resources."""
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("BLOOMZ model resources cleaned up")


class DummyLLMAdapter(LLMAdapter):
    """Dummy LLM adapter for testing purposes."""

    def __init__(self, model_name: str = "dummy-model", device: str = "cpu", **kwargs):
        self.model_name = model_name
        self.device = device
        self.responses = [
            "I believe this candidate shows strong potential based on their experience.",
            "The applicant demonstrates excellent qualifications for this position.",
            "This person would be a valuable addition to our team.",
            "I have some concerns about this candidate's fit for the role.",
            "The candidate's background aligns well with our requirements.",
        ]
        self.call_count = 0

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a dummy response."""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response

    def get_model_info(self) -> Dict[str, Any]:
        """Return dummy model information."""
        return {
            "model_name": self.model_name,
            "model_type": "Dummy",
            "parameters": "0",
            "context_length": 2048,
            "device": self.device,
        }
