"""
LLM Adapter classes for the evaluation framework.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

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


class Qwen25_14BAdapter(LLMAdapter):
    """Adapter for Qwen2.5-14B model from BrainDAO."""

    def __init__(self, model_name: str = "braindao/Qwen2.5-14B", device: str = "auto"):
        """
        Initialize the Qwen2.5-14B adapter.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load the model on ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.device = self._get_device(device)

        logger.info(f"Loading Qwen2.5-14B model on {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        # Set pad_token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Smart model loading optimized for 16GB MacBook Pro
        self.model, self.actual_device = self._load_model_optimally(model_name)
        self.model.eval()

        logger.info(f"Qwen2.5-14B model loaded successfully on {self.actual_device}")

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

        # Strategy 1: Direct MPS loading (best for Apple Silicon with 16GB)
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
            # Determine actual device
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
            # Check device mapping if available
            if hasattr(model, "hf_device_map") and model.hf_device_map:
                first_device = list(model.hf_device_map.values())[0]
                return str(first_device)

            # Check device of first parameter
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
            # Tokenize input and ensure it's on the same device as model
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            inputs = inputs.to(self.actual_device)

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
            new_tokens = outputs[0][inputs.shape[1] :]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            return generated_text.strip()

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return f"Error: {str(e)}"

    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the model."""
        return {
            "model_name": self.model_name,
            "model_type": "Qwen2.5-14B",
            "parameters": "14.7B",
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
        logger.info("Qwen2.5-14B adapter cleaned up")


class Qwen25_7BAdapter(LLMAdapter):
    """Adapter for Qwen2.5-7B model."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B", device: str = "auto"):
        """
        Initialize the Qwen2.5-7B adapter.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load the model on ('auto', 'cuda', 'cpu', 'mps')
        """
        self.model_name = model_name
        self.device = self._get_device(device)

        logger.info(f"Loading Qwen2.5-7B model on {self.device}")

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

        logger.info(f"Qwen2.5-7B model loaded successfully on {self.actual_device}")

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

        # Strategy 1: Direct MPS loading (should work well for 7B model)
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
            "model_type": "Qwen2.5-7B",
            "parameters": "7.6B",
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
        logger.info("Qwen2.5-7B adapter cleaned up")


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
