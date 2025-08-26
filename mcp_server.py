#!/usr/bin/env python3
"""
Multi-Model MCP Server for LLM_Train

This server provides MCP (Model Context Protocol) access to multiple local models
managed by the LLM_Train application. It supports model discovery, switching,
and inference through a standardized interface.
"""

import asyncio
import json
import logging
import os
import sys
import platform
import subprocess
from typing import Any, Dict, List, Optional, Sequence
import argparse
from pathlib import Path

try:
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        CallToolRequestParams,
        GetPromptRequestParams,
        ListPromptsRequestParams,
        ListToolsRequestParams,
        Prompt,
        PromptMessage,
        Resource,
        TextContent,
        Tool,
        EmbeddedResource,
    )
except ImportError:
    print("MCP library not found. Install with: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Import our local modules
try:
    from model_library import ModelLibrary, ModelInfo
    from settings_manager import SettingsManager
    from llama_cpp import Llama
except ImportError as e:
    print(f"Required modules not found: {e}", file=sys.stderr)
    sys.exit(1)


class MultiModelMCPServer:
    """MCP Server for managing multiple local models with CUDA support."""
    
    def __init__(self, settings_path: str = "settings.json"):
        self.settings = SettingsManager(settings_path)
        self.library = None
        self.current_model = None
        self.current_llm = None
        self.model_cache = {}  # Cache for loaded models
        
        # Initialize model library if configured
        library_root = self.settings.get('library.root_folder', '')
        if library_root and os.path.exists(library_root):
            max_depth = self.settings.get('library.max_depth', 3)
            self.library = ModelLibrary(library_root, max_depth)
            self.library._load_index()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Detect system capabilities
        self.system_info = self._detect_system_capabilities()
        self.logger.info(f"System capabilities: {self.system_info}")
    
    def _detect_system_capabilities(self) -> Dict[str, Any]:
        """Detect system capabilities including CUDA, ROCm, and Metal support."""
        capabilities = {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "cuda_available": False,
            "cuda_version": None,
            "cuda_devices": 0,
            "rocm_available": False,
            "metal_available": False,
            "intel_gpu_available": False,
            "recommended_layers": 0
        }
        
        try:
            # Check for CUDA (NVIDIA)
            if capabilities["platform"] in ["Windows", "Linux"]:
                try:
                    # Try nvidia-smi command
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=count,driver_version", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        if lines and lines[0]:
                            parts = lines[0].split(', ')
                            if len(parts) >= 2:
                                capabilities["cuda_devices"] = len(lines)
                                capabilities["cuda_version"] = parts[1]
                                capabilities["cuda_available"] = True
                                
                                # Recommend using most GPU layers for CUDA
                                capabilities["recommended_layers"] = 35  # Good default for most models
                                self.logger.info(f"CUDA detected: {capabilities['cuda_devices']} device(s), driver {capabilities['cuda_version']}")
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                    pass
                
                # Check for Intel GPU on Windows (Arc/Iris Xe)
                if capabilities["platform"] == "Windows":
                    try:
                        result = subprocess.run(
                            ["wmic", "path", "win32_VideoController", "get", "name"],
                            capture_output=True, text=True, timeout=10
                        )
                        if result.returncode == 0 and "intel" in result.stdout.lower():
                            capabilities["intel_gpu_available"] = True
                            if not capabilities["cuda_available"]:
                                capabilities["recommended_layers"] = 15  # Conservative for Intel GPU
                            self.logger.info("Intel GPU detected")
                    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                        pass
            
            # Check for ROCm (AMD) on Linux
            elif capabilities["platform"] == "Linux":
                try:
                    result = subprocess.run(
                        ["rocm-smi", "--showproductname"],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0:
                        capabilities["rocm_available"] = True
                        capabilities["recommended_layers"] = 25  # Good default for ROCm
                        self.logger.info("ROCm (AMD GPU) detected")
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                    pass
            
            # Check for Metal (Apple Silicon) on macOS
            elif capabilities["platform"] == "Darwin":
                # Check if running on Apple Silicon
                if "arm" in capabilities["architecture"].lower() or "apple" in platform.processor().lower():
                    capabilities["metal_available"] = True
                    capabilities["recommended_layers"] = 30  # Good default for Apple Silicon
                    self.logger.info("Apple Silicon (Metal) detected")
            
        except Exception as e:
            self.logger.warning(f"Error detecting system capabilities: {e}")
        
        return capabilities
    
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models."""
        if self.library:
            return list(self.library.models.values())
        return []
    
    def load_model(self, model_path: str, **kwargs) -> bool:
        """Load a model for inference with optimized GPU acceleration."""
        try:
            # Check if model is already loaded
            if model_path in self.model_cache:
                self.current_model = model_path
                self.current_llm = self.model_cache[model_path]
                return True
            
            # Get parameters with smart defaults based on system capabilities
            n_ctx = kwargs.get('n_ctx', 4096)
            n_threads = kwargs.get('n_threads', min(os.cpu_count() or 4, 8))  # Cap threads for stability
            
            # Smart GPU layer detection
            n_gpu_layers = kwargs.get('n_gpu_layers')
            if n_gpu_layers is None:
                # Auto-detect optimal GPU layers
                if self.system_info["cuda_available"]:
                    n_gpu_layers = self.system_info["recommended_layers"]
                elif self.system_info["rocm_available"]:
                    n_gpu_layers = self.system_info["recommended_layers"]
                elif self.system_info["metal_available"]:
                    n_gpu_layers = self.system_info["recommended_layers"]
                elif self.system_info["intel_gpu_available"]:
                    n_gpu_layers = self.system_info["recommended_layers"]
                else:
                    n_gpu_layers = 0  # CPU only
            
            # Additional optimizations based on platform
            llm_kwargs = {
                "model_path": model_path,
                "n_ctx": n_ctx,
                "n_gpu_layers": n_gpu_layers,
                "n_threads": n_threads,
                "verbose": False
            }
            
            # Platform-specific optimizations
            if self.system_info["platform"] == "Windows":
                # Windows optimizations
                if self.system_info["cuda_available"]:
                    llm_kwargs["n_batch"] = 512  # Good batch size for CUDA on Windows
                elif self.system_info["intel_gpu_available"]:
                    llm_kwargs["n_batch"] = 256  # Conservative for Intel GPU
            
            elif self.system_info["platform"] == "Linux":
                # Linux optimizations
                if self.system_info["cuda_available"]:
                    llm_kwargs["n_batch"] = 512
                    llm_kwargs["use_mmap"] = True  # Better memory management on Linux
                elif self.system_info["rocm_available"]:
                    llm_kwargs["n_batch"] = 256  # Conservative for ROCm
            
            elif self.system_info["platform"] == "Darwin":
                # macOS optimizations
                if self.system_info["metal_available"]:
                    llm_kwargs["n_batch"] = 512
                    llm_kwargs["use_mmap"] = True
            
            self.logger.info(f"Loading model with: {n_gpu_layers} GPU layers, {n_threads} threads")
            
            llm = Llama(**llm_kwargs)
            
            # Cache the model (limit cache size)
            if len(self.model_cache) >= 3:  # Max 3 models in cache
                # Remove oldest model
                oldest_key = next(iter(self.model_cache))
                del self.model_cache[oldest_key]
            
            self.model_cache[model_path] = llm
            self.current_model = model_path
            self.current_llm = llm
            
            self.logger.info(f"Successfully loaded model: {model_path}")
            if n_gpu_layers > 0:
                acceleration = "CUDA" if self.system_info["cuda_available"] else \
                              "ROCm" if self.system_info["rocm_available"] else \
                              "Metal" if self.system_info["metal_available"] else \
                              "Intel GPU" if self.system_info["intel_gpu_available"] else "GPU"
                self.logger.info(f"Using {acceleration} acceleration with {n_gpu_layers} layers")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_path}: {e}")
            return False
    
    def generate_text(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text using the current model."""
        if not self.current_llm:
            return {"error": "No model loaded"}
        
        try:
            max_tokens = kwargs.get('max_tokens', 256)
            temperature = kwargs.get('temperature', 0.7)
            stream = kwargs.get('stream', False)
            
            if stream:
                # For MCP, we'll collect the stream and return the full result
                result = ""
                for chunk in self.current_llm.create_completion(
                    prompt, 
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True
                ):
                    if 'choices' in chunk and chunk['choices']:
                        text = chunk['choices'][0].get('text', '')
                        result += text
                
                return {
                    "text": result,
                    "model": self.current_model,
                    "tokens": len(result.split())
                }
            else:
                response = self.current_llm.create_completion(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                return {
                    "text": response['choices'][0]['text'],
                    "model": self.current_model,
                    "tokens": response['usage']['total_tokens']
                }
                
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return {"error": str(e)}
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate chat completion using the current model."""
        if not self.current_llm:
            return {"error": "No model loaded"}
        
        try:
            max_tokens = kwargs.get('max_tokens', 256)
            temperature = kwargs.get('temperature', 0.7)
            
            response = self.current_llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return {
                "message": response['choices'][0]['message'],
                "model": self.current_model,
                "tokens": response['usage']['total_tokens']
            }
            
        except Exception as e:
            self.logger.error(f"Chat completion failed: {e}")
            return {"error": str(e)}


# Global server instance
mcp_server = MultiModelMCPServer()

# Create MCP server
server = Server("llm-train-models")


@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available tools."""
    return [
        Tool(
            name="list_models",
            description="List all available models in the library",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="load_model",
            description="Load a specific model for inference",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                        "description": "Path to the model file"
                    },
                    "n_ctx": {
                        "type": "integer",
                        "description": "Context window size",
                        "default": 4096
                    },
                    "n_gpu_layers": {
                        "type": "integer", 
                        "description": "Number of GPU layers",
                        "default": 0
                    }
                },
                "required": ["model_path"]
            }
        ),
        Tool(
            name="generate_text",
            description="Generate text using the current model",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The text prompt for generation"
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens to generate",
                        "default": 256
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature",
                        "default": 0.7
                    }
                },
                "required": ["prompt"]
            }
        ),
        Tool(
            name="chat_completion",
            description="Generate chat completion using the current model",
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "description": "Chat messages",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "content": {"type": "string"}
                            },
                            "required": ["role", "content"]
                        }
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens to generate",
                        "default": 256
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature",
                        "default": 0.7
                    }
                },
                "required": ["messages"]
            }
        ),
        Tool(
            name="get_current_model",
            description="Get information about the currently loaded model",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_system_info",
            description="Get system capabilities and GPU acceleration status",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    
    if name == "list_models":
        models = mcp_server.get_available_models()
        model_list = []
        
        for model in models:
            model_info = {
                "name": model.name,
                "path": model.path,
                "type": model.file_type,
                "size_mb": round(model.size_mb, 1),
                "modified": model.modified_date,
                "tags": model.tags,
                "metadata": model.metadata
            }
            model_list.append(model_info)
        
        return [TextContent(
            type="text",
            text=json.dumps(model_list, indent=2)
        )]
    
    elif name == "load_model":
        model_path = arguments.get("model_path")
        n_ctx = arguments.get("n_ctx", 4096)
        n_gpu_layers = arguments.get("n_gpu_layers", 0)
        
        if not model_path:
            return [TextContent(type="text", text="Error: model_path is required")]
        
        success = mcp_server.load_model(
            model_path, 
            n_ctx=n_ctx, 
            n_gpu_layers=n_gpu_layers
        )
        
        if success:
            return [TextContent(
                type="text",
                text=f"Successfully loaded model: {model_path}"
            )]
        else:
            return [TextContent(
                type="text",
                text=f"Failed to load model: {model_path}"
            )]
    
    elif name == "generate_text":
        prompt = arguments.get("prompt")
        max_tokens = arguments.get("max_tokens", 256)
        temperature = arguments.get("temperature", 0.7)
        
        if not prompt:
            return [TextContent(type="text", text="Error: prompt is required")]
        
        result = mcp_server.generate_text(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    elif name == "chat_completion":
        messages = arguments.get("messages", [])
        max_tokens = arguments.get("max_tokens", 256)
        temperature = arguments.get("temperature", 0.7)
        
        if not messages:
            return [TextContent(type="text", text="Error: messages are required")]
        
        result = mcp_server.chat_completion(
            messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    elif name == "get_current_model":
        if mcp_server.current_model:
            # Find model info
            models = mcp_server.get_available_models()
            current_info = None
            
            for model in models:
                if model.path == mcp_server.current_model:
                    current_info = {
                        "name": model.name,
                        "path": model.path,
                        "type": model.file_type,
                        "size_mb": round(model.size_mb, 1),
                        "metadata": model.metadata
                    }
                    break
            
            if current_info:
                return [TextContent(
                    type="text",
                    text=json.dumps(current_info, indent=2)
                )]
        
        return [TextContent(type="text", text="No model currently loaded")]
    
    elif name == "get_system_info":
        system_info = {
            "platform": mcp_server.system_info["platform"],
            "architecture": mcp_server.system_info["architecture"],
            "acceleration": {
                "cuda_available": mcp_server.system_info["cuda_available"],
                "cuda_version": mcp_server.system_info["cuda_version"],
                "cuda_devices": mcp_server.system_info["cuda_devices"],
                "rocm_available": mcp_server.system_info["rocm_available"],
                "metal_available": mcp_server.system_info["metal_available"],
                "intel_gpu_available": mcp_server.system_info["intel_gpu_available"],
                "recommended_layers": mcp_server.system_info["recommended_layers"]
            },
            "current_model_acceleration": "Unknown"
        }
        
        # Add current model acceleration info
        if mcp_server.current_llm:
            if mcp_server.system_info["cuda_available"]:
                system_info["current_model_acceleration"] = "CUDA (NVIDIA)"
            elif mcp_server.system_info["rocm_available"]:
                system_info["current_model_acceleration"] = "ROCm (AMD)"
            elif mcp_server.system_info["metal_available"]:
                system_info["current_model_acceleration"] = "Metal (Apple)"
            elif mcp_server.system_info["intel_gpu_available"]:
                system_info["current_model_acceleration"] = "Intel GPU"
            else:
                system_info["current_model_acceleration"] = "CPU Only"
        
        return [TextContent(
            type="text",
            text=json.dumps(system_info, indent=2)
        )]
    
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


@server.list_prompts()
async def handle_list_prompts() -> List[Prompt]:
    """List available prompts."""
    return [
        Prompt(
            name="model_comparison",
            description="Compare multiple models on the same prompt",
            arguments=[
                {
                    "name": "prompt", 
                    "description": "The prompt to test across models",
                    "required": True
                },
                {
                    "name": "models",
                    "description": "List of model paths to compare",
                    "required": True
                }
            ]
        ),
        Prompt(
            name="model_benchmark",
            description="Benchmark a model with standard prompts",
            arguments=[
                {
                    "name": "model_path",
                    "description": "Path to the model to benchmark",
                    "required": True
                }
            ]
        )
    ]


@server.get_prompt()
async def handle_get_prompt(name: str, arguments: Dict[str, str]) -> Prompt:
    """Handle prompt requests."""
    
    if name == "model_comparison":
        prompt_text = arguments.get("prompt", "")
        models = arguments.get("models", "").split(",")
        
        messages = [
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"Compare the following models on this prompt: '{prompt_text}'\n\n"
                         f"Models to test: {', '.join(models)}\n\n"
                         f"For each model, load it and generate a response, then provide a comparison."
                )
            )
        ]
        
        return Prompt(
            name=name,
            description="Compare multiple models",
            messages=messages
        )
    
    elif name == "model_benchmark":
        model_path = arguments.get("model_path", "")
        
        benchmark_prompts = [
            "Explain quantum computing in simple terms.",
            "Write a Python function to calculate fibonacci numbers.",
            "What are the main causes of climate change?",
            "Describe the process of photosynthesis.",
            "Write a short story about a robot learning to paint."
        ]
        
        messages = [
            PromptMessage(
                role="user", 
                content=TextContent(
                    type="text",
                    text=f"Benchmark the model at: {model_path}\n\n"
                         f"Test it with these prompts:\n" + 
                         "\n".join(f"{i+1}. {p}" for i, p in enumerate(benchmark_prompts)) +
                         "\n\nProvide the model's response to each prompt and evaluate quality."
                )
            )
        ]
        
        return Prompt(
            name=name,
            description="Benchmark model performance",
            messages=messages
        )
    
    else:
        raise ValueError(f"Unknown prompt: {name}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Multi-Model MCP Server")
    parser.add_argument(
        "--settings",
        default="settings.json",
        help="Path to settings file"
    )
    
    args = parser.parse_args()
    
    # Initialize server with settings
    global mcp_server
    mcp_server = MultiModelMCPServer(args.settings)
    
    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="llm-train-models",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None
                )
            )
        )


if __name__ == "__main__":
    asyncio.run(main())