# DarkHal 2.0 User Manual

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Main Interface](#main-interface)
4. [Model Management](#model-management)
5. [Agent Mode](#agent-mode)
6. [Advanced Features](#advanced-features)
7. [Troubleshooting](#troubleshooting)
8. [Keyboard Shortcuts](#keyboard-shortcuts)
9. [Command Reference](#command-reference)
10. [FAQ](#faq)

---

## Introduction

DarkHal 2.0 is an advanced AI model management platform that provides comprehensive tools for loading, running, and interacting with Large Language Models (LLMs). This manual covers all features and capabilities of the platform.

### System Requirements

**Minimum Requirements:**
- Windows 10/11, Linux (Ubuntu 20.04+), or macOS 11+
- Python 3.8 or higher
- 8GB RAM
- 20GB free disk space
- Internet connection for downloading models

**Recommended Requirements:**
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM
- 100GB+ free disk space for models
- High-speed internet for model downloads

---

## Getting Started

### First Launch

1. **Start DarkHal**:
   ```bash
   python main.py --gui
   ```

2. **Initial Setup**:
   - The splash screen will appear showing system information
   - Configure your models directory via Settings menu
   - Set up HuggingFace API token if you plan to download models

3. **Load Your First Model**:
   - Click "Browse Model" to select a model file
   - Or use "Browse Folder" to select a model directory
   - Click "Load Model" to initialize

### Understanding the Interface

The main window consists of several tabs:
- **Run**: Chat interface and model loading
- **Model Library**: Browse and manage local models
- **HuggingFace**: Download models from HuggingFace Hub
- **Downloads**: Monitor active downloads
- **MCP**: Model Context Protocol server
- **Converter**: Convert between model formats
- **Chess**: Specialized chess engine interface

---

## Main Interface

### Run Tab

The Run tab is your primary workspace for interacting with models.

#### Model Selection Panel

**Model Path Input**:
- Enter the full path to your model file or directory
- Supports drag-and-drop from file explorer
- Auto-completes recently used models

**Browse Model Button**:
- Opens file dialog to select model files
- Filters: GGUF, SafeTensors, PyTorch, GPTQ, AWQ, EXL2
- Shows all supported formats

**Browse Folder Button**:
- Select directories containing model files
- Useful for HuggingFace format models
- Auto-detects config.json

**Load Model Button**:
- Initializes the selected model
- Shows loading progress
- Displays model information when complete

#### Chat Interface

**Chat Mode Options**:
- **Stream Output**: Shows text as it's generated
- **Chess Mode**: Enables ChessGPT for chess moves
- **Agent Mode**: Enables system command execution

**Input Area**:
- Multi-line text input
- Supports Ctrl+Enter for sending
- Maintains conversation history

**Output Display**:
- Shows conversation with "You:" and "Assistant:" prefixes
- Auto-scrolls to latest message
- Supports text selection and copying

**Control Buttons**:
- **Send (Chat)**: Submit your message
- **Stop**: Interrupt generation
- **Clear Output**: Clear conversation display
- **Clear History**: Reset conversation context

### Model Settings Tab

#### Basic Settings

**Context Size (n_ctx)**:
- Range: 512 to 32768 tokens
- Default: 4096
- Higher values use more memory but allow longer conversations

**GPU Layers**:
- Range: 0 to model layer count
- 0 = CPU only
- Higher values offload more to GPU

**Max Tokens**:
- Maximum tokens to generate
- Range: 1 to context size
- Default: 2048

#### Advanced Loading Options

**Quantization**:
- `none`: Full precision (FP16/FP32)
- `4bit`: ~75% memory savings
- `8bit`: ~50% memory savings
- `gptq`: Pre-quantized GPTQ format
- `awq`: Pre-quantized AWQ format
- `exl2`: Pre-quantized EXL2 format

**Device Strategy**:
- `auto`: Automatic distribution
- `force_gpu`: All layers on GPU
- `balanced_split`: Split between CPU/GPU
- `cpu_only`: CPU processing only

**GPU Memory Limit**:
- Maximum VRAM to use (in GB)
- Used with balanced_split strategy
- Prevents out-of-memory errors

#### Sampling Parameters

**Temperature** (0.0 - 2.0):
- Controls randomness
- 0.0 = Deterministic
- 0.7 = Balanced (default)
- 1.5+ = Very creative

**Top-p** (0.0 - 1.0):
- Nucleus sampling threshold
- 0.9 = Default
- Lower values = More focused

**Repetition Penalty** (1.0 - 2.0):
- Reduces repetitive text
- 1.0 = No penalty
- 1.1 = Light penalty (default)

**Min-p** (0.0 - 1.0):
- Minimum probability threshold
- 0.0 = Disabled (default)

**Typical-p** (0.0 - 1.0):
- Typical sampling threshold
- 1.0 = Disabled (default)

---

## Model Management

### Supported Formats

DarkHal 2.0 supports multiple model formats:

| Format | Extension | Use Case | Pros | Cons |
|--------|-----------|----------|------|------|
| **GGUF** | `.gguf` | CPU/GPU hybrid | Fast loading, efficient | Limited to llama.cpp models |
| **SafeTensors** | `.safetensors` | HuggingFace models | Secure, fast | Larger file sizes |
| **PyTorch** | `.bin`, `.pt`, `.pth` | Research models | Flexible | Slower loading |
| **GPTQ** | `*gptq*.safetensors` | GPU inference | 4-bit quantized | GPU required |
| **AWQ** | `*awq*.safetensors` | GPU inference | Optimized quantization | GPU required |
| **EXL2** | `.exl2` | ExLlamaV2 | Very fast | Specific hardware needs |

### Model Library Tab

The Model Library provides comprehensive model management:

**Features**:
- Automatic scanning of model directories
- Metadata extraction (parameters, architecture)
- Search by name, type, or tags
- Size and modification date display
- One-click loading

**Using the Library**:
1. Set your models directory in Settings
2. Click "Scan" to index models
3. Use search box to filter
4. Double-click to load model

### Downloading Models

#### HuggingFace Tab

**Search and Browse**:
- Enter model name or organization
- Browse trending models
- Filter by task type
- View model cards

**Download Process**:
1. Enter model ID (e.g., "meta-llama/Llama-2-7b")
2. Click "Get File List"
3. Select files to download
4. Click "Start Download"
5. Monitor progress in Downloads tab

**File Selection**:
- Use checkboxes to select specific files
- "Select All" for complete model
- Size estimates shown for each file
- Automatic resume on failure

#### Downloads Tab

**Download Management**:
- Grouped display by model
- Individual file progress
- Speed and time estimates
- Pause/resume capability
- Automatic retry on failure

**Controls**:
- Collapse/expand model groups
- Cancel individual files
- Clear completed downloads
- Set bandwidth limits

---

## Agent Mode

### ⚠️ WARNING

Agent Mode grants the AI unrestricted system access. Only enable with trusted models and full understanding of risks.

### Enabling Agent Mode

1. Load any model
2. Check "🤖 Agent Mode (SYSTEM ACCESS)"
3. Confirm security warning
4. Agent mode indicator shows "ACTIVE"

### Capabilities

**System Control**:
- Execute shell commands
- Run PowerShell scripts
- Execute Bash commands
- Launch applications

**File Operations**:
- Read any file
- Write/create files
- Delete files
- List directories

**Application Control**:
- Open programs (Word, Notepad, etc.)
- Control mouse movement
- Send keyboard input
- Automate workflows

**Programming**:
- Execute Python code
- Run scripts
- Install packages
- Compile code

### Example Commands

**Opening Applications**:
```
"Open PowerShell"
"Launch Microsoft Word"
"Start notepad"
"Open calculator"
```

**File Operations**:
```
"List files in current directory"
"Create a file called test.txt with 'Hello World'"
"Read the contents of config.json"
"Delete temporary files"
```

**System Commands**:
```
"Show system information"
"Check disk space"
"List running processes"
"Create a new folder called Projects"
```

**Document Creation**:
```
"Open Word and create a document about Python"
"Create an Apache server setup guide"
"Write a bash script to backup files"
```

### Safety Guidelines

1. **Review Commands**: Always review AI-generated commands before execution
2. **Backup Data**: Keep backups before allowing file operations
3. **Limit Scope**: Use specific requests rather than broad permissions
4. **Monitor Activity**: Watch the output for unexpected behavior
5. **Disable When Done**: Turn off Agent Mode after use

---

## Advanced Features

### Chat Templates

Chat templates format conversations for different model architectures.

**Loading Templates**:
1. Click "Load" next to Chat Template dropdown
2. Select JSON file with templates
3. Choose template from dropdown

**Adding Custom Templates**:
1. Click "Add" button
2. Define template format
3. Set special tokens
4. Save to templates file

**Template Format**:
```json
{
  "name": "llama3",
  "template": "<|begin_of_text|>{% for message in messages %}...",
  "bos_token": "<|begin_of_text|>",
  "eos_token": "<|eot_id|>"
}
```

### Model Conversion

The Converter tab allows format transformation:

**Supported Conversions**:
- GGUF → SafeTensors
- SafeTensors → GGUF
- PyTorch → GGUF
- GPTQ → GGUF

**Conversion Process**:
1. Select source model
2. Choose target format
3. Set quantization options
4. Click "Convert"
5. Monitor progress

### MCP Server

Model Context Protocol enables remote access:

**Starting Server**:
1. Go to MCP tab
2. Configure port and settings
3. Click "Start Server"
4. Note the connection URL

**Connecting Clients**:
- Claude Desktop integration
- Remote control GUI
- Custom API clients
- Web interfaces

**Available Endpoints**:
- `/list_models` - Available models
- `/load_model` - Load specific model
- `/generate` - Text generation
- `/chat` - Conversation mode

### Chess Mode

Specialized interface for chess AI:

**Features**:
- FEN notation support
- Move generation
- Position evaluation
- Game analysis

**Using Chess Mode**:
1. Enable "Chess Mode" checkbox
2. Enter position in FEN format
3. Request move analysis
4. Get UCI format moves

---

## Troubleshooting

### Common Issues

#### Model Won't Load

**Symptoms**: Error message when loading model

**Solutions**:
- Verify file path is correct
- Check file isn't corrupted
- Ensure sufficient RAM/VRAM
- Try reducing GPU layers
- Lower context size

#### Out of Memory

**Symptoms**: Application crashes or freezes

**Solutions**:
- Use quantized models (4-bit/8-bit)
- Reduce context size
- Lower GPU layers
- Use CPU-only mode
- Close other applications

#### Slow Generation

**Symptoms**: Very slow text generation

**Solutions**:
- Enable GPU acceleration
- Increase GPU layers
- Use smaller models
- Reduce context size
- Check CPU/GPU usage

#### Download Failures

**Symptoms**: Downloads fail or hang

**Solutions**:
- Check internet connection
- Verify HuggingFace token
- Clear download cache
- Use VPN if blocked
- Try different mirror

### Error Messages

**"CUDA out of memory"**:
- Reduce GPU layers
- Use smaller batch size
- Enable memory efficient attention
- Use quantized model

**"Model file not found"**:
- Check file path
- Verify file exists
- Check permissions
- Try absolute path

**"Invalid model format"**:
- Verify file format
- Check model compatibility
- Update DarkHal
- Try conversion

**"Token limit exceeded"**:
- Reduce input length
- Lower max tokens
- Clear conversation history
- Increase context size

---

## Keyboard Shortcuts

### Global Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+N` | New conversation |
| `Ctrl+O` | Open model |
| `Ctrl+S` | Save conversation |
| `Ctrl+Q` | Quit application |
| `F1` | Open help |
| `F5` | Refresh model list |

### Chat Interface

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Send message |
| `Ctrl+L` | Clear output |
| `Ctrl+H` | Clear history |
| `Esc` | Stop generation |
| `Ctrl+C` | Copy selected text |
| `Ctrl+A` | Select all |

### Model Library

| Shortcut | Action |
|----------|--------|
| `Ctrl+F` | Focus search |
| `Enter` | Load selected model |
| `Delete` | Remove from library |
| `F5` | Rescan directory |

---

## Command Reference

### CLI Arguments

```bash
python main.py [options]
```

**Options**:
- `--gui` - Launch GUI mode (default)
- `--model PATH` - Model file path
- `--prompt TEXT` - Initial prompt
- `--stream` - Enable streaming
- `--n_ctx N` - Context size
- `--n_gpu_layers N` - GPU layers
- `--lora PATH` - LoRA adapter path

### Configuration Files

**settings.json**:
```json
{
  "paths": {
    "models_directory": "./models",
    "download_directory": "./downloads"
  },
  "model_settings": {
    "default_n_ctx": 4096,
    "default_n_gpu_layers": 0,
    "stream_by_default": true,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1
  }
}
```

**HUGGINGFACE.env**:
```
HF_API_KEY=your_token_here
HF_HOME=./models/huggingface
```

---

## FAQ

### General Questions

**Q: What models work with DarkHal?**
A: Any model in GGUF, SafeTensors, PyTorch, GPTQ, AWQ, or EXL2 format. Most HuggingFace models are compatible.

**Q: How much RAM do I need?**
A: Depends on model size. 7B models need ~8GB, 13B need ~16GB, 70B need ~64GB. Quantization reduces requirements.

**Q: Can I run without GPU?**
A: Yes, CPU-only mode works but is slower. Use GGUF models with 0 GPU layers.

**Q: Is my data private?**
A: Yes, all processing is local. No data is sent to external servers unless using HuggingFace downloads.

### Model Questions

**Q: What's the difference between formats?**
A: GGUF is optimized for CPU/GPU hybrid. SafeTensors is HuggingFace standard. GPTQ/AWQ/EXL2 are quantized for GPU.

**Q: How do I choose quantization?**
A: 4-bit saves most memory with slight quality loss. 8-bit balances quality and size. None uses full precision.

**Q: Why is generation slow?**
A: Check GPU usage, reduce context size, use quantized models, or enable more GPU layers.

**Q: Can I use multiple models?**
A: One model at a time in current version. Switch models by loading different ones.

### Agent Mode Questions

**Q: Is Agent Mode safe?**
A: Agent Mode grants full system access. Only use with trusted models and review commands.

**Q: What can Agent Mode do?**
A: Execute any system command, control applications, manage files, run code, automate tasks.

**Q: How do I limit Agent Mode?**
A: Currently all-or-nothing. Future versions will have granular permissions.

**Q: Can Agent Mode access internet?**
A: Yes, through system commands like curl or wget, and Python's requests library.

### Troubleshooting Questions

**Q: Download keeps failing?**
A: Check internet, verify HF token, try VPN, clear cache, or download manually.

**Q: Model won't load?**
A: Verify path, check format, ensure enough memory, try different quantization.

**Q: Getting CUDA errors?**
A: Update GPU drivers, check CUDA version, reduce GPU layers, or use CPU mode.

**Q: Application crashes?**
A: Check error logs, reduce memory usage, update dependencies, file bug report.

---

## Support

### Getting Help

**Documentation**: [https://darkhal.readthedocs.io](https://darkhal.readthedocs.io)
**GitHub Issues**: [https://github.com/darkhal/issues](https://github.com/darkhal/issues)
**Discussions**: [https://github.com/darkhal/discussions](https://github.com/darkhal/discussions)
**Email Support**: support@darkhal.ai

### Reporting Bugs

Include:
1. System information (OS, GPU, RAM)
2. Model details (format, size, source)
3. Error messages and logs
4. Steps to reproduce
5. Screenshots if applicable

### Contributing

We welcome contributions! See CONTRIBUTING.md for guidelines.

---

## Appendices

### A. Model Compatibility Matrix

| Model Family | GGUF | SafeTensors | GPTQ | AWQ | EXL2 |
|--------------|------|-------------|------|-----|------|
| Llama 2/3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| Mistral | ✅ | ✅ | ✅ | ✅ | ✅ |
| Mixtral | ✅ | ✅ | ✅ | ✅ | ❌ |
| Qwen | ✅ | ✅ | ✅ | ✅ | ✅ |
| Yi | ✅ | ✅ | ✅ | ✅ | ✅ |
| Gemma | ✅ | ✅ | ❌ | ❌ | ❌ |

### B. Performance Benchmarks

| Model | Format | GPU | Speed (tok/s) | Memory |
|-------|--------|-----|---------------|---------|
| Llama-7B | GGUF Q4 | RTX 3060 | 45 | 4.5GB |
| Llama-13B | GGUF Q4 | RTX 3090 | 35 | 8.5GB |
| Mistral-7B | GPTQ | RTX 4090 | 65 | 5.0GB |
| Mixtral-8x7B | AWQ | A100 | 25 | 24GB |

### C. Glossary

**Context Size**: Maximum tokens the model can process at once
**GPU Layers**: Model layers offloaded to GPU for acceleration
**Quantization**: Reducing model precision to save memory
**LoRA**: Low-Rank Adaptation for model fine-tuning
**Token**: Basic unit of text (roughly 0.75 words)
**VRAM**: Video RAM on graphics card
**Streaming**: Showing text as it's generated
**KV Cache**: Key-value cache for faster inference

---

*DarkHal 2.0 User Manual - Version 2.0.0*
*Last Updated: January 2025*