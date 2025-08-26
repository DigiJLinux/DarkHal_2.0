# LLM_Train Remote Control

A standalone GUI application for remotely controlling and managing LLM_Train MCP servers. This allows you to connect to a running MCP server from a separate application and perform model management and inference operations remotely.

## Features

### 🔗 **Connection Management**
- Connect to local or remote MCP servers
- Real-time connection status monitoring
- Automatic reconnection handling
- Server path configuration and browsing

### 🤖 **Model Management**
- List all available models from the server
- Remote model loading with configuration
- Context size and GPU layer settings
- Current model status display
- Model unloading capabilities

### 💬 **Inference Interface**
- Text generation with configurable parameters
- Chat mode for conversational interactions
- Adjustable temperature and max tokens
- Real-time output display
- Chat history management

### 📊 **System Monitoring**
- Real-time system information display
- GPU acceleration status
- Platform and architecture details
- Performance metrics

### 📝 **Logging & Debugging**
- Comprehensive operation logging
- Error tracking and display
- Log saving functionality
- Connection event monitoring

## Quick Start

### Option 1: Batch File (Windows)
```bash
# Double-click launch_remote_control.bat
```

### Option 2: Direct Python Execution
```bash
python remotecontrol.py
```

## Requirements

- **Python 3.7+** with tkinter support
- **Running MCP Server** (from main LLM_Train application)
- **Network access** to the MCP server (if remote)

## Usage Guide

### 1. Starting the Remote Control

1. **Launch the application** using one of the methods above
2. The GUI will open with the connection panel at the top

### 2. Connecting to MCP Server

1. **Specify server path**: 
   - Default: `mcp_server.py` (local)
   - Browse to select different server script
   - Can be local file or network path

2. **Click "Connect"** button
3. **Connection status** will show:
   - 🔴 **Disconnected** (Red): Not connected
   - 🟢 **Connected** (Green): Successfully connected

### 3. Model Management

**Models Tab:**
- **View available models**: Automatically populated on connection
- **Select model**: Click on model in the list
- **Configure parameters**:
  - Context Size: 512 - 32768 tokens
  - GPU Layers: 0 - 100 layers
- **Load model**: Click "Load Model" button
- **Monitor status**: Current model display shows loaded model

### 4. Text Generation

**Inference Tab:**
- **Enter prompt**: Type or paste text in the input area
- **Configure generation**:
  - Max Tokens: 1 - 8192
  - Temperature: 0.0 - 2.0
- **Generate**: Click "Generate" button
- **View output**: Results appear in the output area
- **Chat mode**: Enable for conversational interface

### 5. System Information

**System Tab:**
- **View system details**: Platform, architecture, acceleration
- **Monitor GPU status**: CUDA, ROCm, Metal availability
- **Check performance**: Current acceleration method
- **Refresh**: Update information in real-time

### 6. Logging

**Log Tab:**
- **Monitor operations**: All actions are logged with timestamps
- **Error tracking**: Errors are highlighted and detailed
- **Save logs**: Export logs to text file for debugging
- **Clear logs**: Reset log display

## Advanced Configuration

### Server Connection Options

**Local Server:**
```
Server Path: mcp_server.py
```

**Custom Path:**
```
Server Path: C:\path\to\your\mcp_server.py
```

**Network Server (if supported):**
```
Server Path: \\network\path\mcp_server.py
```

### Model Configuration

**High-Performance Setup:**
- Context Size: 8192+
- GPU Layers: Maximum supported
- Temperature: 0.1-0.3 for focused responses

**Balanced Setup:**
- Context Size: 4096
- GPU Layers: Auto-detected optimum
- Temperature: 0.7 for creative responses

**CPU-Only Setup:**
- Context Size: 2048
- GPU Layers: 0
- Temperature: 0.5-1.0

### Generation Parameters

**Creative Writing:**
- Max Tokens: 1024+
- Temperature: 0.8-1.2

**Code Generation:**
- Max Tokens: 512
- Temperature: 0.1-0.3

**Question Answering:**
- Max Tokens: 256
- Temperature: 0.3-0.7

## Troubleshooting

### Connection Issues

**"Server file not found"**
- Verify the server path is correct
- Ensure the MCP server file exists
- Check file permissions

**"Connection failed"**
- Ensure the MCP server is not already running
- Check if the server script is executable
- Verify Python dependencies are installed

**"Disconnected unexpectedly"**
- Check server logs for errors
- Verify system resources are available
- Restart both applications

### Model Loading Issues

**"No models found"**
- Ensure model library is configured in main application
- Verify model files exist in specified directories
- Check library settings and scan depth

**"Failed to load model"**
- Verify model file is not corrupted
- Check available system memory
- Reduce context size or GPU layers

**"Out of memory"**
- Reduce context size
- Lower GPU layers
- Close other applications

### Generation Issues

**"No model loaded"**
- Load a model first using the Models tab
- Verify model loaded successfully
- Check current model display

**"Generation timeout"**
- Reduce max tokens
- Simplify the prompt
- Check system resources

**"Invalid parameters"**
- Verify temperature is between 0.0-2.0
- Ensure max tokens is reasonable
- Check prompt is not empty

## Technical Details

### MCP Protocol
- Uses JSON-RPC 2.0 over stdin/stdout
- Asynchronous request/response handling
- Automatic request ID management
- Error handling and recovery

### Threading Model
- Main UI thread for interface
- AsyncIO event loop for MCP communication
- Background threads for I/O operations
- Thread-safe callback system

### Security Considerations
- Local process communication only
- No network ports exposed
- Input validation on all parameters
- Error sanitization in logs

## Integration Examples

### Automated Workflows
```python
# Example: Batch text generation
prompts = ["Explain AI", "Code a function", "Write a story"]
for prompt in prompts:
    # Use remote control to generate text
    # Save results to files
```

### API Integration
```python
# Example: Integration with other tools
remote_control = RemoteControlClient()
remote_control.connect("mcp_server.py")
result = remote_control.generate("Your prompt here")
```

### Monitoring Scripts
```python
# Example: System monitoring
while True:
    system_info = remote_control.get_system_info()
    log_performance_metrics(system_info)
    time.sleep(60)
```

## Support and Development

### Extending Functionality
- Add new MCP tool integrations
- Implement custom inference modes
- Create automation scripts
- Build monitoring dashboards

### Contributing
- Follow Python coding standards
- Add comprehensive logging
- Include error handling
- Write unit tests

### Reporting Issues
- Include full log output
- Specify system configuration
- Provide reproduction steps
- Attach relevant files

## License

This remote control application is part of the LLM_Train project and follows the same licensing terms as the main application.