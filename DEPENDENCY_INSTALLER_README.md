# Windows Dependency Installer for LLM_Train

This dependency installer helps you automatically install common software packages required for LLM_Train on Windows using the Chocolatey package manager.

## Quick Start

### Option 1: Batch File (Recommended)
1. Double-click `install_dependencies.bat`
2. Grant administrator permissions when prompted
3. Select packages to install in the GUI

### Option 2: PowerShell Script
1. Right-click `install_dependencies.ps1` → "Run with PowerShell"
2. Grant administrator permissions when prompted
3. Select packages to install in the GUI

### Option 3: Direct Python Execution
```bash
# Run as administrator
python windows_dependency_installer.py
```

## What Gets Installed

### Essential Packages (Auto-selected)
- **Git** - Version control system (required for repository cloning)
- **Python 3** - Python programming language (if not already installed)
- **7-Zip** - File archiver for extracting downloads
- **Visual C++ Redistributables** - Microsoft runtime libraries

### Development Tools
- **Visual Studio Code** - Advanced code editor with Python support
- **Notepad++** - Enhanced text editor

### GPU Acceleration
- **CUDA Toolkit** - NVIDIA CUDA development toolkit for GPU acceleration
- **NVIDIA Display Driver** - Latest NVIDIA graphics drivers

### System Utilities
- **Wget** - Command-line downloader
- **cURL** - Data transfer tool
- **PowerToys** - Windows system utilities

### Runtimes
- **.NET Runtime** - Microsoft .NET framework

### Optional Tools
- **WinRAR** - Alternative file archiver
- **Firefox** - Web browser
- **VLC Media Player** - Media player

## System Requirements

- **Windows 10/11** (Windows 8.1 may work but is not tested)
- **Administrator privileges** (required for Chocolatey and package installation)
- **Internet connection** (for downloading packages)
- **Python 3.7+** (for running the installer GUI)

## Features

### Chocolatey Integration
- Automatically installs Chocolatey if not present
- Uses Chocolatey's robust package management
- Handles dependencies automatically

### Smart Package Selection
- **Select Essential** - Chooses only required packages
- **Select All** - Selects all available packages
- **Custom Selection** - Pick individual packages

### Installation Monitoring
- Real-time installation log
- Progress tracking
- Success/failure reporting
- Package status checking

### System Status Checks
- Administrator privilege detection
- Chocolatey installation status
- Individual package installation status

## Troubleshooting

### "Python not found" Error
1. Install Python from https://python.org/downloads/
2. During installation, check "Add Python to PATH"
3. Restart your command prompt/PowerShell

### "Administrator privileges required" Error
1. Right-click the batch file → "Run as administrator"
2. Or open Command Prompt as administrator and run manually

### "Chocolatey installation failed" Error
1. Ensure you're running as administrator
2. Check your internet connection
3. Temporarily disable antivirus software during installation
4. Check Windows execution policy: `Set-ExecutionPolicy RemoteSigned`

### Package Installation Failures
1. Check the installation log for specific error messages
2. Try installing packages individually
3. Ensure sufficient disk space
4. Check for conflicting software

### Network/Firewall Issues
1. Ensure Chocolatey URLs are not blocked:
   - https://community.chocolatey.org/
   - https://packages.chocolatey.org/
2. Configure proxy settings if behind corporate firewall
3. Temporarily disable firewall/antivirus

## Manual Installation

If the automatic installer fails, you can install Chocolatey manually:

1. Open PowerShell as Administrator
2. Run:
   ```powershell
   Set-ExecutionPolicy Bypass -Scope Process -Force;
   [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072;
   iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
   ```
3. Then install packages manually:
   ```powershell
   choco install git python 7zip vcredist-all -y
   choco install cuda nvidia-display-driver -y  # For GPU support
   ```

## Package Descriptions

### Why Each Package?

- **Git**: Required for cloning model repositories and version control
- **Python**: Core runtime for LLM_Train (if system Python is outdated)
- **7-Zip**: Many model files are compressed and need extraction
- **Visual C++ Redistributables**: Required by many Python packages and binaries
- **CUDA Toolkit**: Enables GPU acceleration for faster model inference
- **NVIDIA Drivers**: Latest drivers for optimal GPU performance
- **Visual Studio Code**: Best IDE for Python development and debugging
- **Wget/cURL**: Alternative download tools for model files
- **PowerToys**: Useful Windows utilities for power users

## Security Notes

- All packages are installed from official Chocolatey community repository
- Chocolatey packages are maintained by the community and Microsoft
- Administrator privileges are required only for system-wide installation
- No personal data is collected or transmitted

## Support

If you encounter issues:

1. Check the installation log for error messages
2. Search for the specific error on Chocolatey community forums
3. Try installing individual packages manually
4. Ensure your Windows is up to date

## License

This installer uses Chocolatey (Apache 2.0 License) and installs various packages with their respective licenses. Please review individual package licenses as needed.