#!/usr/bin/env python3
"""
Install MCP server dependencies

This script installs the required packages for the MCP server functionality.
"""

import subprocess
import sys
import os
from pathlib import Path


def install_mcp_dependencies():
    """Install MCP server dependencies."""
    print("Installing MCP server dependencies...")
    
    # Check if we're in a virtual environment
    venv_path = None
    if hasattr(sys, 'prefix') and sys.prefix != sys.base_prefix:
        # We're in a virtual environment
        if sys.platform == "win32":
            venv_path = Path(sys.prefix) / "Scripts" / "pip.exe"
        else:
            venv_path = Path(sys.prefix) / "bin" / "pip"
    
    # Use appropriate pip command
    if venv_path and venv_path.exists():
        pip_cmd = [str(venv_path)]
    else:
        pip_cmd = [sys.executable, "-m", "pip"]
    
    # Install packages
    packages = [
        "mcp>=1.0.0",
        "uvloop>=0.19.0; sys_platform != 'win32'"
    ]
    
    try:
        for package in packages:
            print(f"Installing {package}...")
            result = subprocess.run(
                pip_cmd + ["install", package],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓ {package} installed successfully")
        
        print("\n✅ All MCP dependencies installed successfully!")
        print("\nTo use the MCP server:")
        print("1. Configure it via Tools → MCP Server Config")
        print("2. Run: python mcp_server.py")
        print("3. Add the configuration to Claude Desktop")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Command error: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def check_mcp_installation():
    """Check if MCP is already installed."""
    try:
        import mcp
        print(f"✓ MCP is already installed (version: {mcp.__version__})")
        return True
    except ImportError:
        print("ℹ MCP is not installed")
        return False


def main():
    """Main installation function."""
    print("=== LLM_Train MCP Server Setup ===\n")
    
    if check_mcp_installation():
        response = input("MCP is already installed. Reinstall? (y/N): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return
    
    success = install_mcp_dependencies()
    
    if success:
        print("\n🎉 Setup complete! You can now use the MCP server functionality.")
    else:
        print("\n💥 Setup failed. Please check the error messages above.")
        print("You may need to run this script as administrator or check your internet connection.")


if __name__ == "__main__":
    main()