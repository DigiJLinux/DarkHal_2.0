#!/usr/bin/env python3
"""
Simple PNG to ICO converter using Windows tools
"""

import os
import sys
import subprocess
from pathlib import Path

def png_to_ico_windows(png_path, ico_path):
    """Convert PNG to ICO using Windows PowerShell and .NET Image classes."""
    
    png_path = str(Path(png_path).resolve())
    ico_path = str(Path(ico_path).resolve())
    
    # PowerShell script to convert PNG to ICO
    ps_script = f'''
Add-Type -AssemblyName System.Drawing
$png = [System.Drawing.Image]::FromFile("{png_path}")
$ico = New-Object System.Drawing.Icon($png.GetHbitmap(), 256, 256)
$ico.Save("{ico_path}")
$ico.Dispose()
$png.Dispose()
'''
    
    try:
        # Try PowerShell method first
        result = subprocess.run([
            'powershell', '-Command', ps_script
        ], capture_output=True, text=True, shell=True)
        
        if result.returncode == 0 and os.path.exists(ico_path):
            print(f"Successfully converted {png_path} to {ico_path}")
            return True
        else:
            print(f"PowerShell conversion failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False

def main():
    png_file = "assets/Halico.png"
    ico_file = "assets/Halico.ico"
    
    if not os.path.exists(png_file):
        print(f"PNG file not found: {png_file}")
        return 1
    
    print(f"Converting {png_file} to {ico_file}...")
    
    if png_to_ico_windows(png_file, ico_file):
        print("Conversion completed successfully!")
        return 0
    else:
        print("Conversion failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())