#!/usr/bin/env python3
"""
DarkHal 2.0 Cross-Platform Build Script
Supports: Windows (x64, ARM64), Linux (x64, ARM64), macOS (x64, ARM64)
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path

# Build configurations for different platforms/architectures
BUILD_CONFIGS = {
    'windows-x64': {
        'platform': 'win32',
        'arch': 'x64',
        'pyinstaller_args': [
            '--onefile',
            '--windowed',
            '--icon', 'assets/logo.ico',
            '--add-data', 'assets/*;assets',
            '--add-data', 'llm_runtime;llm_runtime',
            '--hidden-import', 'tkinter',
            '--hidden-import', 'tkinter.ttk',
            '--hidden-import', 'torch',
            '--hidden-import', 'transformers',
            '--target-architecture', 'x86_64'
        ],
        'output_name': 'DarkHal-2.0-windows-x64.exe'
    },
    'windows-arm64': {
        'platform': 'win32',
        'arch': 'arm64',
        'pyinstaller_args': [
            '--onefile',
            '--windowed',
            '--icon', 'assets/logo.ico',
            '--add-data', 'assets/*;assets',
            '--add-data', 'llm_runtime;llm_runtime',
            '--hidden-import', 'tkinter',
            '--hidden-import', 'tkinter.ttk',
            '--hidden-import', 'torch',
            '--hidden-import', 'transformers',
            '--target-architecture', 'arm64'
        ],
        'output_name': 'DarkHal-2.0-windows-arm64.exe'
    },
    'linux-x64': {
        'platform': 'linux',
        'arch': 'x64',
        'pyinstaller_args': [
            '--onefile',
            '--add-data', 'assets/*:assets',
            '--add-data', 'llm_runtime:llm_runtime',
            '--hidden-import', 'tkinter',
            '--hidden-import', 'tkinter.ttk',
            '--hidden-import', 'torch',
            '--hidden-import', 'transformers',
            '--target-architecture', 'x86_64'
        ],
        'output_name': 'DarkHal-2.0-linux-x64'
    },
    'linux-arm64': {
        'platform': 'linux',
        'arch': 'arm64',
        'pyinstaller_args': [
            '--onefile',
            '--add-data', 'assets/*:assets',
            '--add-data', 'llm_runtime:llm_runtime',
            '--hidden-import', 'tkinter',
            '--hidden-import', 'tkinter.ttk',
            '--hidden-import', 'torch',
            '--hidden-import', 'transformers',
            '--target-architecture', 'arm64'
        ],
        'output_name': 'DarkHal-2.0-linux-arm64'
    },
    'macos-x64': {
        'platform': 'darwin',
        'arch': 'x64',
        'pyinstaller_args': [
            '--onefile',
            '--windowed',
            '--icon', 'assets/logo.icns',
            '--add-data', 'assets/*:assets',
            '--add-data', 'llm_runtime:llm_runtime',
            '--hidden-import', 'tkinter',
            '--hidden-import', 'tkinter.ttk',
            '--hidden-import', 'torch',
            '--hidden-import', 'transformers',
            '--target-architecture', 'x86_64'
        ],
        'output_name': 'DarkHal-2.0-macos-x64'
    },
    'macos-arm64': {
        'platform': 'darwin',
        'arch': 'arm64',
        'pyinstaller_args': [
            '--onefile',
            '--windowed',
            '--icon', 'assets/logo.icns',
            '--add-data', 'assets/*:assets',
            '--add-data', 'llm_runtime:llm_runtime',
            '--hidden-import', 'tkinter',
            '--hidden-import', 'tkinter.ttk',
            '--hidden-import', 'torch',
            '--hidden-import', 'transformers',
            '--target-architecture', 'arm64'
        ],
        'output_name': 'DarkHal-2.0-macos-arm64'
    }
}

def get_current_platform():
    """Detect current platform and architecture"""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == 'windows':
        platform_name = 'windows'
    elif system == 'linux':
        platform_name = 'linux'
    elif system == 'darwin':
        platform_name = 'macos'
    else:
        platform_name = system
    
    if machine in ['amd64', 'x86_64']:
        arch = 'x64'
    elif machine in ['arm64', 'aarch64']:
        arch = 'arm64'
    else:
        arch = machine
    
    return f"{platform_name}-{arch}"

def install_pyinstaller():
    """Install PyInstaller if not present"""
    try:
        import PyInstaller
        print("✓ PyInstaller already installed")
        return True
    except ImportError:
        print("Installing PyInstaller...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyinstaller'])
            print("✓ PyInstaller installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install PyInstaller: {e}")
            return False

def clean_build_files():
    """Clean previous build files"""
    dirs_to_remove = ['build', 'dist']
    files_to_remove = ['*.spec']
    
    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            import shutil
            shutil.rmtree(dir_name)
            print(f"Cleaned {dir_name}/ directory")
    
    import glob
    for pattern in files_to_remove:
        for file in glob.glob(pattern):
            os.remove(file)
            print(f"Cleaned {file}")

def build_executable(target_platform, clean=True):
    """Build executable for specified platform"""
    if target_platform not in BUILD_CONFIGS:
        print(f"✗ Unknown target platform: {target_platform}")
        print(f"Available platforms: {', '.join(BUILD_CONFIGS.keys())}")
        return False
    
    config = BUILD_CONFIGS[target_platform]
    
    print(f"Building DarkHal 2.0 for {target_platform}...")
    print(f"Platform: {config['platform']}, Architecture: {config['arch']}")
    
    # Check if we have required files
    if not os.path.exists('main.py'):
        print("✗ main.py not found. Please run from the correct directory.")
        return False
    
    if not os.path.exists('llm_runtime'):
        print("✗ llm_runtime directory not found")
        return False
    
    # Clean previous builds
    if clean:
        clean_build_files()
    
    # Build PyInstaller command
    cmd = [
        'pyinstaller',
        '--name', config['output_name'].replace('.exe', '').replace('.app', ''),
        '--distpath', f'dist/{target_platform}',
        '--workpath', f'build/{target_platform}',
        '--specpath', f'build/{target_platform}'
    ] + config['pyinstaller_args'] + ['main.py']
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Build completed successfully")
        
        # Check output file
        expected_output = f"dist/{target_platform}/{config['output_name']}"
        # PyInstaller might create without extension on some platforms
        possible_outputs = [
            expected_output,
            expected_output.replace('.exe', ''),
            f"dist/{target_platform}/{config['output_name'].replace('.exe', '').replace('.app', '')}"
        ]
        
        output_file = None
        for possible in possible_outputs:
            if os.path.exists(possible):
                output_file = possible
                break
        
        if output_file:
            # Rename to correct name if needed
            if output_file != expected_output and expected_output.endswith('.exe'):
                os.rename(output_file, expected_output)
                output_file = expected_output
            
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            print(f"✓ Executable created: {output_file}")
            print(f"  File size: {file_size:.1f} MB")
            return True
        else:
            print("✗ Executable not found after build")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"✗ Build failed: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def build_all_platforms():
    """Build for all supported platforms"""
    current = get_current_platform()
    print(f"Current platform detected: {current}")
    
    success_count = 0
    total_count = len(BUILD_CONFIGS)
    
    for platform in BUILD_CONFIGS.keys():
        print(f"\n{'='*60}")
        print(f"Building for {platform}")
        print('='*60)
        
        if build_executable(platform, clean=False):
            success_count += 1
        else:
            print(f"Failed to build for {platform}")
    
    print(f"\n{'='*60}")
    print(f"Build Summary: {success_count}/{total_count} platforms successful")
    print('='*60)
    
    if success_count == total_count:
        print("✓ All builds completed successfully!")
        return True
    else:
        print(f"✗ {total_count - success_count} builds failed")
        return False

def main():
    parser = argparse.ArgumentParser(description='DarkHal 2.0 Cross-Platform Build Tool')
    parser.add_argument('target', nargs='?', choices=list(BUILD_CONFIGS.keys()) + ['all', 'current'], 
                       default='current', help='Target platform to build for')
    parser.add_argument('--no-clean', action='store_true', help='Skip cleaning build files')
    parser.add_argument('--list', action='store_true', help='List available platforms')
    
    args = parser.parse_args()
    
    if args.list:
        print("Available build targets:")
        for platform, config in BUILD_CONFIGS.items():
            print(f"  {platform:15} - {config['platform']} {config['arch']}")
        return
    
    # Install PyInstaller
    if not install_pyinstaller():
        return
    
    if args.target == 'all':
        success = build_all_platforms()
    elif args.target == 'current':
        current_platform = get_current_platform()
        if current_platform not in BUILD_CONFIGS:
            print(f"✗ Current platform {current_platform} not supported")
            print("Available platforms:", ', '.join(BUILD_CONFIGS.keys()))
            return
        success = build_executable(current_platform, not args.no_clean)
    else:
        success = build_executable(args.target, not args.no_clean)
    
    if success:
        print("\n🎉 Build completed successfully!")
    else:
        print("\n❌ Build failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()