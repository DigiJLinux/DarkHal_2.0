#!/usr/bin/env python3
"""
DarkHal 2.0 Build Script
Builds for: Windows x64, Linux x64, Linux ARM64
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path

# Build configurations for the three target platforms
BUILD_CONFIGS = {
    'windows-x64': {
        'pyinstaller_args': [
            '--onefile',
            '--windowed',
            '--name', 'DarkHal-2.0-windows-x64',
            '--icon', 'assets/logo.ico',
            '--add-data', 'assets/*;assets',
            '--add-data', 'llm_runtime;llm_runtime',
            '--hidden-import', 'tkinter',
            '--hidden-import', 'tkinter.ttk',
            '--hidden-import', 'torch',
            '--hidden-import', 'transformers'
        ],
        'output_name': 'DarkHal-2.0-windows-x64.exe'
    },
    'linux-x64': {
        'pyinstaller_args': [
            '--onefile',
            '--name', 'DarkHal-2.0-linux-x64',
            '--add-data', 'assets/*:assets',
            '--add-data', 'llm_runtime:llm_runtime',
            '--hidden-import', 'tkinter',
            '--hidden-import', 'tkinter.ttk',
            '--hidden-import', 'torch',
            '--hidden-import', 'transformers'
        ],
        'output_name': 'DarkHal-2.0-linux-x64'
    },
    'linux-arm64': {
        'pyinstaller_args': [
            '--onefile',
            '--name', 'DarkHal-2.0-linux-arm64', 
            '--add-data', 'assets/*:assets',
            '--add-data', 'llm_runtime:llm_runtime',
            '--hidden-import', 'tkinter',
            '--hidden-import', 'tkinter.ttk',
            '--hidden-import', 'torch',
            '--hidden-import', 'transformers',
            '--target-architecture', 'arm64'
        ],
        'output_name': 'DarkHal-2.0-linux-arm64'
    }
}

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
    import shutil
    import glob
    
    dirs_to_remove = ['build', 'dist']
    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"Cleaned {dir_name}/ directory")
    
    for spec_file in glob.glob('*.spec'):
        os.remove(spec_file)
        print(f"Cleaned {spec_file}")

def build_executable(target_platform):
    """Build executable for specified platform"""
    if target_platform not in BUILD_CONFIGS:
        print(f"✗ Unknown target platform: {target_platform}")
        print(f"Available platforms: {', '.join(BUILD_CONFIGS.keys())}")
        return False
    
    config = BUILD_CONFIGS[target_platform]
    
    print(f"\n{'='*50}")
    print(f"Building DarkHal 2.0 for {target_platform}")
    print('='*50)
    
    # Check required files
    if not os.path.exists('main.py'):
        print("✗ main.py not found. Please run from the correct directory.")
        return False
    
    if not os.path.exists('llm_runtime'):
        print("✗ llm_runtime directory not found")
        return False
    
    # Create dist directory for this platform
    dist_dir = f'dist/{target_platform}'
    os.makedirs(dist_dir, exist_ok=True)
    
    # Build PyInstaller command
    cmd = ['pyinstaller'] + config['pyinstaller_args'] + [
        '--distpath', dist_dir,
        '--workpath', f'build/{target_platform}',
        '--specpath', f'build/{target_platform}',
        'main.py'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✓ PyInstaller completed")
        
        # Find the created executable
        expected_path = os.path.join(dist_dir, config['output_name'])
        name_without_ext = config['output_name'].replace('.exe', '')
        possible_paths = [
            expected_path,
            os.path.join(dist_dir, name_without_ext),
            os.path.join(dist_dir, name_without_ext, name_without_ext),
            os.path.join(dist_dir, name_without_ext, name_without_ext + '.exe')
        ]
        
        executable_path = None
        for path in possible_paths:
            if os.path.exists(path) and os.path.isfile(path):
                executable_path = path
                break
        
        if executable_path:
            # Move to correct location if needed
            final_path = expected_path
            if executable_path != final_path:
                if os.path.exists(final_path):
                    os.remove(final_path)
                os.rename(executable_path, final_path)
                executable_path = final_path
            
            # Make executable on Linux
            if target_platform.startswith('linux'):
                os.chmod(executable_path, 0o755)
            
            file_size = os.path.getsize(executable_path) / (1024 * 1024)
            print(f"✓ SUCCESS! Executable created: {executable_path}")
            print(f"  File size: {file_size:.1f} MB")
            return True
        else:
            print("✗ Executable not found after build")
            print("Files in dist directory:")
            for root, dirs, files in os.walk(dist_dir):
                for file in files:
                    print(f"  {os.path.join(root, file)}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"✗ Build failed with exit code {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description='DarkHal 2.0 Cross-Platform Build Tool')
    parser.add_argument('target', nargs='?', 
                       choices=['windows-x64', 'linux-x64', 'linux-arm64', 'all'], 
                       default='all', 
                       help='Target platform to build for')
    parser.add_argument('--no-clean', action='store_true', help='Skip cleaning build files')
    
    args = parser.parse_args()
    
    print("DarkHal 2.0 Cross-Platform Builder")
    print("Supporting: Windows x64, Linux x64, Linux ARM64")
    
    # Install PyInstaller
    if not install_pyinstaller():
        sys.exit(1)
    
    # Clean build files unless requested not to
    if not args.no_clean:
        clean_build_files()
    
    # Build targets
    if args.target == 'all':
        targets = ['windows-x64', 'linux-x64', 'linux-arm64']
        success_count = 0
        
        for target in targets:
            if build_executable(target):
                success_count += 1
        
        print(f"\n{'='*60}")
        print(f"Build Summary: {success_count}/{len(targets)} platforms successful")
        print('='*60)
        
        if success_count == len(targets):
            print("🎉 All builds completed successfully!")
            print("\nExecutables created:")
            for target in targets:
                config = BUILD_CONFIGS[target]
                path = f"dist/{target}/{config['output_name']}"
                if os.path.exists(path):
                    size = os.path.getsize(path) / (1024 * 1024)
                    print(f"  {path} ({size:.1f} MB)")
        else:
            print(f"❌ {len(targets) - success_count} builds failed!")
            sys.exit(1)
    else:
        if build_executable(args.target):
            print("\n🎉 Build completed successfully!")
        else:
            print("\n❌ Build failed!")
            sys.exit(1)

if __name__ == '__main__':
    main()