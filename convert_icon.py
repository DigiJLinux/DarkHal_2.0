#!/usr/bin/env python3
"""
Icon Converter Utility
Converts PNG images to ICO format for Windows applications
"""

import os
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("Pillow is required for image conversion. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image, ImageDraw


def create_default_icon():
    """Create a default LLM_Train icon if no PNG is provided."""
    # Create a 256x256 icon with gradient background
    size = 256
    image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    
    # Create gradient background
    for i in range(size):
        alpha = int(255 * (i / size))
        color = (45, 85, 135, alpha)  # Blue gradient
        draw.line([(0, i), (size, i)], fill=color)
    
    # Draw "LLM" text-like representation
    center = size // 2
    
    # Draw stylized "LLM" using rectangles
    # L
    draw.rectangle([40, 60, 60, 180], fill=(255, 255, 255, 255))
    draw.rectangle([40, 160, 100, 180], fill=(255, 255, 255, 255))
    
    # L
    draw.rectangle([110, 60, 130, 180], fill=(255, 255, 255, 255))
    draw.rectangle([110, 160, 170, 180], fill=(255, 255, 255, 255))
    
    # M
    draw.rectangle([180, 60, 200, 180], fill=(255, 255, 255, 255))
    draw.rectangle([200, 60, 220, 100], fill=(255, 255, 255, 255))
    draw.rectangle([210, 100, 230, 120], fill=(255, 255, 255, 255))
    draw.rectangle([220, 120, 240, 180], fill=(255, 255, 255, 255))
    
    # Add subtle border
    draw.rectangle([10, 10, size-10, size-10], outline=(255, 255, 255, 128), width=2)
    
    return image


def png_to_ico(png_path, ico_path=None, sizes=None):
    """
    Convert PNG to ICO format with multiple sizes.
    
    Args:
        png_path (str): Path to the input PNG file
        ico_path (str): Path for the output ICO file (optional)
        sizes (list): List of sizes to include in ICO (default: [16, 32, 48, 64, 128, 256])
    """
    if sizes is None:
        sizes = [16, 32, 48, 64, 128, 256]
    
    png_path = Path(png_path)
    
    if not png_path.exists():
        print(f"PNG file not found: {png_path}")
        print("Creating default icon instead...")
        image = create_default_icon()
    else:
        try:
            image = Image.open(png_path)
            print(f"Loaded PNG: {png_path} ({image.size})")
        except Exception as e:
            print(f"Error loading PNG: {e}")
            print("Creating default icon instead...")
            image = create_default_icon()
    
    # Convert to RGBA if not already
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Create different sizes
    icon_images = []
    for size in sizes:
        # Resize image maintaining aspect ratio
        resized = image.resize((size, size), Image.Resampling.LANCZOS)
        icon_images.append(resized)
        print(f"Created {size}x{size} icon")
    
    # Determine output path
    if ico_path is None:
        ico_path = png_path.with_suffix('.ico')
    else:
        ico_path = Path(ico_path)
    
    # Save as ICO
    try:
        icon_images[0].save(
            ico_path,
            format='ICO',
            sizes=[(img.width, img.height) for img in icon_images],
            append_images=icon_images[1:]
        )
        print(f"Successfully created ICO: {ico_path}")
        return str(ico_path)
    except Exception as e:
        print(f"Error creating ICO: {e}")
        return None


def convert_assets_folder(assets_dir="assets"):
    """Convert all PNG files in assets folder to ICO format."""
    assets_path = Path(assets_dir)
    
    if not assets_path.exists():
        print(f"Assets directory not found: {assets_path}")
        return []
    
    png_files = list(assets_path.glob("*.png"))
    
    if not png_files:
        print(f"No PNG files found in {assets_path}")
        print("Creating default application icon...")
        
        # Create default icon
        default_ico = assets_path / "llm_train.ico"
        png_to_ico("nonexistent.png", default_ico)
        return [str(default_ico)]
    
    converted_files = []
    for png_file in png_files:
        ico_file = png_file.with_suffix('.ico')
        result = png_to_ico(png_file, ico_file)
        if result:
            converted_files.append(result)
    
    return converted_files


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert PNG images to ICO format")
    parser.add_argument("input", nargs="?", help="Input PNG file or assets directory")
    parser.add_argument("-o", "--output", help="Output ICO file")
    parser.add_argument("--sizes", nargs="+", type=int, 
                       default=[16, 32, 48, 64, 128, 256],
                       help="Icon sizes to include")
    parser.add_argument("--assets", action="store_true",
                       help="Convert all PNG files in assets directory")
    
    args = parser.parse_args()
    
    if args.assets or not args.input:
        print("Converting assets folder...")
        converted = convert_assets_folder("assets")
        if converted:
            print(f"Converted {len(converted)} files:")
            for file in converted:
                print(f"  - {file}")
        else:
            print("No files converted.")
    else:
        if Path(args.input).is_file():
            result = png_to_ico(args.input, args.output, args.sizes)
            if result:
                print(f"Conversion successful: {result}")
            else:
                print("Conversion failed.")
        else:
            print(f"Input file not found: {args.input}")


if __name__ == "__main__":
    main()