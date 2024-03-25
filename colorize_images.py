#!/usr/bin/env python
#SBATCH -N 1 # n nodes
#SBATCH --ntasks-per-node=8 # cpu cores
#SBATCH --job-name=colorizing
#SBATCH --mem=20GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -x gpu005
import argparse
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recoloring')
    parser.add_argument('--image_dir', type=Path, required=True, help='Input image dir')
    parser.add_argument('--output_dir', type=Path, required=True, help='Output dir')
    opt = parser.parse_args()

    output_dir: Path = opt.output_dir
    image_dir: Path = opt.image_dir

    output_dir.mkdir(exist_ok=True)

    from colorization_benchmark.model_wrappers.ddcolor import DDColor

    model_path = Path("damo/cv_ddcolor_image-colorization")
    colorizer = DDColor(model_path)

    for image_path in tqdm(list(image_dir.glob("*.jpg"))):
        results = colorizer.colorize(input_path=image_path, reference_paths=[])

        color = results["color"]
        color.save(output_dir / image_path.name)
