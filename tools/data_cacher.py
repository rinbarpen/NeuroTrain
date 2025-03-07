from utils.dataset.dataset import dataset_to_numpy
from argparse import ArgumentParser
import logging
from pathlib import Path

if __name__ == '__main__':
    parser = ArgumentParser(description="Cache dataset to numpy format")
    parser.add_argument('-n', '--dataset_name', type=str, required=True, help="Name of the dataset")
    parser.add_argument('--save_dir', type=str, required=True, help="The directory saving numpy data")
    parser.add_argument('--base_dir', type=str, required=True, help="The directory of dataset")
    parser.add_argument('--to_rgb', type=str, default=False, help="Convert to rgb image")

    args = parser.parse_args()
    to_rgb = args.to_rgb
    dataset_name = args.dataset_name
    save_dir = Path(args.save_dir)
    base_dir = Path(args.base_dir)
    try:
        dataset_to_numpy(dataset_name, save_dir, base_dir, to_rgb=to_rgb)
        logging.info("Dataset cached successfully.")
    except Exception as e:
        logging.error(f"Error caching dataset: {e}")


