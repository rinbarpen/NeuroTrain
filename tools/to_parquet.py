import os.path
import pandas as pd
import os
from argparse import ArgumentParser

def to_parquet(filename):
    in_type = os.path.splitext(filename)[-1].lower()
    output = os.path.join('output', os.path.basename(filename) + '.parquet')
    os.makedirs('output', exist_ok=True)
    match in_type:
        case '.csv': df = pd.read_csv(filename)
        case '.xls'|'.xlsx': df = pd.read_excel(filename)

    df.to_parquet(output)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', required=True, type=str)
    args = parser.parse_args()

    input = args.input

    if os.path.isfile(input):
        to_parquet(input)
    elif os.path.isdir(input):
        # Recursively convert each file in the directory
        for root, _, files in os.walk(input):
            for file in files:
                full_path = os.path.join(root, file)
                to_parquet(full_path)
    else:
        print(f"Input path {input} is neither a file nor a directory.")
