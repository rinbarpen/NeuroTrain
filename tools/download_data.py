import os
import subprocess
from curl_cffi import requests

os.makedirs('data', exist_ok=True)

DATASETS_INFO = {
    'DRIVE': {
        'url': '',
    }
}

def download(dataset: str):
    subprocess.call(['aria2c', dataset])