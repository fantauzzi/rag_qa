import logging
import os
from pathlib import Path
from urllib import request

import wandb
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(levelname)s %(message)s")
_logger = logging.getLogger()
info = _logger.info
warning = _logger.warning
error = _logger.error


def log_into_wandb():
    wandb_dir = Path('..')
    if os.environ.get('WANDB_DIR') is None:
        os.environ['WANDB_DIR'] = str(wandb_dir)
    wandb.login()


def fetch_file(url: str, destination: str) -> int:
    req = request.Request(url, method='HEAD')
    http_resp = request.urlopen(req)
    if http_resp.status != 200:
        return http_resp.status
    file_size = int(http_resp.headers['Content-Length'])
    with tqdm(total=file_size) as pbar:
        def report_hook(block_n, read_size, total_size):
            pbar.update(read_size)

        request.urlretrieve(url=url, filename=destination, reporthook=report_hook)
    return http_resp.status
