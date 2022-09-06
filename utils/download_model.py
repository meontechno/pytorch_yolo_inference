import os
from pathlib import Path

import torch

from utils.logger import logger


def safe_download(url, file="weights/best.pt", url2=None, min_bytes=1E0, error_msg=''):
    # Attempts to download file from url or url2, checks and removes incomplete downloads < min_bytes
    if not os.path.exists("weights"):
        os.mkdir("weights")

    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:  # url1
        logger.info(f'Downloading weights from {url} to {file}...')
        #os.system(f"curl -L '{url2 or url}' -o '{file}' --retry 3 -C -")  
        torch.hub.download_url_to_file(url, str(file))
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg  # check
    except Exception as e:  # url2
        file.unlink(missing_ok=True)  # remove partial downloads
        logger.info(f'ERROR: {e}\nRe-attempting {url2 or url} to {file}...')
        # curl download, retry and resume on fail
        os.system(f"curl -L '{url2 or url}' -o '{file}' --retry 3 -C -")  
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:  # check
            file.unlink(missing_ok=True)  # remove partial downloads
            logger.info(f"ERROR: {assert_msg}\n{error_msg}")


if __name__ == "__main__":
    safe_download("weights/best.pt", "https://github.com/meontechno/frictionless_weights/raw/main/v10.0/best.pt")
