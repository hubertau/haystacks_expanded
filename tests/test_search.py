import subprocess
import glob
import os
import pytest
import json
from pathlib import Path

def test_search(tmp_path):
    result = subprocess.run(['hay', 'search', 'health', '--savepath', tmp_path], capture_output=True)
    assert b"200" in result.stderr

def test_one_download(tmp_path):
    # get most recent json file
    # files = glob.glob(os.path.join('./../data/01_raw', '*.json'))
    # files = sorted(files, key=os.path.getmtime, reverse=True)
    # file = files[0]
    _ = subprocess.run(['hay', 'download', '../data/01_raw/for_pytest.json', '--savepath', tmp_path,   '--max_download', '2'])
    downloaded_files = glob.glob(os.path.join(tmp_path, 'videos/*.mp4'))
    assert len(downloaded_files) == 2

def test_comment_retrieval(tmp_path):
    _ = subprocess.run(['hay', 'comments', '../data/01_raw/for_pytest.json', '--savepath', tmp_path])
    file = Path(tmp_path) / 'comments' / 'for_pytest_comments.json'
    with open(file, 'r') as f:
        x = json.load(f)
    assert len(x) == 3
    result = subprocess.run(['hay', 'comments', '../data/01_raw/for_pytest.json', '--savepath', tmp_path], capture_output=True)
    assert b"FileExistsError" in result.stderr