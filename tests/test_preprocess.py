import subprocess
import glob
import os
import pytest
import pandas as pd
from pathlib import Path

def test_extract(tmp_path):
    _ = subprocess.run(['hay', 'extract', '--metadata', '../data/01_raw/for_pytest.json', '--output', f"{Path(tmp_path)/'test.csv'}", "--overwrite"], capture_output=True)
    file = pd.read_csv(Path(tmp_path)/'test.csv')
    assert (len(file) > 0)

def test_reextract(tmp_path):
    _ = subprocess.run(['hay', 'extract', '--metadata', '../data/01_raw/for_pytest.json', '--output', f"{Path(tmp_path)/'test.csv'}", '--reextract', '--overwrite'], capture_output=True)
    file = pd.read_csv(Path(tmp_path)/'test.csv')
    assert (len(file) > 0)

def test_detect_from_metadata(tmp_path):
    _ = subprocess.run(['hay', 'detect', '--metadata', '../data/01_raw/for_pytest.json', '--output', f"{Path(tmp_path)/'output.csv'}", '--overwrite'], capture_output=True)
    file = pd.read_csv(f"{Path(tmp_path)/'output.csv'}")
    assert len(file) > 0

def test_detect_from_features_file(tmp_path):
    _ = subprocess.run(['hay', 'detect', '--features_file', '../data/02_intermediate/for_pytest_feat.csv', '--output', f"{Path(tmp_path)/'output.csv'}", '--overwrite'], capture_output=True)
    file = pd.read_csv(f"{Path(tmp_path)/'output.csv'}")
    assert len(file) > 0

def test_detect_error_throw():
    result = subprocess.run(['hay', 'detect'], capture_output=True)
    assert b"ValueError" in result.stderr