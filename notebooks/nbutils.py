import os
from pathlib import Path


def chdir_to_root():
    if not Path('cudams').is_dir():
        os.chdir('..')