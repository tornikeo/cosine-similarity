from pathlib import Path
import os

def chdir_to_root():
    if not Path('cudams').is_dir():
        os.chdir('..')