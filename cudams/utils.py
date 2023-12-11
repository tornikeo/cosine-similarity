import shutil
from pathlib import Path
import re
import sys
from contextlib import contextmanager

def batches(lst, batch_size) -> list:
    """
    Batch data from the iterable into tuples of length n. The last batch may be shorter than n.
    """
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


def argbatch(lst, batch_size) -> tuple[int, int]:
    """
    Batch data from the iterable into tuples of start-end indices
    """
    for i in range(0, len(lst), batch_size):
        yield i, i + batch_size


def mkdir(p: Path, clean=False) -> Path:
    p = Path(p)
    if clean and p.is_dir() and p.exists():
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(exist_ok=True, parents=True)
    return p


def name2idx(p: Path) -> tuple[int, int, int, int]:
    match = re.match(r"(\d+)-(\d+)\.(\d+)-(\d+)", p.stem)
    rstart, rend, qstart, qend = map(int, match.groups())
    return rstart, rend, qstart, qend

@contextmanager
def mute_stdout():
    stdout = sys.stdout
    sys.stdout = None
    yield
    sys.stdout = stdout
    