from pathlib import Path
import re

def batches(lst, batch_size) -> list:
    """
    Batch data from the iterable into tuples of length n. The last batch may be shorter than n.
    """
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def argbatch(lst, batch_size) -> tuple[int, int]:
    """
    Batch data from the iterable into tuples of start-end indices
    """
    for i in range(0, len(lst), batch_size):
        yield i, i + batch_size
        
def mkdir(p:Path) -> Path:
    p = Path(p)
    p.mkdir(exist_ok=True, parents=True)
    return p

def name2idx(p: Path) -> tuple[int, int, int, int]:
    match = re.match(r'(\d+)-(\d+)\.(\d+)-(\d+)', p.stem)
    rstart, rend, qstart, qend = map(int, match.groups())
    return rstart, rend, qstart, qend