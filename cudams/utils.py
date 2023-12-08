from pathlib import Path

def batches(lst, batch_size):
    """
    Batch data from the iterable into tuples of length n. The last batch may be shorter than n.
    """
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def argbatch(lst, batch_size):
    """
    Batch data from the iterable into tuples of start-end indices
    """
    for i in range(0, len(lst), batch_size):
        yield i, i + batch_size
        
        
def mkdir(p:Path) -> Path:
    p = Path(p)
    p.mkdir(exist_ok=True)
    return p
