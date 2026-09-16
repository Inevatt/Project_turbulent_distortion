"""Prevent silent reuse of checkpoints/results made by different physics."""
from pathlib import Path
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1)
def physics_id():
    root=Path(__file__).parent
    paths=sorted((root/'distortion').glob('*.py'))+sorted((root/'distortion/assets').glob('*.npz'))
    paths += [root/'optics.py',root/'data.py',root/'pair_pipeline.py']
    h=hashlib.sha256()
    for p in paths:
        h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    return h.hexdigest()


@lru_cache(maxsize=1)
def protocol_id():
    """Also guard network, optimizer loop and metric conventions."""
    root = Path(__file__).parent
    h = hashlib.sha256(physics_id().encode())
    for name in ('unet.py', 'train.py', 'metrics.py'):
        h.update(name.encode())
        h.update((root / name).read_bytes())
    return h.hexdigest()
