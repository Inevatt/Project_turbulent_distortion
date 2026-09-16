"""Evaluate paired real images described by a user-supplied JSON manifest.

Manifest: [{"id":"scene/frame", "scene":"scene", "input":"path.png",
            "target":"reference.png"}, ...]. Paths are relative to manifest.
All models and no-op share the same optional integer translation, estimated
from the INPUT only. No model-specific alignment, resizing or target fitting.
Images with no reliable target cannot yield PSNR/SSIM using this script.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from skimage.registration import phase_cross_correlation
import yaml
from .distortion import LEVELS
from .eval_matrix import atomic_npz, file_sha256, load_model
from .metrics import psnr, ssim
from .physics_version import protocol_id


def align_pair(image, target, registration):
    if image.shape != target.shape:
        raise ValueError('real input and target shapes differ; supply registered pairs without resizing')
    if registration == 'none':
        return image, target, (0, 0)
    shift, _, _ = phase_cross_correlation(target, image, upsample_factor=1, normalization=None)
    dy, dx = (int(round(v)) for v in shift)
    h, w = image.shape
    y0, y1 = max(0, -dy), min(h, h-dy)
    x0, x1 = max(0, -dx), min(w, w-dx)
    if min(y1-y0, x1-x0) < 11:
        raise ValueError('insufficient overlap after integer registration')
    return image[y0:y1,x0:x1], target[y0+dy:y1+dy,x0+dx:x1+dx], (dy,dx)


@torch.no_grad()
def predict_patches(net, image, patch, device, batch=16):
    """Overlapping training-size windows, uniform averaging, full-image output.
    GroupNorm always sees the training patch size. This is an explicit real
    inference protocol, not claimed equivalent to whole-image GroupNorm.
    """
    h, w = image.shape
    padded = np.pad(image, ((0,max(0,patch-h)),(0,max(0,patch-w))), mode='reflect')
    def starts(n):
        return sorted(set([*range(0,n-patch+1,max(1,patch//2)),n-patch]))
    positions = [(y,x) for y in starts(padded.shape[0]) for x in starts(padded.shape[1])]
    out = np.zeros_like(padded); norm = np.zeros_like(padded)
    for start in range(0,len(positions),batch):
        pos = positions[start:start+batch]
        inputs = np.stack([padded[y:y+patch,x:x+patch] for y,x in pos])[:,None]
        predictions = net(torch.from_numpy(inputs).to(device)).cpu().numpy()[:,0]
        for (y,x), pred in zip(pos,predictions):
            out[y:y+patch,x:x+patch] += pred
            norm[y:y+patch,x:x+patch] += 1
    return (out/norm)[:h,:w]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--registration', choices=['none','integer'], default='none')
    a = p.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    entries = json.loads(a.manifest.read_text())
    if not entries or len({e['id'] for e in entries}) != len(entries):
        p.error('manifest must be nonempty with unique ids')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    models={'noop':None, **{lv:load_model(cfg,lv,device) for lv in LEVELS}}
    result = {f'{metric}_{row}':[] for row in models for metric in ('psnr','ssim')}
    shifts=[]; source_hashes=[]
    for e in entries:
        paths=[a.manifest.parent/e[k] for k in ('input','target')]
        arrays=[]
        for path in paths:
            with Image.open(path) as im:
                if im.mode not in ('L','RGB','RGBA'):
                    raise ValueError(f'{path}: require 8-bit grayscale/RGB; explicitly convert higher bit depths')
                arrays.append(np.asarray(im.convert('L'),np.float32)/255)
        image,target,shift=align_pair(*arrays,a.registration);shifts.append(shift)
        if min(image.shape)<11:raise ValueError('SSIM requires image dimensions >= 11')
        source_hashes.append([file_sha256(path) for path in paths])
        for row,net in models.items():
            pred=image if net is None else predict_patches(net,image,cfg['data']['crop_px'],device,cfg['train']['batch_size'])
            result[f'psnr_{row}'].append(psnr(pred[None,None],target[None,None])[0])
            result[f'ssim_{row}'].append(ssim(pred[None,None],target[None,None])[0])
        print(e['id'],flush=True)
    meta=dict(cfg=cfg,protocol_id=protocol_id(),registration=a.registration,
              manifest=file_sha256(a.manifest),sources=source_hashes,
              evaluator=file_sha256(__file__),
              checkpoints={lv:file_sha256(Path(cfg['out_dir'])/lv/'last.pt') for lv in LEVELS},
              inference='training-size 50% overlapping patches, uniform averaging; metric per full aligned image')
    atomic_npz(a.output,provenance=json.dumps(meta,sort_keys=True),
               sample_id=np.array([e['id'] for e in entries]),
               scene=np.array([e['scene'] for e in entries]),shift=np.array(shifts),
               **{k:np.asarray(v,np.float32) for k,v in result.items()})
    print(a.output,flush=True)


if __name__=='__main__':main()
