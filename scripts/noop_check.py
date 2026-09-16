"""No-op on frozen validation crops; D3 is rendered in the main process."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse
from pathlib import Path
import numpy as np,torch,yaml
from torch.utils.data import DataLoader
from src.data import split_indices
from src.distortion import LEVELS
from src.pair_pipeline import pairs_class,render_batch
from src.metrics import psnr,ssim

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/seed1.yaml');p.add_argument('--samples',type=int,default=256);p.add_argument('--levels',nargs='+',choices=sorted(LEVELS),default=list(LEVELS));p.add_argument('--workers',type=int,default=0);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');a=p.parse_args()
 cfg=yaml.safe_load(Path(a.config).read_text());d=cfg['data'];device=torch.device(a.device)
 _,va,_=split_indices(d['tiles'],d['val_frac'],d['test_frac'],cfg['split_seed'])
 if a.samples>0:va=va[:a.samples]
 for name in a.levels:
  deg=LEVELS[name](cfg['degradation']['diffraction_fwhm_px'])
  ds=pairs_class(deg,True)(d['tiles'],va,deg,crop_px=d['crop_px'],margin_px=d['margin_px'],d_over_r0_range=tuple(d['d_over_r0_range']),seed=cfg['eval_seed'])
  dl=DataLoader(ds,batch_size=int(cfg['train']['batch_size']),num_workers=a.workers,**({'prefetch_factor':1} if a.workers else {}))
  ps,ss=[],[]
  for packed in dl:
   x,y,_=render_batch(packed,deg,device);ps.append(psnr(x,y));ss.append(ssim(x,y))
  print(f'no-op {name}: PSNR {np.concatenate(ps).mean():.2f} SSIM {np.concatenate(ss).mean():.4f} N={len(ds)}',flush=True)
if __name__=='__main__':main()
