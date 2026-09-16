"""End-to-end timed steps, including synthesis, transfers and UNet backward.
Does not save a model or change the experiment's epoch/LR budget.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
import argparse,time,json,math
from pathlib import Path
import numpy as np
import torch,yaml
from torch.utils.data import DataLoader
from src.data import split_indices
from src.distortion import LEVELS
from src.pair_pipeline import pairs_class,render_batch
from src.unet import UNet
from src.metrics import psnr,ssim
from src.physics_version import physics_id

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/seed1.yaml');p.add_argument('--level',default='d3',choices=sorted(LEVELS));p.add_argument('--steps',type=int,default=30);p.add_argument('--warmup',type=int,default=5);p.add_argument('--val-steps',type=int,default=5);p.add_argument('--device',default='cuda');p.add_argument('--output',type=Path);a=p.parse_args()
 if a.steps<1 or a.warmup<0 or a.val_steps<1: p.error('invalid step count')
 cfg=yaml.safe_load(Path(a.config).read_text());d,t,m=cfg['data'],cfg['train'],cfg['model']
 device=torch.device(a.device)
 if device.type=='cuda' and not torch.cuda.is_available(): raise SystemExit('CUDA unavailable; run this benchmark on the training server')
 torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 deg=LEVELS[a.level](cfg['degradation']['diffraction_fwhm_px'])
 tr,va,te=split_indices(d['tiles'],d['val_frac'],d['test_frac'],cfg['split_seed'])
 kw=dict(tiles_dir=d['tiles'],degradation=deg,margin_px=d['margin_px'],crop_px=d['crop_px'],d_over_r0_range=d['d_over_r0_range'])
 ds=pairs_class(deg)(indices=tr,seed=cfg['seed'],samples_per_tile=d['samples_per_tile'],**kw)
 vd=pairs_class(deg,True)(indices=va,seed=cfg['eval_seed'],**kw)
 workers=int(t['num_workers']);batch=int(t['batch_size'])
 if hasattr(deg,'sampler'):deg.sampler._prepare(d['crop_px']+2*d['margin_px'])
 dlkw=dict(batch_size=batch,num_workers=workers,pin_memory=device.type=='cuda',**({'prefetch_factor':1} if workers else {}))
 dl=DataLoader(ds,shuffle=False,drop_last=True,**dlkw);vl=DataLoader(vd,shuffle=False,**dlkw)
 if len(dl)<a.steps+a.warmup or len(vl)<a.val_steps: raise SystemExit('not enough batches for requested benchmark')
 torch.manual_seed(cfg['seed']);net=UNet(in_ch=m['channels'],out_ch=m['channels'],base=m['base'],depth=m['depth']).to(device)
 opt=torch.optim.Adam(net.parameters(),lr=float(t['lr']))
 def sync():
  if device.type=='cuda':torch.cuda.synchronize()
 it=iter(dl);times=[]
 for i in range(a.warmup+a.steps):
  sync();start=time.perf_counter();packed=next(it);x,y,_=render_batch(packed,deg,device)
  opt.zero_grad(set_to_none=True);loss=(net(x)-y).abs().mean();loss.backward();opt.step();sync()
  if i>=a.warmup:times.append(time.perf_counter()-start)
  print(f'step {i+1}/{a.warmup+a.steps}, loss={loss.item():.5f}',flush=True)
 del it
 net.eval();vit=iter(vl);vtime=[];synth_times=[]
 with torch.no_grad():
  for i in range(a.val_steps):
   sync();start=time.perf_counter();x,y,_=render_batch(next(vit),deg,device);sync();synth_times.append(time.perf_counter()-start);pred=net(x);psnr(pred,y);ssim(pred,y);sync();vtime.append(time.perf_counter()-start)
 train_sec=float(np.mean(times));val_sec=float(np.mean(vtime));epoch=train_sec*len(dl)+val_sec*len(vl)
 result=dict(physics_id=physics_id(),device=str(device),gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,level=a.level,train_step_seconds=train_sec,val_step_seconds=val_sec,train_step_p90=float(np.quantile(times,.9)),epoch_estimate_seconds=epoch,hours_estimate=epoch*t['epochs']/3600,hours_with_20pct_reserve=epoch*t['epochs']/3600*1.2,epochs=t['epochs'],batch_size=batch,workers=workers,train_batches=len(dl),val_batches=len(vl),test_batches=math.ceil(len(te)/batch),val_synthesis_seconds=float(np.mean(synth_times)),peak_cuda_GiB=torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else None,note='Estimate; epoch worker startup, contention and checkpoint I/O can change runtime')
 print(json.dumps(result,indent=2),flush=True)
 if a.output:
  a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
