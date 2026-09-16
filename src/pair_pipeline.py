"""Keep CUDA out of DataLoader workers; render dense D3 in the main process."""
import numpy as np
import torch
from .data import DegradedPairs, FrozenDegraded, random_crop, _rng
from .distortion.d3n import NOISE_SIGMA, SIGMA_SPREAD

class PreparedPairs(DegradedPairs):
    def __getitem__(self,idx):
        tile=self.tiles[self.indices[idx%len(self.indices)]]
        m,n=self.margin_px,self.crop_px
        crop_rng=_rng(self.seed,self.epoch,idx,1)
        big=np.asarray(random_crop(tile,n+2*m,crop_rng),np.float32)/255
        dr=float(crop_rng.uniform(self.d_lo,self.d_hi))
        rng=_rng(self.seed,self.epoch,idx,2)
        a,shift=self.deg.prepare(big,dr,rng)
        oy,ox=(int(round(v)) for v in shift)
        clean=big[m-oy:m-oy+n,m-ox:m-ox+n].copy()[None]
        noise=np.zeros_like(big)
        if self.deg.name=='d3n':
            sigma=float(rng.uniform((1-SIGMA_SPREAD)*NOISE_SIGMA,(1+SIGMA_SPREAD)*NOISE_SIGMA))
            noise=rng.standard_normal(big.shape,dtype=np.float32)*sigma
        return big,clean,np.float32(dr),a,noise

class FrozenPreparedPairs(PreparedPairs):
    def set_epoch(self,epoch):
        pass

def pairs_class(deg,frozen=False):
    if deg.name in ('d3','d3n'):
        return FrozenPreparedPairs if frozen else PreparedPairs
    return FrozenDegraded if frozen else DegradedPairs

@torch.no_grad()
def render_batch(batch,deg,device):
    if len(batch)==3:
        x,y,dr=batch
        return x.to(device,non_blocking=True),y.to(device,non_blocking=True),dr
    big,clean,dr,a,noise=batch
    big=big.to(device,non_blocking=True);a=a.to(device,non_blocking=True)
    noise=noise.to(device,non_blocking=True)
    renderer=deg.renderer(device)
    n=clean.shape[-1];m=(big.shape[-1]-n)//2
    xs=[]
    for i in range(len(big)):
        x=(renderer(big[i],a[i],crop=m)+noise[i,m:m+n,m:m+n]).clamp(0,1)
        xs.append(x)
    return torch.stack(xs)[:,None],clean.to(device,non_blocking=True),dr
