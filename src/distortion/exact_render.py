"""Dense exact pupil PSFs; normalized source scattering, streamed by rows.

No PSF anchors, learned map or surrogate assets. Runs on CPU for verification
and on the training GPU in the main process. torch.fold performs the actual
source scattering (not output-weighted gathering).
"""
import numpy as np
import torch
import torch.nn.functional as F
from .wavefront import PSF_RADIUS_PX

class ExactRenderer:
    def __init__(self,wf,device='cpu',rows=None):
        self.device=torch.device(device)
        self.rows=int(rows or (32 if self.device.type=="cuda" else 8))
        self.n_grid=wf.n_grid; self.r=PSF_RADIUS_PX
        self.basis=torch.as_tensor(wf.basis[2:].astype(np.float32),device=self.device)
        self.mask=torch.as_tensor(wf.mask,device=self.device)
        # Crop in unshifted FFT coordinates before abs/square: same PSF,
        # without fftshift and intensity arrays for 128x128 discarded pixels.
        freq=torch.arange(-self.r,self.r+1,device=self.device)%self.n_grid
        self.freq_y,self.freq_x=freq[:,None],freq[None,:]
        self.px_per_rad=wf.px_per_rad

    @torch.no_grad()
    def __call__(self,img,a,crop=0):
        # img H,W; a 35,H,W, already tilt-clipped
        img=torch.as_tensor(img,dtype=torch.float32,device=self.device)
        a=torch.as_tensor(a,dtype=torch.float32,device=self.device)
        H,W=img.shape; r=self.r; k=2*r+1
        yy,xx=torch.meshgrid(torch.arange(H,device=self.device),torch.arange(W,device=self.device),indexing='ij')
        x=xx-a[0]*self.px_per_rad; y=yy-a[1]*self.px_per_rad
        grid=torch.stack((2*x/(W-1)-1,2*y/(H-1)-1),-1)[None]
        # align_corners=True is required by the W-1 coordinate conversion.
        warped=F.grid_sample(img[None,None],grid,mode='bilinear',padding_mode='reflection',align_corners=True)[0,0]
        # numpy symmetric padding, implemented by integer reflection indices.
        def symmetric_idx(n):
            z=torch.arange(-r,n+r,device=self.device)%(2*n)
            return torch.where(z<n,z,2*n-1-z)
        if crop:
            if crop < r or 2*crop >= min(H,W):
                raise ValueError("crop must cover PSF support")
            src=warped[crop-r:H-crop+r,crop-r:W-crop+r]
            high=a[2:,crop-r:H-crop+r,crop-r:W-crop+r]
            output_h,output_w=H-2*crop,W-2*crop
        else:
            iy,ix=symmetric_idx(H),symmetric_idx(W)
            src=warped[iy[:,None],ix[None,:]]
            high=a[2:][:,iy[:,None],ix[None,:]]
            output_h,output_w=H,W
        hp,wp=src.shape
        out=torch.zeros((hp+2*r,wp+2*r),device=self.device)
        norm=torch.zeros_like(out)
        for y0 in range(0,hp,self.rows):
            y1=min(y0+self.rows,hp); nr=y1-y0
            coeff=high[:,y0:y1].reshape(33,-1).T
            phase=coeff@self.basis
            pupil=torch.zeros((len(coeff),self.n_grid,self.n_grid),dtype=torch.complex64,device=self.device)
            pupil[:,self.mask]=torch.polar(torch.ones_like(phase),phase)
            spectrum=torch.fft.fft2(pupil)
            h=spectrum[:,self.freq_y,self.freq_x].abs().square()
            del spectrum,pupil,phase
            h/=h.sum((-2,-1),keepdim=True)
            patches=h.reshape(-1,k*k).T[None]
            output_size=(nr+2*r,wp+2*r)
            out[y0:y1+2*r]+=F.fold(patches*src[y0:y1].reshape(1,1,-1),output_size,(k,k))[0,0]
            norm[y0:y1+2*r]+=F.fold(patches,output_size,(k,k))[0,0]
        out=out[2*r:2*r+output_h,2*r:2*r+output_w]
        norm=norm[2*r:2*r+output_h,2*r:2*r+output_w]
        return (out/norm.clamp_min(1e-12)).clamp(0,1)
