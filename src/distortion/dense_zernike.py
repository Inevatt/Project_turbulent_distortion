"""DF-P2S-style approximate covariance: mode-specific autocorrelations, then
pointwise Noll Cholesky mixing (Chimitt et al. 2022, Eq. 9).

Autocorrelations are the circular-pupil Zernike/Kolmogorov Bessel integrals.
The geometry is a fixed collapsed-layer separation, NOT a fitted Cn2 profile.
55 px is the half-correlation of the angular-averaged *latent tilt* field.
Higher modes have their own shorter, direction-dependent correlations.
Finite FFT embedding and optional coefficient interpolation are numerical
approximations. This does not reproduce the full intermodal spatial tensor.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
import scipy.fft as fft
from scipy.interpolate import CubicSpline
from .wavefront import K_MODES, noll_cov, noll_nm

TILT_CORR_HALF_PX = 55.0
ANCHOR_SPACING_PX = 1.0

@lru_cache(maxsize=1)
def _table():
    with np.load(Path(__file__).parent/'assets/modal_corr.npz', allow_pickle=False) as f:
        return {k:f[k] for k in f.files}

def _bilinear_resize(x,n):
    g=x.shape[-1]
    if g==n: return x.copy()
    q=np.linspace(0,g-1,n); i=np.floor(q).astype(int); j=np.minimum(i+1,g-1)
    t=(q-i).astype(np.float32)
    v=x[:,i,:]*(1-t)[None,:,None]+x[:,j,:]*t[None,:,None]
    return v[:,:,i]*(1-t)[None,None,:]+v[:,:,j]*t[None,None,:]

class DenseZernikeSampler:
    def __init__(self,k_modes=K_MODES,corr_half_px=TILT_CORR_HALF_PX,
                 anchor_spacing_px=ANCHOR_SPACING_PX):
        if k_modes!=36: raise ValueError('tabulated modes are Noll 1..36')
        if corr_half_px<=0 or anchor_spacing_px<=0: raise ValueError('invalid scale')
        self.k_modes=k_modes; self.n_coeff=k_modes-1
        self.corr_half_px=float(corr_half_px)
        self.anchor_spacing_px=float(anchor_spacing_px)
        self.cov=noll_cov(k_modes)[1:,1:]
        self.root=np.linalg.cholesky(self.cov).astype(np.float32)
        tab=_table(); c=tab['n1_o0']/tab['n1_o0'][0]
        idx=np.flatnonzero(c<=.5)[0]
        sh=np.interp(.5,c[idx-1:idx+1][::-1],tab['s'][idx-1:idx+1][::-1])
        self.px_per_s=self.corr_half_px/sh
        self._cache={}

    def modal_covariance_from_roots(self):
        return self.root.astype(float)@self.root.astype(float).T

    def latent_corr(self,j,dy,dx):
        tab=_table(); n,m=noll_nm(j); am=abs(m)
        s=np.hypot(dy,dx)/self.px_per_s
        v=CubicSpline(tab["s"],tab[f"n{n}_o0"])(s)
        if am:
            # Fourier transform of cos^2/sin^2 angular pupil factors.
            sign=1 if m>0 else -1
            v=v+sign*(-1)**am*np.cos(2*am*np.arctan2(dy,dx))*CubicSpline(tab["s"],tab[f"n{n}_o{2*am}"])(s)
        return v/tab[f'n{n}_o0'][0]

    def _geometry(self,n):
        g=min(n,int(np.ceil((n-1)/self.anchor_spacing_px))+1)
        return g,(n-1)/(g-1)

    def _prepare(self,n):
        if n in self._cache: return self._cache[n]
        g,step=self._geometry(n); m=2*fft.next_fast_len(g)
        q=np.fft.fftfreq(m)*m*step; dy,dx=np.meshgrid(q,q,indexing='ij')
        if np.hypot(dy, dx).max()/self.px_per_s > _table()["s"][-1]:
            raise ValueError("Image/geometry exceeds precomputed correlation table")
        roots=[]; variances=[]; negatives=[]
        t=np.linspace(0,g-1,n)%1; a=1-t; b=t
        ws=[a[:,None]*a[None,:],a[:,None]*b[None,:],b[:,None]*a[None,:],b[:,None]*b[None,:]]
        pos=[(0,0),(0,1),(1,0),(1,1)]
        for j in range(2,self.k_modes+1):
            cov=self.latent_corr(j,dy,dx)
            lam=fft.fft2(cov).real
            negatives.append(float(-lam[lam<0].sum()/np.abs(lam).sum()))
            lam=np.maximum(lam,0)
            lam/=lam.mean()
            actual=fft.ifft2(lam).real
            var=np.zeros((n,n))
            for u,(yu,xu) in enumerate(pos):
                for v,(yv,xv) in enumerate(pos):
                    var+=ws[u]*ws[v]*actual[(yu-yv)%m,(xu-xv)%m]
            roots.append(np.sqrt(lam[:,:m//2+1]))
            variances.append(1/np.sqrt(var))
        result=(np.asarray(roots,np.float32),np.asarray(variances,np.float32),m,g,np.array(negatives))
        self._cache[n]=result
        return result

    def sample(self,n,d_over_r0,rng):
        if n<2 or not np.isfinite(d_over_r0) or d_over_r0<0: raise ValueError('invalid n or D/r0')
        roots,invstd,m,g,_=self._prepare(n)
        white=rng.standard_normal((self.n_coeff,m,m),dtype=np.float32)
        latent=fft.irfft2(fft.rfft2(white)*roots,s=(m,m))[:,:g,:g]
        latent=_bilinear_resize(latent,n)*invstd
        return (self.root@latent.reshape(self.n_coeff,-1)).reshape(self.n_coeff,n,n)*np.float32(d_over_r0**(5/6))

    def sample_for_d3(self,n,d_over_r0,rng,high_grid):
        a=self.sample(n,d_over_r0,rng)
        from .d3 import _sample_regular
        return a[:2],_sample_regular(a[2:],high_grid)
