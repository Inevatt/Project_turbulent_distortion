"""Independent acceptance checks, no production tiles required."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import unittest,tempfile
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve
from scipy.fft import irfft2
from src.distortion.wavefront import Kolmogorov,noll_cov
from src.distortion.dense_zernike import DenseZernikeSampler
from src.distortion.exact_render import ExactRenderer
from src.distortion import LEVELS
from src.pair_pipeline import pairs_class,render_batch

torch.set_num_threads(1)
class Checks(unittest.TestCase):
 def test_noll(self):
  C=noll_cov(36)[1:,1:]
  self.assertAlmostEqual(C[0,0],.448,delta=.001)
  self.assertAlmostEqual(np.trace(C),1.0299-.0122,delta=.002)
  self.assertNotEqual(C[2,9],0)
  self.assertGreater(np.linalg.eigvalsh(C).min(),0)
  np.testing.assert_allclose(C,C.T,atol=1e-12)
  print('Noll tilt variance / C(j4,j11):',C[0,0],C[2,9],flush=True)

 def test_psf_and_tilt_sign(self):
  wf=Kolmogorov(2);a=np.zeros(35);a[0]=3/wf.px_per_rad;a[1]=-2/wf.px_per_rad
  def full(v):
   p=np.zeros((128,128),complex);p[wf.mask]=np.exp(1j*(v@wf.basis))
   h=abs(np.fft.fft2(p))**2;return h/h.sum()
  np.testing.assert_allclose(full(a),np.roll(full(a*0),(-2,3),(0,1)),atol=1e-14)
  for dr in [0,1,3,5]:
   h=wf.psf(wf.coeffs(dr,np.random.default_rng(2)),32)
   self.assertGreaterEqual(h.min(),0);self.assertAlmostEqual(h.sum(),1,places=12)

 def test_spatial_covariance(self):
  s=DenseZernikeSampler();rng=np.random.default_rng(8)
  samples=np.stack([s.sample(24,1,rng)[:,7,9] for _ in range(1800)])
  whiten=np.linalg.solve(s.root,samples.T)
  err=np.max(abs(np.cov(whiten)-np.eye(35)));self.assertLess(err,.11)
  np.testing.assert_allclose(s.modal_covariance_from_roots(),s.cov,atol=1e-7)
  self.assertAlmostEqual(float((s.latent_corr(2,0,55)+s.latent_corr(3,0,55))/2),.5,places=5)
  self.assertGreater(abs(s.latent_corr(2,0,20)-s.latent_corr(4,0,20)),.1)
  roots,_,m,g,neg=s._prepare(208);errcorr=0
  for i in range(35):
   actual=irfft2(roots[i]**2,s=(m,m))
   for dy,dx in [(0,1),(0,10),(10,0),(20,20),(0,55),(100,100)]:
    errcorr=max(errcorr,abs(actual[dy,dx]-s.latent_corr(i+2,dy,dx)))
  self.assertLess(errcorr,.02)
  print('whitened cov error / embedding corr probe error / negative PSD fraction:',err,errcorr,neg.max(),flush=True)

 def test_dense_scattering_independent(self):
  wf=Kolmogorov(2);rng=np.random.default_rng(7);n=8;r=19
  img=rng.random((n,n),dtype=np.float32);a=rng.normal(0,.05,(35,n,n)).astype(np.float32)
  yy,xx=np.mgrid[:n,:n];warped=map_coordinates(img,[yy-a[1]*wf.px_per_rad,xx-a[0]*wf.px_per_rad],order=1,mode='mirror')
  src=np.pad(warped,r,mode='symmetric');high=np.pad(a[2:],((0,0),(r,r),(r,r)),mode='symmetric')
  hp=len(src);out=np.zeros((hp+2*r,hp+2*r));norm=np.zeros_like(out);kernels={}
  for y in range(hp):
   for x in range(hp):
    hi=high[:,y,x];key=hi.tobytes()
    if key not in kernels:kernels[key]=wf.psf(np.r_[0.,0.,hi],r)
    h=kernels[key];out[y:y+39,x:x+39]+=src[y,x]*h;norm[y:y+39,x:x+39]+=h
  ref=(out/norm.clip(1e-12))[2*r:2*r+n,2*r:2*r+n]
  got=ExactRenderer(wf)(img,a).numpy();err=np.max(abs(got-ref));self.assertLess(err,3e-6)
  print('independent dense renderer max error:',err,flush=True)

 def test_constant_and_isoplanatic_limit(self):
  wf=Kolmogorov(2);r=ExactRenderer(wf);n=12;rng=np.random.default_rng(11)
  a=rng.normal(0,.2,(35,n,n)).astype(np.float32)
  np.testing.assert_allclose(r(np.ones((n,n),np.float32)*.37,a),.37,atol=2e-6)
  a[:2]=0;a[2:]=a[2:,0,0,None,None]
  img=rng.random((n,n),dtype=np.float32);h=wf.psf(a[:,0,0],19)
  ref=fftconvolve(np.pad(img,19,mode='symmetric'),h,mode='valid')
  np.testing.assert_allclose(r(img,a),ref,atol=2e-6)
  # A spatially constant D3 must also agree with D2 for fractional tilt.
  a[0]=.41;a[1]=-.27
  d2=LEVELS['d2'](2);ref,_=d2._render_coeffs(img,a[:,0,0])
  np.testing.assert_allclose(r(img,a),ref,atol=2e-6)

 def test_d15_d2_and_support(self):
  img=np.random.default_rng(1).random((48,48),dtype=np.float32)
  d2=LEVELS['d2'](2);d15=LEVELS['d15'](2,j_max=36)
  for dr in [1,3,5]:
   y,s=d2(img,dr,np.random.default_rng(10));z,t=d15(img,dr,np.random.default_rng(10))
   np.testing.assert_array_equal(y,z);self.assertEqual(s,t)
  for cls in LEVELS.values():self.assertLessEqual(cls(2).support_radius_px(5),40)

 def test_data_noise_and_crop(self):
  with tempfile.TemporaryDirectory() as td:
   tile=np.random.default_rng(1).integers(0,256,(2,88,88),dtype=np.uint8);np.save(Path(td)/'tiles.npy',tile)
   kw=dict(tiles_dir=td,indices=[0],margin_px=40,crop_px=8,seed=77,d_over_r0_range=(3,3))
   datasets=[]
   for lv in ['d3','d3n']:
    deg=LEVELS[lv](2);datasets.append(pairs_class(deg)(degradation=deg,**kw))
   a=datasets[0][0];b=datasets[1][0]
   for i in [0,1,2,3]:np.testing.assert_array_equal(a[i],b[i])
   self.assertGreater(np.std(b[4]),0)
   from torch.utils.data import default_collate
   x,y,_=render_batch(default_collate([a]),datasets[0].deg,torch.device('cpu'))
   expected=datasets[0].deg.renderer()(a[0],a[3]).numpy()[40:48,40:48]
   np.testing.assert_allclose(x[0,0],expected,atol=2e-6)
   datasets[0].set_epoch(1);self.assertFalse(np.array_equal(a[3],datasets[0][0][3]))
   frozen=pairs_class(datasets[0].deg,True)(degradation=datasets[0].deg,**kw)
   c=frozen[0];frozen.set_epoch(99);np.testing.assert_array_equal(c[3],frozen[0][3])

 @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
 def test_cuda_parity(self):
  wf=Kolmogorov(2);rng=np.random.default_rng(5);img=rng.random((48,48),dtype=np.float32);a=rng.normal(0,.1,(35,48,48)).astype(np.float32)
  cpu=ExactRenderer(wf,'cpu')(img,a,crop=20).numpy();gpu=ExactRenderer(wf,'cuda')(img,a,crop=20).cpu().numpy()
  np.testing.assert_allclose(cpu,gpu,atol=1e-5,rtol=1e-5)

if __name__=='__main__':unittest.main(verbosity=2)
