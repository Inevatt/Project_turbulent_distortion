import numpy as np
from scipy.integrate import quad
from scipy.special import jv
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.distortion.wavefront import noll_nm


def main():
    s=np.r_[np.linspace(0,5,201),np.geomspace(5.05,100,120)]
    out={};start=time.time()
    for n,m in sorted(set((n,abs(m)) for n,m in [noll_nm(j) for j in range(2,37)])):
     for order in set([0,2*m]):
      key=f'n{n}_o{order}'
      if key in out: continue
      def f(z,sep):
       return z**(-14/3)*jv(n+1,z)**2*jv(order,2*sep*z)
      vals=[]
      for sep in s:
       vals.append(quad(f,0,150,args=(sep,),epsabs=1e-13,epsrel=1e-7,limit=3000,points=np.arange(2.,150.,2.))[0])
      out[key]=np.array(vals)
      print(key,round(time.time()-start,1),flush=True)
    np.savez_compressed(Path(__file__).resolve().parents[1]/'src/distortion/assets/modal_corr.npz',s=s,**out)


if __name__ == '__main__':
    main()
