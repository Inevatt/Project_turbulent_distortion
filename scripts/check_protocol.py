"""Small independent checks for experiment contracts and the real training loop."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import contextlib
import copy
import csv
import io
import tempfile
import socket
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from src.data import DegradedPairs, FrozenDegraded, split_indices
from src.distortion import LEVELS
from src.metrics import psnr, ssim
from src.unet import UNet
from src.eval_matrix import atomic_npz, run_models, run_column, main as evaluate
from scripts.summarize_results import retained, main as summarize
from src import train

torch.set_num_threads(1)

class ProtocolChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        tiles = np.random.default_rng(1).integers(0,256,(40,96,96),dtype=np.uint8)
        np.save(self.root/'tiles.npy',tiles)
        np.save(self.root/'source_id.npy',np.repeat(np.arange(20),2))

    def tearDown(self):
        self.tmp.cleanup()

    def dataset(self, level='d0', frozen=False):
        cls = FrozenDegraded if frozen else DegradedPairs
        return cls(self.root,[0,2,4,6],LEVELS[level](2),margin_px=40,crop_px=16,seed=23)

    def test_group_split_and_invalid_data(self):
        parts=split_indices(self.root)
        sid=np.load(self.root/'source_id.npy')
        sets=[set(sid[x]) for x in parts]
        self.assertEqual(len(np.concatenate(parts)),40)
        for i in range(3):
            for j in range(i): self.assertFalse(sets[i]&sets[j])
        with self.assertRaises(ValueError):split_indices(self.root,.7,.4)
        with self.assertRaises(ValueError):DegradedPairs(self.root,[0],LEVELS['d0'](2),0,16)

    def test_workers(self):
        try:
            probe=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);probe.close()
        except PermissionError:
            self.skipTest('runtime blocks local sockets required by multiprocessing; run on server')
        ds=self.dataset()
        a=list(DataLoader(ds,batch_size=2,num_workers=0))
        b=list(DataLoader(ds,batch_size=2,num_workers=2))
        for x,y in zip(a,b):
            for u,v in zip(x,y): torch.testing.assert_close(u,v,rtol=0,atol=0)
    def test_epoch_and_freeze(self):
        ds=self.dataset()
        old=ds[0];ds.set_epoch(1)
        self.assertFalse(np.array_equal(old[0],ds[0][0]))
        frozen=self.dataset(frozen=True);old=frozen[0];frozen.set_epoch(99)
        np.testing.assert_array_equal(old[0],frozen[0][0])

    def test_integer_registration_and_rng_pairing(self):
        class Translation:
            def support_radius_px(self,dr):return 4
            def __call__(self,img,dr,rng):return np.roll(img,(3,-2),(0,1)),(3,-2)
        ds=DegradedPairs(self.root,[0],Translation(),margin_px=40,crop_px=16)
        x,y,_=ds[0];np.testing.assert_array_equal(x,y)
        base=self.dataset()
        for lv in ['d1','d15','d2']:
            self.assertEqual(base[0][2],self.dataset(lv)[0][2])

    def test_model_and_metrics(self):
        net=UNet(base=4,depth=2)
        x=torch.rand(2,1,17,19);torch.testing.assert_close(net(x),x,atol=0,rtol=0)
        target=np.zeros((2,1,16,16));pred=np.full_like(target,.1)
        np.testing.assert_allclose(psnr(pred,target),20)
        np.testing.assert_allclose(ssim(target,target),1)
        r,d=retained(31,28,30);self.assertEqual(r,1.5);self.assertEqual(d,2)
        self.assertIsNone(retained(31,28,28)[0])

    def test_real_inference_and_registration(self):
        from src.eval_real import align_pair, predict_patches
        rng=np.random.default_rng(33)
        reference=rng.random((31,35),dtype=np.float32)
        image=np.roll(reference,(3,-2),(0,1))
        x,y,shift=align_pair(image,reference,'integer')
        self.assertEqual(shift,(-3,2));np.testing.assert_array_equal(x,y)
        net=UNet(base=4,depth=1).eval()
        for img in (reference, reference[:11,:13]):
            out=predict_patches(net,img,16,torch.device('cpu'),2)
            np.testing.assert_allclose(out,img,atol=1e-7)

    def test_shared_evaluation_equivalence(self):
        net=UNet(base=4,depth=1).eval()
        with torch.no_grad():net.head.bias.fill_(.03)
        ds=self.dataset(frozen=True);device=torch.device('cpu')
        got=run_models({'noop':None,'model':net},ds,2,0,device)
        for row,model in [('noop',None),('model',net)]:
            reference=run_column(model,ds,2,0,device)
            for x,y in zip(got[row],reference):np.testing.assert_array_equal(x,y)

    def test_configs_only_three_differences(self):
        configs=[yaml.safe_load(Path(f'configs/seed{s}.yaml').read_text()) for s in range(1,6)]
        for cfg in configs[1:]:
            changed={k for k in configs[0] if configs[0][k]!=cfg[k]}
            self.assertEqual(changed,{'seed','split_seed','out_dir'})
        self.assertEqual(configs[0]['train']['epochs'],60)

    def test_training_resume_and_evaluation_cli(self):
        cfg=yaml.safe_load(Path('configs/seed1.yaml').read_text())
        cfg['data'].update(tiles=str(self.root),crop_px=16)
        cfg['train'].update(epochs=2,batch_size=8,num_workers=0)
        cfg['model'].update(base=4,depth=1)
        cfg['out_dir']=str(self.root/'experiments_smoke')
        config=self.root/'config.yaml';config.write_text(yaml.safe_dump(cfg))
        first=[];save=torch.save
        def capture(state,path):
            if state['epoch']==1:first.append(copy.deepcopy(state))
            save(state,path)
        argv=['train','--config',str(config),'--level','d0']
        with contextlib.redirect_stdout(io.StringIO()),patch('sys.argv',argv),patch.object(torch,'save',capture):train.main()
        ckpt=Path(cfg['out_dir'])/'d0/last.pt'
        complete=torch.load(ckpt,weights_only=False)
        save(first[0],ckpt)  # Simulate a crash with an extra log row.
        with contextlib.redirect_stdout(io.StringIO()),patch('sys.argv',argv+['--resume']):train.main()
        resumed=torch.load(ckpt,weights_only=False)
        for k in complete['model']:torch.testing.assert_close(complete['model'][k],resumed['model'][k],rtol=0,atol=0)
        self.assertEqual(complete['sched'],resumed['sched'])
        with (ckpt.parent/'log.csv').open() as f:self.assertEqual([int(r['epoch']) for r in csv.DictReader(f)],[1,2])
        # Tiny evaluator integration; deliberately reuse test weights for all
        # rows. These files are temporary wiring fixtures, never experiment results.
        for level in LEVELS:
            state=copy.deepcopy(complete);state['level']=level
            dest=Path(cfg['out_dir'])/level/'last.pt';dest.parent.mkdir(exist_ok=True)
            save(state,dest)
        output=self.root/'results'
        args=['eval','--config',str(config),'--output',str(output),'--columns','d0','--dr-grid','1']
        with contextlib.redirect_stdout(io.StringIO()),patch('sys.argv',args):evaluate()
        with contextlib.redirect_stdout(io.StringIO()),patch('sys.argv',args),patch('src.eval_matrix.run_models',side_effect=AssertionError('cache missed')):evaluate()
        args=['summary','--configs',str(config),'--results',str(output),'--output',str(output/'summary.json')]
        with contextlib.redirect_stdout(io.StringIO()),patch('sys.argv',args):summarize()
        self.assertTrue((output/'summary.json').exists())
        from scripts.bench_training import main as bench
        args=['bench','--config',str(config),'--level','d0','--device','cpu','--steps','1','--warmup','0','--val-steps','1','--output',str(output/'bench.json')]
        with contextlib.redirect_stdout(io.StringIO()),patch('sys.argv',args):bench()
        self.assertTrue((output/'bench.json').exists())


if __name__=='__main__':unittest.main(verbosity=2)
