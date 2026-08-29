"""tests/test_oracle.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pytest
from experiments.exp_d_oracle import oracle_gating
from src.evaluation.metrics import compute_metrics

def _mock(n=400, seed=0):
    rng=np.random.default_rng(seed); y=rng.choice([0,1,2],n,p=[0.80,.14,.06])
    pred=y.copy(); sam=np.where(y==2)[0]
    if len(sam)>2: pred[sam[:2]]=1
    oe=np.zeros(n,bool)
    if len(sam)>0: oe[sam[:max(1,len(sam)//3)]]=True
    return y,pred,oe

def test_oracle_100_detects_all():
    y,pred,oe=_mock(500); g=oracle_gating(pred,oe,1.0,42)
    assert np.all(g[oe]==2)

def test_oracle_0_unchanged():
    y,pred,oe=_mock(500); g=oracle_gating(pred.copy(),oe,0.0,42)
    np.testing.assert_array_equal(pred,g)

def test_non_oedema_unchanged():
    y,pred,oe=_mock(500); g=oracle_gating(pred.copy(),oe,1.0)
    np.testing.assert_array_equal(pred[~oe],g[~oe])

def test_monotone_recall():
    y,pred,oe=_mock(1000,seed=7)
    recalls=[compute_metrics(y,oracle_gating(pred.copy(),oe,a,42))["sam_recall"]
             for a in [0.0,0.5,0.8,1.0]]
    assert all(recalls[i]<=recalls[i+1]+1e-9 for i in range(len(recalls)-1))
