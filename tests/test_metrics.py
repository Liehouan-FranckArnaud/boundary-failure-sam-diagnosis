"""tests/test_metrics.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pytest
from src.evaluation.metrics import (
    compute_metrics, boundary_evaluation,
    boundary_vs_global_summary, wilson_ci,
)

def test_perfect_sam_recall():
    y=np.array([0,1,2,2,2]); m=compute_metrics(y,y)
    assert m["sam_recall"]==1.0 and m["fn_sam"]==0

def test_zero_sam_recall():
    y=np.array([2,2,2]); p=np.array([0,1,1])
    m=compute_metrics(y,p); assert m["sam_recall"]==0.0 and m["fn_sam"]==3

def test_kwashiorkor_recall():
    y=np.array([2,2,2,0]); p=np.array([2,0,2,0]); oe=np.array([True,True,False,False])
    m=compute_metrics(y,p,oedema=oe); assert m["kwashiorkor_recall"]==0.5

def test_boundary_evaluation_smoke():
    rng=np.random.default_rng(0); n=500
    y=rng.choice([0,1,2],n,p=[0.68,.20,.12]); p=y.copy()
    df=boundary_evaluation(y,p,rng.normal(-2,1.5,n),rng.normal(125,15,n))
    assert len(df)>0 and "sam_recall" in df.columns

def test_wilson_ci_contains_point():
    for k,n in [(5,47),(46,47),(47,47),(0,47)]:
        r,lo,hi=wilson_ci(k,n); assert lo<=r<=hi

def test_no_sam_in_zone():
    y=np.array([0,1,0,1]); p=np.array([0,1,0,1])
    whz=np.array([1.,2.,3.,4.]); muac=np.array([130.,130.,130.,130.])
    df=boundary_evaluation(y,p,whz,muac)
    # No SAM children at all -> boundary_evaluation returns an empty
    # DataFrame (no zone has sam_support > 0, so no rows are added).
    assert df.empty

def test_boundary_vs_global_summary():
    rng=np.random.default_rng(1); n=800
    y=rng.choice([0,1,2],n,p=[0.68,.20,.12]); p=y.copy()
    whz=rng.normal(-2,1.5,n); muac=rng.normal(125,15,n)
    g=compute_metrics(y,p); df=boundary_evaluation(y,p,whz,muac)
    s=boundary_vs_global_summary(g,df)
    assert "global_sam_recall" in s and "recall_gap" in s
