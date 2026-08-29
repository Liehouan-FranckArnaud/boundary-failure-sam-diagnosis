"""tests/test_missingness.py"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd, pytest
from src.data.loader import _load_csv
from src.missingness.protocols import inject_missingness, sc3_probabilities, MISSING_FEATURES, ALL_SCENARIOS

def _mock(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    lbl = rng.choice(["Normal","MAM","SAM"], n, p=[0.80,0.14,0.06])
    df  = pd.DataFrame({
        "age_months": rng.integers(6,60,n), "sex": rng.choice(["M","F"],n),
        "muac_cm": np.where(lbl=="SAM", rng.normal(10.9,.5,n),
                   np.where(lbl=="MAM", rng.normal(12.,0.5,n), rng.normal(14.,1.,n))),
        "whz": np.where(lbl=="SAM", rng.normal(-3.8,.4,n),
               np.where(lbl=="MAM", rng.normal(-2.5,.3,n), rng.normal(-.3,.8,n))),
        "edema": ((lbl=="SAM")&(rng.random(n)<0.15)).astype(int),
        "nutritional_status": lbl,
    })
    p = tempfile.mktemp(suffix=".csv"); df.to_csv(p,index=False)
    X,y,oe = _load_csv(p,"mock")
    assert "muac_mm" in X.columns
    return X,y,oe

@pytest.fixture
def sample_data(): return _mock(5000)

def test_sc0_no_missing(sample_data):
    X,y,_ = sample_data; Xm = inject_missingness(X,y,"Sc0",0.30)
    assert Xm["whz"].isna().sum()==0 and Xm["muac_mm"].isna().sum()==0

def test_sc0_returns_copy(sample_data):
    X,y,_ = sample_data; assert inject_missingness(X,y,"Sc0") is not X

def test_sc1_near_uniform(sample_data):
    X,y,_ = sample_data; Xm = inject_missingness(X,y,"Sc1",0.30,10)
    assert abs(Xm.loc[y==2,"whz"].isna().mean() - Xm.loc[y!=2,"whz"].isna().mean()) < 0.15

def test_sc3_ratio(sample_data):
    X,y,_ = sample_data; Xm = inject_missingness(X,y,"Sc3",0.30,10)
    r = Xm.loc[y==2,"whz"].isna().mean() / Xm.loc[y!=2,"whz"].isna().mean()
    assert 1.1 < r < 4.0

def test_sc3_marginal_rate(sample_data):
    X,y,_ = sample_data; Xm = inject_missingness(X,y,"Sc3",0.30,10)
    assert abs(Xm["whz"].isna().mean() - 0.30) < 0.08

def test_sc3_muac_also_affected(sample_data):
    X,y,_ = sample_data; Xm = inject_missingness(X,y,"Sc3",0.30,10)
    assert Xm["whz"].isna().sum()>0 and Xm["muac_mm"].isna().sum()>0

def test_sc3_formula():
    y = np.array([0,1,2,2,0]); r = 0.30
    p = sc3_probabilities(y, r)
    np.testing.assert_allclose(p[y==2], 2*r/(1+r), rtol=1e-6)
    np.testing.assert_allclose(p[y!=2], r/(1+r),   rtol=1e-6)

def test_missing_features_canonical():
    assert "muac_mm" in MISSING_FEATURES and "whz" in MISSING_FEATURES

def test_all_scenarios_defined(): assert set(ALL_SCENARIOS)=={"Sc0","Sc1","Sc2","Sc3"}

def test_unknown_scenario_raises(sample_data):
    X,y,_ = sample_data
    with pytest.raises(ValueError): inject_missingness(X,y,"Sc99")

def test_sc3_probabilities_dict():
    info = sc3_probabilities(0.30, 0.059)
    assert "p_sam" in info and "marginal" in info
    assert abs(info["marginal"] - 0.30*(1+0.059)/(1+0.30)) < 0.001
