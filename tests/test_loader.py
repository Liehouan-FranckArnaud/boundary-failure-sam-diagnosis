"""tests/test_loader.py"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd, pytest
from src.data.loader import _load_csv, TRAIN_CONFIGS, TEST_CONFIG, VAL_CONFIG

def _make_csv(n=200):
    rng = np.random.default_rng(0)
    lbl = rng.choice(["Normal","MAM","SAM"], n, p=[0.84,0.10,0.06])
    df  = pd.DataFrame({
        "age_months": rng.integers(6,60,n), "sex": rng.choice(["M","F"],n),
        "muac_cm": np.where(lbl=="SAM", rng.normal(10.9,.5,n),
                   np.where(lbl=="MAM", rng.normal(12.0,.5,n), rng.normal(14.,1.,n))),
        "whz": np.where(lbl=="SAM", rng.normal(-3.8,.4,n),
               np.where(lbl=="MAM", rng.normal(-2.5,.3,n), rng.normal(-.3,.8,n))),
        "edema": ((lbl=="SAM")&(rng.random(n)<0.15)).astype(int),
        "nutritional_status": lbl,
    })
    p = tempfile.mktemp(suffix=".csv"); df.to_csv(p,index=False); return p

@pytest.fixture
def dataset(): return _load_csv(_make_csv(), "mock")

def test_shape(dataset):         assert len(dataset[0]) == 200
def test_muac_column_name(dataset): assert "muac_mm" in dataset[0].columns
def test_muac_in_mm(dataset):    assert dataset[0]["muac_mm"].median() > 50
def test_label_range(dataset):   assert set(np.unique(dataset[1])).issubset({0,1,2})
def test_oedema_bool(dataset):   assert dataset[2].dtype == bool
def test_feature_columns(dataset):
    for c in ["whz","muac_mm","age_months","sex_bin"]:
        assert c in dataset[0].columns
def test_sex_binary(dataset):    assert set(dataset[0]["sex_bin"].unique()).issubset({0.,1.})
def test_split_constants():
    assert "low_burden" in TRAIN_CONFIGS and "moderate_burden" in TRAIN_CONFIGS
    assert TEST_CONFIG == "high_burden" and VAL_CONFIG == "crisis"
