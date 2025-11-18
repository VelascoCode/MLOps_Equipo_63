import pandas as pd
from mlops_equipo_63.pipeline import MLOpsPipeline
from mlops_equipo_63.Configuration import Config


def test_pipeline_load_and_prepare(tmp_path):
    df = pd.DataFrame({"shares": [1, 2, 3], "a": [1.0, 2.0, 3.0]})
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)

    cfg = Config()
    cfg.data_path = str(p)
    pl = MLOpsPipeline(cfg)
    pl.load()
    assert pl.df_raw is not None
    pl.prepare()
    assert pl.df_numeric is not None
