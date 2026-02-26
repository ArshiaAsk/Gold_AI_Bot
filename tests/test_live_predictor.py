import pandas as pd

from api_layer.live_predictor import LivePredictor


class _DummyLogger:
    def info(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None


def _predictor_stub() -> LivePredictor:
    predictor = LivePredictor.__new__(LivePredictor)
    predictor.lookback = 4
    predictor.feature_columns = ["Gold_LogRet", "USD_LogRet"]
    predictor.logger = _DummyLogger()
    return predictor


def test_estimate_confidence_thresholds():
    predictor = _predictor_stub()

    assert predictor._estimate_confidence(0.025) == 0.85
    assert predictor._estimate_confidence(0.015) == 0.75
    assert predictor._estimate_confidence(0.007) == 0.65
    assert predictor._estimate_confidence(0.002) == 0.55


def test_build_live_feature_sequence_appends_latest_row():
    predictor = _predictor_stub()
    predictor.get_historical_features = lambda days: pd.DataFrame(
        [
            {"Date": "2026-01-01", "Gold_LogRet": 0.01, "USD_LogRet": 0.02},
            {"Date": "2026-01-02", "Gold_LogRet": 0.02, "USD_LogRet": 0.03},
            {"Date": "2026-01-03", "Gold_LogRet": 0.03, "USD_LogRet": 0.04},
        ]
    )

    latest_prices = {"Gold_IRR": 1.0, "USD_IRR": 1.0, "Ounce_USD": 1.0, "Oil_USD": 1.0}
    latest_features = {"Gold_LogRet": 0.09, "USD_LogRet": 0.08}

    output = predictor._build_live_feature_sequence(
        latest_prices=latest_prices,
        latest_features=latest_features,
    )

    assert output is not None
    assert len(output) == predictor.lookback
    assert output.iloc[-1]["Gold_LogRet"] == 0.09
    assert output.iloc[-1]["USD_LogRet"] == 0.08
