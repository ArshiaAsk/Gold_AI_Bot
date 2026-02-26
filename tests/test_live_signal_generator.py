import api_layer.live_signal_generator as live_signal_module


class _DummyLogger:
    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None


class _DummyFeatureEngineer:
    def __init__(self, features):
        self._features = features

    def get_cached_features(self):
        return self._features


def _prediction_payload(predicted_log_return: float, confidence: float = 0.9):
    return {
        "predicted_log_return": predicted_log_return,
        "predicted_return_pct": predicted_log_return * 100,
        "confidence": confidence,
        "current_price": 1000.0,
    }


def test_generate_signal_returns_hold_for_low_confidence(monkeypatch):
    monkeypatch.setattr(live_signal_module.LiveSignalGenerator, "_setup_logger", lambda self: _DummyLogger())
    monkeypatch.setattr(
        live_signal_module,
        "LiveFeatureEngineer",
        lambda: _DummyFeatureEngineer({"RSI_14": 50, "MACD": 1, "SMA_7": 900, "SMA_30": 850}),
    )

    generator = live_signal_module.LiveSignalGenerator()
    signal = generator.generate_signal(_prediction_payload(predicted_log_return=0.02, confidence=0.1))

    assert signal["action"] == live_signal_module.SignalType.HOLD.value
    assert any("Low confidence" in reason for reason in signal["reasoning"])


def test_generate_signal_returns_buy_when_conditions_are_met(monkeypatch):
    monkeypatch.setattr(live_signal_module.LiveSignalGenerator, "_setup_logger", lambda self: _DummyLogger())
    monkeypatch.setattr(
        live_signal_module,
        "LiveFeatureEngineer",
        lambda: _DummyFeatureEngineer({"RSI_14": 55, "MACD": 5, "SMA_7": 900, "SMA_30": 850}),
    )

    generator = live_signal_module.LiveSignalGenerator()
    signal = generator.generate_signal(_prediction_payload(predicted_log_return=0.02, confidence=0.9))

    assert signal["action"] == live_signal_module.SignalType.BUY.value
    assert signal["strength"] > 0
