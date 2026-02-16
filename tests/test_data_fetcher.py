def test_generate_signal():
    mock_prediction = {'predicted_log_return': 0.05}
    generator = SignalGenerator()  # Assuming SignalGenerator is the class containing generate_signal
    signal = generator.generate_signal(mock_prediction)
    assert signal is not None  # Adjust the assertion based on expected signal output