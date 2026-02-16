def test_signal_generator():
    mock_prediction = {'predicted_log_return': 0.05}
    signal = generator.generate_signal(mock_prediction)
    assert signal is not None  # Replace with actual assertions based on expected signal output