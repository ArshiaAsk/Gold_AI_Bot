def pytest_configure():
    import pytest

    pytest.mock_prediction = {
        'predicted_log_return': 0.01
    }