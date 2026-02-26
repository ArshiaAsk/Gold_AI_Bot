from api_layer.data_fetcher import TGJUDataFetcher


def test_clean_value_handles_strings_and_markup():
    assert TGJUDataFetcher.clean_value("1,234,567") == 1234567.0
    assert TGJUDataFetcher.clean_value("<span>12.5%</span>") == 12.5
    assert TGJUDataFetcher.clean_value("not-a-number") == 0.0


def test_clean_value_handles_null_and_numeric():
    assert TGJUDataFetcher.clean_value(None) == 0.0
    assert TGJUDataFetcher.clean_value(42) == 42.0
    assert TGJUDataFetcher.clean_value(3.14) == 3.14
