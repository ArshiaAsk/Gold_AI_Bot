from prometheus_client import Counter, Histogram, Gauge

prediction_latency = Histogram('prediction_latency_seconds', 'prediction latency')
signal_counter = Counter ('trading_signals_total', 'Trading signals', ['action'])
confidence_gauge = Gauge('signal_confidence', 'Signal confidence')