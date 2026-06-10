"""
Prometheus Metrics Exporter
Exposes system and application metrics for monitoring
"""

from prometheus_client import start_http_server, Gauge, Counter, Histogram
import time
import psutil
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# System metrics
cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
memory_usage = Gauge('system_memory_usage_percent', 'Memory usage percentage')
disk_usage = Gauge('system_disk_usage_percent', 'Disk usage percentage')

# Application metrics
service_uptime = Gauge('service_uptime_seconds', 'Service uptime in seconds')
active_requests = Gauge('active_requests', 'Number of active requests')
total_requests = Counter('requests_total', 'Total number of requests')

# Model metrics
model_predictions = Counter('model_predictions_total', 'Total model predictions')
model_latency = Histogram('model_latency_seconds', 'Model inference latency')
model_errors = Counter('model_errors_total', 'Total model errors')

def collect_system_metrics():
    """Collect system resource metrics"""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_usage.set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage.set(memory.percent)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage.set(disk.percent)
        
        logger.debug(f"Metrics: CPU={cpu_percent}%, Memory={memory.percent}%, Disk={disk.percent}%")
    except Exception as e:
        logger.error(f"Error collecting metrics: {e}")

def main():
    """Start Prometheus exporter"""
    port = 9100
    
    try:
        start_http_server(port)
        logger.info(f"Prometheus exporter started on port {port}")
        logger.info(f"Metrics available at http://localhost:{port}/metrics")
        
        start_time = time.time()
        
        while True:
            # Update uptime
            uptime = time.time() - start_time
            service_uptime.set(uptime)
            
            # Collect system metrics
            collect_system_metrics()
            
            # Sleep before next collection
            time.sleep(10)
    
    except Exception as e:
        logger.error(f"Exporter error: {e}")
        raise

if __name__ == "__main__":
    main()
