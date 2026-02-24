import time

from src.mlops.background_tasks import SchedulerBackgroundService


class DummyScheduler:
    def __init__(self):
        self.configure_calls = 0
        self.ticks = 0

    def configure(self):
        self.configure_calls += 1

    def run_pending_once(self):
        self.ticks += 1


def test_scheduler_background_service_start_stop():
    scheduler = DummyScheduler()
    service = SchedulerBackgroundService(scheduler=scheduler, poll_seconds=0)

    service.start()
    time.sleep(0.02)
    status_running = service.status()
    service.stop(timeout_seconds=1)
    status_stopped = service.status()

    assert scheduler.configure_calls == 1
    assert scheduler.ticks > 0
    assert status_running["running"] is True
    assert status_stopped["running"] is False


def test_scheduler_background_service_start_is_idempotent():
    scheduler = DummyScheduler()
    service = SchedulerBackgroundService(scheduler=scheduler, poll_seconds=0)

    service.start()
    service.start()
    service.stop(timeout_seconds=1)

    assert scheduler.configure_calls == 1
