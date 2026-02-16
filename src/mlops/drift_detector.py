from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

class DriftDetector:
    def __init__(self, reference_data):
        self.reference_data = reference_data

    def check_drift(self, current_data):
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=self.reference_data,
                   current_data=current_data)
        return report.as_dict()