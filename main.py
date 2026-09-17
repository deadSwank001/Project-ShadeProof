# Copyright (c) 2026 David Burmeister. All rights reserved.
# wraps PyQT to Neptune-backend

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QPushButton, QLabel, QProgressBar, QTextEdit
)
from backend import ExperimentWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Surrogate Model Adversarial Evaluation")
        self.resize(500, 350)

        layout = QVBoxLayout()
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)

        self.start_button = QPushButton("Launch Experiment")
        self.start_button.clicked.connect(self.start_experiment)

        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.start_button)
        layout.addWidget(self.log_area)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.worker = None

    def start_experiment(self):
        config = {
            "neptune_project": "workspace/adversarial-benchmarks",
            "iterations": 100,
            "lr": 1e-3,
            "target_prompt": "a leather handbag",
        }

        self.worker = ExperimentWorker(config=config)
        self.worker.progress.connect(self.update_progress)
        self.worker.status_updated.connect(self.log_message)
        self.worker.metric_logged.connect(self.log_metric)
        self.worker.finished.connect(lambda: self.start_button.setEnabled(True))
        self.worker.error_occurred.connect(self.log_error)

        self.start_button.setEnabled(False)
        self.worker.start()

    def update_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def log_message(self, text):
        self.status_label.setText(text)
        self.log_area.append(f"[INFO] {text}")

    def log_metric(self, name, val):
        self.log_area.append(f"[METRIC] {name}: {val:.4f}")

    def log_error(self, err):
        self.log_area.append(f"[ERROR] {err}")
        self.start_button.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
