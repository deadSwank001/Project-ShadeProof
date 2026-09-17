"""
Project-ShadeProof: Academic Evaluation Desktop Interface
Copyright (c) 2026 David Burmeister. All rights reserved.
Licensed under the Apache License, Version 2.0.

PyQt6 Academic GUI for Surrogate Model Adversarial Analysis,
Data Sanitization Benchmarking, and Perceptual Residual Visualization.
"""

import sys
import numpy as np
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QPushButton, QLabel, QProgressBar,
    QTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QFont

from backend import ExperimentWorker


class ImageDisplayWidget(QWidget):
    """Component for displaying normalized [H, W, 3] float32 image tensors."""

    def __init__(self, title: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)

        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 11px;")

        self.image_label = QLabel("No Data")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(180, 180)
        self.image_label.setStyleSheet("border: 1px solid #aaa; background-color: #f0f0f0;")

        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label)
        self.setLayout(layout)

    def set_image_array(self, arr: np.ndarray):
        """Converts [H, W, 3] float32 array in [0, 1] to QPixmap."""
        h, w, c = arr.shape
        byte_data = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        bytes_per_line = c * w
        q_img = QImage(byte_data.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Project-ShadeProof | Surrogate Adversarial Evaluation & Sanitization")
        self.resize(1050, 700)

        self.worker: Optional[ExperimentWorker] = None
        self._init_ui()

    def _init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # ======================================================================
        # LEFT PANEL: Parameters & Controls
        # ======================================================================
        left_panel = QVBoxLayout()

        # Threat Configuration
        config_group = QGroupBox("Adversarial Threat Configuration")
        config_layout = QGridLayout()

        config_layout.addWidget(QLabel("Source Prompt:"), 0, 0)
        self.input_src_prompt = QLineEdit("a photograph of a dog")
        config_layout.addWidget(self.input_src_prompt, 0, 1)

        config_layout.addWidget(QLabel("Target Concept:"), 1, 0)
        self.input_tgt_prompt = QLineEdit("a luxury leather handbag")
        config_layout.addWidget(self.input_tgt_prompt, 1, 1)

        config_layout.addWidget(QLabel("Epsilon Budget (L_inf):"), 2, 0)
        self.input_eps = QDoubleSpinBox()
        self.input_eps.setRange(0.005, 0.15)
        self.input_eps.setSingleStep(0.005)
        self.input_eps.setValue(8.0 / 255.0)
        self.input_eps.setDecimals(4)
        config_layout.addWidget(self.input_eps, 2, 1)

        config_layout.addWidget(QLabel("Step Size (Alpha):"), 3, 0)
        self.input_alpha = QDoubleSpinBox()
        self.input_alpha.setRange(0.001, 0.05)
        self.input_alpha.setSingleStep(0.002)
        self.input_alpha.setValue(2.0 / 255.0)
        self.input_alpha.setDecimals(4)
        config_layout.addWidget(self.input_alpha, 3, 1)

        config_layout.addWidget(QLabel("PGD Iterations:"), 4, 0)
        self.input_iters = QSpinBox()
        self.input_iters.setRange(5, 300)
        self.input_iters.setValue(50)
        config_layout.addWidget(self.input_iters, 4, 1)

        config_group.setLayout(config_layout)
        left_panel.addWidget(config_group)

        # Defense & Sanitization Layer
        defense_group = QGroupBox("Dataset Defense & Sanitization")
        defense_layout = QGridLayout()

        defense_layout.addWidget(QLabel("Sanitization Filter:"), 0, 0)
        self.combo_defense = QComboBox()
        self.combo_defense.addItems([
            "none",
            "jpeg_compression",
            "gaussian_blur",
            "median_filter",
            "total_variation"
        ])
        defense_layout.addWidget(self.combo_defense, 0, 1)

        defense_layout.addWidget(QLabel("Filter Strength:"), 1, 0)
        self.input_defense_strength = QDoubleSpinBox()
        self.input_defense_strength.setRange(0.1, 3.0)
        self.input_defense_strength.setSingleStep(0.2)
        self.input_defense_strength.setValue(1.0)
        defense_layout.addWidget(self.input_defense_strength, 1, 1)

        defense_group.setLayout(defense_layout)
        left_panel.addWidget(defense_group)

        # Experiment Tracking (Neptune)
        tracking_group = QGroupBox("Experiment Tracking")
        tracking_layout = QVBoxLayout()

        self.chk_neptune = QCheckBox("Enable Neptune Cloud Telemetry")
        tracking_layout.addWidget(self.chk_neptune)

        self.input_neptune_project = QLineEdit("workspace/adversarial-benchmarks")
        self.input_neptune_project.setPlaceholderText("workspace/project-name")
        tracking_layout.addWidget(self.input_neptune_project)

        tracking_group.setLayout(tracking_layout)
        left_panel.addWidget(tracking_group)

        # Action Buttons
        self.btn_launch = QPushButton("Launch Adversarial Evaluation")
        self.btn_launch.setStyleSheet("font-weight: bold; padding: 8px; background-color: #2b5797; color: white;")
        self.btn_launch.clicked.connect(self.start_experiment)
        left_panel.addWidget(self.btn_launch)

        self.btn_cancel = QPushButton("Cancel Evaluation")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_experiment)
        left_panel.addWidget(self.btn_cancel)

        left_panel.addStretch()

        # ======================================================================
        # RIGHT PANEL: Tabs (Convergence, Visual Inspection, Empirical Table)
        # ======================================================================
        right_panel = QVBoxLayout()

        self.status_label = QLabel("Status: System Ready")
        self.status_label.setStyleSheet("font-weight: bold; color: #333;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        right_panel.addWidget(self.status_label)
        right_panel.addWidget(self.progress_bar)

        self.tab_widget = QTabWidget()

        # Tab 1: Live Telemetry & Execution Log
        tab_log = QWidget()
        tab_log_layout = QVBoxLayout()

        # Real-time metric indicators
        card_layout = QHBoxLayout()
        self.card_tgt_cos = QLabel("Target Cosine: --")
        self.card_src_cos = QLabel("Source Cosine: --")
        self.card_linf = QLabel("L_inf: --")
        self.card_psnr = QLabel("PSNR: --")

        for card in [self.card_tgt_cos, self.card_src_cos, self.card_linf, self.card_psnr]:
            card.setStyleSheet("background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; padding: 6px; font-weight: bold;")
            card_layout.addWidget(card)

        tab_log_layout.addLayout(card_layout)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFont(QFont("Courier New", 9))
        tab_log_layout.addWidget(self.log_area)

        tab_log.setLayout(tab_log_layout)
        self.tab_widget.addTab(tab_log, "Convergence & Telemetry")

        # Tab 2: Perceptual Visual Inspector
        tab_visual = QWidget()
        tab_visual_layout = QHBoxLayout()

        self.disp_clean = ImageDisplayWidget("Clean Input (x)")
        self.disp_adv = ImageDisplayWidget("Adversarial Sample (x_adv)")
        self.disp_sanitized = ImageDisplayWidget("Sanitized Defense (x_san)")
        self.disp_noise = ImageDisplayWidget("Residual Noise (10x |delta|)")

        tab_visual_layout.addWidget(self.disp_clean)
        tab_visual_layout.addWidget(self.disp_adv)
        tab_visual_layout.addWidget(self.disp_sanitized)
        tab_visual_layout.addWidget(self.disp_noise)

        tab_visual.setLayout(tab_visual_layout)
        self.tab_widget.addTab(tab_visual, "Perceptual Inspection")

        # Tab 3: Empirical Benchmark Table
        tab_table = QWidget()
        tab_table_layout = QVBoxLayout()

        self.table_results = QTableWidget()
        self.table_results.setColumnCount(2)
        self.table_results.setHorizontalHeaderLabels(["Scientific Evaluation Metric", "Quantified Value"])
        self.table_results.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table_results.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        tab_table_layout.addWidget(self.table_results)

        tab_table.setLayout(tab_table_layout)
        self.tab_widget.addTab(tab_table, "Benchmark Metrics")

        right_panel.addWidget(self.tab_widget)

        # Assemble main window
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 2)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def start_experiment(self):
        config = {
            "source_prompt": self.input_src_prompt.text().strip(),
            "target_prompt": self.input_tgt_prompt.text().strip(),
            "epsilon": self.input_eps.value(),
            "alpha": self.input_alpha.value(),
            "iterations": self.input_iters.value(),
            "defense_type": self.combo_defense.currentText(),
            "defense_strength": self.input_defense_strength.value(),
            "use_neptune": self.chk_neptune.isChecked(),
            "neptune_project": self.input_neptune_project.text().strip(),
        }

        self.btn_launch.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.log_area.clear()
        self.progress_bar.setValue(0)
        self.status_label.setText("Status: Launching Evaluation Engine...")

        self.worker = ExperimentWorker(config=config)
        self.worker.step_progress.connect(self.update_step_progress)
        self.worker.status_updated.connect(self.log_status)
        self.worker.metric_logged.connect(self.log_metric)
        self.worker.experiment_finished.connect(self.handle_experiment_finished)
        self.worker.error_occurred.connect(self.handle_error)

        self.worker.start()

    def cancel_experiment(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.status_label.setText("Status: Canceling experiment...")
            self.btn_cancel.setEnabled(False)

    def update_step_progress(self, step, total, loss, tgt_cos, src_cos, linf):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(step)
        self.card_tgt_cos.setText(f"Target Cosine: {tgt_cos:.4f}")
        self.card_src_cos.setText(f"Source Cosine: {src_cos:.4f}")
        self.card_linf.setText(f"L_inf: {linf:.4f}")

    def log_status(self, text: str):
        self.status_label.setText(f"Status: {text}")
        self.log_area.append(f"[STATUS] {text}")

    def log_metric(self, name: str, val: float):
        self.log_area.append(f"[METRIC] {name} = {val:.4f}")

    def handle_experiment_finished(self, results: dict):
        self.btn_launch.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.status_label.setText("Status: Evaluation Completed.")

        # Update visual inspector
        if "clean_np" in results:
            self.disp_clean.set_image_array(results["clean_np"])
            self.disp_adv.set_image_array(results["adv_np"])
            self.disp_sanitized.set_image_array(results["sanitized_np"])
            self.disp_noise.set_image_array(results["noise_np"])

        # Populate summary table
        metrics_to_display = [
            ("Execution Device / Surrogate Mode", str(results.get("surrogate_mode"))),
            ("Elapsed Optimization Time", f"{results.get('elapsed_seconds', 0):.2f} s"),
            ("Perturbation Budget (L_inf bound)", f"{results.get('epsilon', 0):.4f}"),
            ("Measured Maximum L_inf Distortion", f"{results.get('l_infinity', 0):.4f}"),
            ("Measured L2 Distortion Norm", f"{results.get('l_2', 0):.4f}"),
            ("Perceptual PSNR Fidelity", f"{results.get('psnr_db', 0):.2f} dB"),
            ("Structural Similarity (SSIM)", f"{results.get('ssim', 0):.4f}"),
            ("Clean Baseline Target Cosine", f"{results.get('baseline_tgt_cos', 0):.4f}"),
            ("Adversarial Target Cosine", f"{results.get('adversarial_final_tgt_cos', 0):.4f}"),
            ("Net Target Cosine Shift (Delta cos)", f"{results.get('target_cosine_shift', 0):.4f}"),
            ("Defense Sanitization Strategy", str(results.get("defense_type"))),
            ("Post-Sanitization Target Cosine", f"{results.get('post_defense_tgt_cos', 0):.4f}"),
            ("Defense Neutralization (Target Drop)", f"{results.get('sanitization_target_drop', 0):.4f}"),
            ("Source Concept Preservation", f"{results.get('post_defense_src_cos', 0):.4f}"),
            ("Latent Representation Discrepancy", f"{results.get('anomaly_score', 0):.4f}"),
        ]

        self.table_results.setRowCount(len(metrics_to_display))
        for row_idx, (metric_name, metric_val) in enumerate(metrics_to_display):
            self.table_results.setItem(row_idx, 0, QTableWidgetItem(metric_name))
            self.table_results.setItem(row_idx, 1, QTableWidgetItem(metric_val))

        self.card_psnr.setText(f"PSNR: {results.get('psnr_db', 0):.2f} dB")
        self.log_area.append("\n=======================================================")
        self.log_area.append("Adversarial benchmark report generated. Check 'Benchmark Metrics' tab.")
        self.log_area.append("=======================================================\n")

    def handle_error(self, err: str):
        self.btn_launch.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.status_label.setText("Status: Error encountered.")
        self.log_area.append(f"[FATAL ERROR] {err}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
