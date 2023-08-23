# coding: utf-8
__author__ = "alphatoaster"


import io
import os
import sys

import yaml
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication, QMainWindow, QDialog, QSlider, QFileDialog

import inference
from gui.ui_mainwindow import Ui_MainWindow
from gui.ui_settingswindow import Ui_Dialog

default_config = {
    "input_audio": list[str],
    "output_folder": f"{os.getcwd()}{os.sep}results{os.sep}",
    "cpu": False,
    "overlap_demucs": 0.6,
    "overlap_MDX": 0.0,
    "overlap_MDXv3": 8,
    "weight_MDXv3": 6,
    "weight_VOCFT": 5,
    "weight_HQ3": 2,
    "single_onnx": False,
    "chunk_size": 500000,
    "large_gpu": False,
    "bigshifts": 11,
    "vocals_only": False,
    "output_format": "PCM_16",
}


class RedirectedStdout(io.StringIO):
    def __init__(self, signal):
        super().__init__()
        self.signal = signal

    def write(self, message):
        self.signal.emit(message)


class WorkerThread(QThread):
    output_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.redirected_stdout = None
        sys.stdout = self.redirected_stdout

    def run(self):
        self.redirected_stdout = RedirectedStdout(window.worker_thread.output_signal)

        inference.options = window.current_config
        print(f"Current configuration: {inference.options}\n========================")
        inference.predict_with_model(inference.options)


def restore_settings():
    apply_config(default_config)


class AdvancedSettingsDialog(QDialog):
    def __init__(self):
        super(AdvancedSettingsDialog, self).__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        # Slider update events
        self.ui.horizontalSlider_BigShifts.valueChanged.connect(update_slider_labels)
        self.ui.horizontalSlider_weight_MDXv3.valueChanged.connect(update_slider_labels)
        self.ui.horizontalSlider_weight_VOCFT.valueChanged.connect(update_slider_labels)
        self.ui.horizontalSlider_weight_HQ3.valueChanged.connect(update_slider_labels)
        self.ui.horizontalSlider_overlap_MDX.valueChanged.connect(update_slider_labels)
        self.ui.horizontalSlider_overlap_MDXv3.valueChanged.connect(
            update_slider_labels
        )
        self.ui.horizontalSlider_overlap_demucs.valueChanged.connect(
            update_slider_labels
        )
        self.ui.button_ExportSettings.clicked.connect(self.export_settings)
        self.ui.button_ImportSettings.clicked.connect(self.import_settings)
        self.ui.button_RestoreDefaults.clicked.connect(restore_settings)
        self.redirected_stdout = RedirectedStdout(window.worker_thread.output_signal)
        sys.stdout = self.redirected_stdout

    def import_settings(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select configuration file", filter="Yaml Source File (.yml)"
        )
        apply_config(read_config(filename))

    def export_settings(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save configuration file", filter="Yaml Source File (.yml)"
        )
        save_config(filename, get_config())


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.input_files: list
        self.current_config: dict = {}

        # Setup stdout redirection and separation thread
        self.worker_thread = WorkerThread()
        self.worker_thread.finished.connect(self.worker_finished)
        self.worker_thread.output_signal.connect(self.append_to_text_edit)
        self.redirected_stdout = RedirectedStdout(self.worker_thread.output_signal)
        sys.stdout = self.redirected_stdout

        # Various UI element events
        self.ui.pushButton_StartSeparation.clicked.connect(self.start_separation_thread)
        self.ui.pushButton_AdvSettings.clicked.connect(self.open_advanced_settings)
        self.ui.horizontalSlider_ChunkSize.valueChanged.connect(update_slider_labels)
        self.ui.button_inputDirectory.clicked.connect(self.apply_input_directory)
        self.ui.button_inputFiles.clicked.connect(self.apply_input_audio)
        self.ui.button_OutputDirectory.clicked.connect(self.apply_output_directory)

    def get_output_directory(self):
        directory, _ = QFileDialog.getExistingDirectory(self, "Select output directory")
        return directory

    def apply_output_directory(self):
        self.current_config["output_folder"] = self.get_output_directory()
        self.ui.lineEdit_OutputDirectory = self.current_config["output_folder"]

    def get_input_audio(self):
        _filter = "WAV (*.wav);;MP3 (*.mp3)"
        filenames, _ = QFileDialog.getOpenFileNames(
            self, "Select input audio files", filter=_filter
        )
        print(f"Selected files: {filenames}")
        return filenames

    def get_input_directory(self):
        directory, _ = QFileDialog.getExistingDirectory(self, "Select input directory")
        filenames: list[str]
        for filename in parse_directory(directory):
            filenames += filename
        print(f"Selected files: {filenames}")
        return filenames

    def apply_input_audio(self):
        self.current_config["input_audio"] = self.get_input_audio()
        self.ui.lineEdit_inputFiles.setText(
            "; ".join(self.current_config["input_audio"])
        )

    def apply_input_directory(self):
        self.current_config["input_audio"] = self.get_input_directory()
        self.ui.lineEdit_inputDirectory.setText(
            "; ".join(self.current_config["input_audio"])
        )

    @staticmethod
    def open_advanced_settings():
        adv_settings.exec()

    def start_separation_thread(self):
        self.ui.pushButton_StartSeparation.setEnabled(False)
        self.current_config = get_config()

        file_selection = 0
        batch_processing = 1
        current_tab = self.ui.tabWidget_Selection.currentIndex()

        print(self.current_config)
        print(self.current_config["input_audio"])
        if self.current_config["input_audio"] == "":
            print("File/directory not selected!")
            if current_tab == file_selection:
                print("File selection tab active. Getting files...")
                self.get_input_audio()
            elif current_tab == batch_processing:
                print("Batch processing tab active. Processing directory...")
                self.get_input_directory()
            else:
                raise AttributeError("tabWidget not found. What?")

        self.worker_thread.start()

    def append_to_text_edit(self, text):
        self.ui.textEdit_StdOut.insertPlainText(text)

    def worker_finished(self):
        self.ui.pushButton_StartSeparation.setEnabled(True)


def parse_directory(path: str):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


def read_config(config_path: str) -> dict:
    """Reads config from a file

    Args:
        config_path (str): Path to an existing config file

    Returns:
        dict: configuration
    """

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config.keys() == default_config.keys():
        return config
    else:
        raise KeyError(
            "Failed to read configuration file from config.yml because it does not have the same structure as "
            "default config!"
        )


def save_config(config_path: str, config: dict):
    """Saves given configuration dictionary into a file

    Args:
        config_path (str): Path to save config file into
        config (dict): config itself
    """
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, default_flow_style=False)


def convert_value_to_slider(slider: QSlider, value: int | float) -> None:
    if slider.property("inputConversionRatio"):
        match value:
            case int():
                slider.setValue(value // slider.property("inputConversionRatio"))
            case float():
                slider.setValue(
                    int(round(value / slider.property("inputConversionRatio")))
                )
    else:
        slider.setValue(value)


def convert_slider_to_value(slider: QSlider) -> int | float:
    if slider.property("inputConversionRatio"):
        return round(slider.value() * slider.property("inputConversionRatio"), 2)
    else:
        return slider.value()


def update_slider_labels() -> None:
    window.ui.label_ChunkSize.setText(
        str(int(convert_slider_to_value(window.ui.horizontalSlider_ChunkSize)))
    )
    adv_settings.ui.label_BigShifts.setText(
        str(convert_slider_to_value(adv_settings.ui.horizontalSlider_BigShifts))
    )
    adv_settings.ui.label_weight_MDXv3.setText(
        str(convert_slider_to_value(adv_settings.ui.horizontalSlider_weight_MDXv3))
    )
    adv_settings.ui.label_weight_VOCFT.setText(
        str(convert_slider_to_value(adv_settings.ui.horizontalSlider_weight_VOCFT))
    )
    adv_settings.ui.label_weight_HQ3.setText(
        str(convert_slider_to_value(adv_settings.ui.horizontalSlider_weight_HQ3))
    )
    adv_settings.ui.label_overlap_MDX.setText(
        str(convert_slider_to_value(adv_settings.ui.horizontalSlider_overlap_MDX))
    )
    adv_settings.ui.label_overlap_MDXv3.setText(
        str(convert_slider_to_value(adv_settings.ui.horizontalSlider_overlap_MDXv3))
    )
    adv_settings.ui.label_overlap_demucs.setText(
        str(convert_slider_to_value(adv_settings.ui.horizontalSlider_overlap_demucs))
    )


def get_config() -> dict:
    """Parses configuration from Qt App to a dictionary"""
    return {
        "input_audio": window.current_config["input_audio"],
        "output_folder": window.ui.lineEdit_OutputDirectory.text(),
        "cpu": window.ui.checkBox_UseCPU.isChecked(),
        "overlap_demucs": convert_slider_to_value(
            adv_settings.ui.horizontalSlider_overlap_demucs
        ),
        "overlap_MDX": convert_slider_to_value(
            adv_settings.ui.horizontalSlider_overlap_MDX
        ),
        "overlap_MDXv3": adv_settings.ui.horizontalSlider_overlap_MDXv3.value(),
        "weight_MDXv3": adv_settings.ui.horizontalSlider_weight_MDXv3.value(),
        "weight_VOCFT": adv_settings.ui.horizontalSlider_weight_VOCFT.value(),
        "weight_HQ3": adv_settings.ui.horizontalSlider_weight_HQ3.value(),
        "single_onnx": window.ui.checkBox_SingleONNX.isChecked(),
        "chunk_size": int(
            convert_slider_to_value(window.ui.horizontalSlider_ChunkSize)
        ),
        "large_gpu": window.ui.checkBox_UseLargeGPU.isChecked(),
        "bigshifts": adv_settings.ui.horizontalSlider_BigShifts.value(),
        "vocals_only": window.ui.checkBox_GenerateVocInstOnly.isChecked(),
        "output_format": adv_settings.ui.comboBox_OutputFormat.currentText(),
    }


def apply_config(config: dict) -> None:
    """Applies configuration to a Qt App

    Args:
        config (dict): Configuration file
    """
    # Main Window
    window.ui.checkBox_UseCPU.setChecked(config["cpu"])
    window.ui.checkBox_UseLargeGPU.setChecked(config["large_gpu"])
    window.ui.checkBox_SingleONNX.setChecked(config["single_onnx"])
    window.ui.checkBox_GenerateVocInstOnly.setChecked(config["vocals_only"])
    convert_value_to_slider(window.ui.horizontalSlider_ChunkSize, config["chunk_size"])
    window.ui.lineEdit_OutputDirectory.setText(config["output_folder"])

    # Advanced Settings Dialog
    convert_value_to_slider(
        adv_settings.ui.horizontalSlider_BigShifts, config["bigshifts"]
    )
    convert_value_to_slider(
        adv_settings.ui.horizontalSlider_weight_MDXv3, config["weight_MDXv3"]
    )
    convert_value_to_slider(
        adv_settings.ui.horizontalSlider_weight_VOCFT, config["weight_VOCFT"]
    )
    convert_value_to_slider(
        adv_settings.ui.horizontalSlider_weight_HQ3, config["weight_HQ3"]
    )
    convert_value_to_slider(
        adv_settings.ui.horizontalSlider_overlap_MDX, config["overlap_MDX"]
    )
    convert_value_to_slider(
        adv_settings.ui.horizontalSlider_overlap_MDXv3, config["overlap_MDXv3"]
    )
    convert_value_to_slider(
        adv_settings.ui.horizontalSlider_overlap_demucs, config["overlap_demucs"]
    )
    adv_settings.ui.comboBox_OutputFormat.setCurrentText(config["output_format"])

    # Update slider labels
    update_slider_labels()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Set theme
    app.setStyle("Fusion")
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)
    app.setStyleSheet(
        "QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }"
    )

    # Initialize windows
    window = MainWindow()
    adv_settings = AdvancedSettingsDialog()
    try:
        apply_config(read_config("config.yml"))
    except KeyError:
        print(
            "WARNING: Using built-in default configuration as config.yml contains non-valid data."
        )
        apply_config(default_config)
    except FileNotFoundError:
        print(
            "WARNING: Using built-in default configuration as config.yml does not exist."
        )
        apply_config(default_config)
    window.show()
    print("Awaiting separation...")

    sys.exit(app.exec())
