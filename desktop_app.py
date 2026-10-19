#!/usr/bin/env python3
"""Qt desktop chat for the local Ollama model."""

import json
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

import requests
from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from app_config import AppSettings, LANGUAGE_NAMES, tr
from chat_history import ChatHistoryStore
from hardware_monitor import get_hardware_stats
from local_ollama_agent import DEFAULT_OLLAMA_URL
from ollama_runtime import OllamaRuntime


STYLESHEET = """
QWidget {
    background: #111318;
    color: #e7e9ee;
    font-family: "Segoe UI";
    font-size: 10pt;
}
QMainWindow { background: #111318; }
QFrame#Sidebar {
    background: #17191f;
    border-right: 1px solid #292d37;
}
QFrame#Header {
    background: #111318;
    border-bottom: 1px solid #292d37;
}
QLabel#ChatTitle { font-size: 12pt; font-weight: 700; }
QFrame#Composer {
    background: #1c1f27;
    border: 1px solid #3b4050;
    border-radius: 16px;
}
QFrame#MessageCard {
    background: #1c1f27;
    border: 1px solid #292d37;
    border-radius: 16px;
}
QFrame#UserMessageCard {
    background: #302d46;
    border: 1px solid #46415f;
    border-radius: 16px;
}
QFrame#HardwareCard {
    background: #1c1f27;
    border: 1px solid #292d37;
    border-radius: 12px;
}
QLabel#BrandMark {
    color: #aa9cff;
    font-size: 20pt;
    font-weight: 700;
}
QLabel#BrandName { font-size: 12pt; font-weight: 700; }
QLabel#SectionLabel {
    color: #9299a8;
    font-size: 8pt;
    font-weight: 700;
    letter-spacing: 1px;
}
QLabel#Muted { color: #9299a8; }
QLabel#ModelStatus { color: #66d19e; font-size: 9pt; }
QLabel#HardwareStatus { color: #66d19e; font-size: 8pt; }
QLabel#WelcomeTitle { font-size: 23pt; font-weight: 700; }
QLabel#MessageAuthor { color: #aeb4c1; font-size: 9pt; font-weight: 600; }
QLabel#WelcomeSubtitle { color: #9299a8; font-size: 10pt; }
QListWidget {
    background: transparent;
    border: none;
    outline: none;
    padding: 2px;
}
QListWidget::item {
    min-height: 30px;
    padding: 7px 9px;
    border-radius: 7px;
}
QListWidget::item:hover { background: #222630; }
QListWidget::item:selected { background: #302d46; color: #ffffff; }
QLineEdit, QPlainTextEdit {
    background: #20232b;
    border: 1px solid #343946;
    border-radius: 9px;
    padding: 8px 10px;
    selection-background-color: #6457a7;
}
QLineEdit:focus, QPlainTextEdit:focus { border: 1px solid #7568b9; }
QPushButton {
    background: #252934;
    color: #e7e9ee;
    border: 1px solid #343946;
    border-radius: 8px;
    padding: 8px 12px;
}
QPushButton:hover { background: #303541; }
QPushButton:pressed { background: #222630; }
QPushButton:disabled { color: #686f7c; background: #1c1f27; }
QPushButton#PrimaryButton {
    background: #9b8cff;
    color: #171522;
    border: none;
    font-weight: 700;
    padding: 10px 18px;
}
QPushButton#PrimaryButton:hover { background: #ad9fff; }
QPushButton#CopyButton {
    color: #aeb4c1;
    border: none;
    background: transparent;
    padding: 4px 8px;
    font-size: 9pt;
}
QPushButton#CopyButton:hover { color: #ffffff; background: #303541; }
QProgressBar {
    background: #2a2e38;
    border: none;
    border-radius: 3px;
    max-height: 6px;
    text-visible: false;
}
QProgressBar#CpuBar::chunk { background: #aa9cff; border-radius: 3px; }
QProgressBar#RamBar::chunk { background: #66d19e; border-radius: 3px; }
QProgressBar#GpuBar::chunk { background: #f5bd66; border-radius: 3px; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 3px;
}
QScrollBar::handle:vertical { background: #393e4a; border-radius: 4px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: #525867; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QStatusBar { color: #9299a8; background: #111318; }
"""

class ConnectionWorker(QThread):
    succeeded = Signal()
    failed = Signal(str)
    models_found = Signal(list)

    def __init__(
        self,
        runtime: OllamaRuntime,
        ollama_url: str,
        model: str,
        shutdown_event: threading.Event,
        parent=None,
        language: str = "tr",
    ):
        super().__init__(parent)
        self.runtime = runtime
        self.ollama_url = ollama_url
        self.model = model
        self.shutdown_event = shutdown_event
        self.language = language

    def run(self):
        try:
            self.runtime.ensure_running(self.shutdown_event)
            if self.shutdown_event.is_set():
                return
            response = requests.get(
                f"{self.ollama_url}/api/tags",
                timeout=(3, 5),
            )
            response.raise_for_status()
            models = response.json().get("models", [])
            names = {item.get("name", "") for item in models}
            self.models_found.emit(sorted(name for name in names if name))
            if self.model not in names:
                self.failed.emit(
                    f"{tr(self.language, 'model_missing_prefix')}: {self.model}. "
                    f"{tr(self.language, 'pull_model_hint').format(model=self.model)}"
                )
                return
            self.succeeded.emit()
        except requests.RequestException as exc:
            self.failed.emit(self.format_connection_error(self.ollama_url, exc, self.language))
        except (RuntimeError, ValueError, AttributeError) as exc:
            self.failed.emit(str(exc))

    @staticmethod
    def format_connection_error(
        ollama_url: str,
        error: requests.RequestException,
        language: str = "tr",
    ) -> str:
        if isinstance(error, requests.ConnectionError):
            return (
                f"{tr(language, 'ollama_connection_error')} ({ollama_url}). "
                f"{tr(language, 'check_ollama')}"
            )
        if isinstance(error, requests.Timeout):
            return f"{tr(language, 'ollama_timeout')} ({ollama_url}). {tr(language, 'ollama_unavailable')}"
        if isinstance(error, requests.HTTPError) and error.response is not None:
            return tr(language, "ollama_http_error").format(
                status=error.response.status_code
            )
        return f"Ollama bağlantısı başarısız: {error}"


class ChatWorker(QThread):
    token = Signal(str)
    completed = Signal(str)
    failed = Signal(str)
    cancelled = Signal(str)

    def __init__(
        self,
        ollama_url: str,
        model: str,
        messages: List[Dict[str, str]],
        cancel_event: threading.Event,
        parent=None,
        language: str = "tr",
    ):
        super().__init__(parent)
        self.ollama_url = ollama_url
        self.model = model
        self.messages = messages
        self.cancel_event = cancel_event
        self.language = language
        self.response: Optional[requests.Response] = None

    def stop(self):
        self.cancel_event.set()
        if self.response is not None:
            self.response.close()

    def run(self):
        chunks: List[str] = []
        try:
            with requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": self.messages,
                    "stream": True,
                    "options": {"temperature": 0.5},
                },
                stream=True,
                timeout=(10, 180),
            ) as response:
                self.response = response
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if self.cancel_event.is_set():
                        break
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")
                    data = json.loads(line)
                    if data.get("error"):
                        raise RuntimeError(str(data["error"]))
                    token = data.get("message", {}).get("content", "")
                    if token:
                        chunks.append(token)
                        self.token.emit(token)
                    if data.get("done"):
                        break
            content = "".join(chunks)
            if self.cancel_event.is_set():
                self.cancelled.emit(content)
            else:
                self.completed.emit(content)
        except requests.HTTPError as exc:
            if self.cancel_event.is_set():
                self.cancelled.emit("".join(chunks))
                return
            if exc.response is not None and exc.response.status_code == 404:
                detail = (
                    f"{tr(self.language, 'model_missing_prefix')}: {self.model}. "
                    f"{tr(self.language, 'pull_model_hint').format(model=self.model)}"
                )
            else:
                detail = ConnectionWorker.format_connection_error(
                    self.ollama_url, exc, self.language
                )
            self.failed.emit(detail)
        except requests.RequestException as exc:
            if self.cancel_event.is_set():
                self.cancelled.emit("".join(chunks))
            else:
                self.failed.emit(
                    ConnectionWorker.format_connection_error(
                        self.ollama_url, exc, self.language
                    )
                )
        except (json.JSONDecodeError, RuntimeError, KeyError, TypeError) as exc:
            self.failed.emit(
                tr(self.language, "ollama_response_error").format(error=exc)
            )


class HardwareWorker(QThread):
    succeeded = Signal(dict)
    failed = Signal(str)

    def __init__(self, ollama_url: str, parent=None):
        super().__init__(parent)
        self.ollama_url = ollama_url

    def run(self):
        try:
            self.succeeded.emit(get_hardware_stats(self.ollama_url))
        except Exception as exc:
            self.failed.emit(str(exc))


class ShutdownWorker(QThread):
    def __init__(self, runtime: OllamaRuntime, model: str, parent=None):
        super().__init__(parent)
        self.runtime = runtime
        self.model = model

    def run(self):
        self.runtime.unload_model_and_stop(self.model)


class MessageInput(QPlainTextEdit):
    send_requested = Signal()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.send_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class MessageCard(QFrame):
    copy_requested = Signal(str)

    def __init__(self, role: str, content: str = "", language: str = "tr", parent=None):
        super().__init__(parent)
        self.role = role
        self.language = language
        self.setObjectName("UserMessageCard" if role == "user" else "MessageCard")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 12, 10)
        layout.setSpacing(5)

        header = QHBoxLayout()
        self.author_label = QLabel(
            tr(language, "user_label" if role == "user" else "assistant_label")
        )
        self.author_label.setObjectName("MessageAuthor")
        header.addWidget(self.author_label)
        header.addStretch(1)
        self.copy_button = QPushButton(tr(language, "copy_message"))
        self.copy_button.setObjectName("CopyButton")
        self.copy_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.copy_button.setToolTip(tr(language, "copy_tooltip"))
        self.copy_button.clicked.connect(lambda: self.copy_requested.emit(self.text))
        header.addWidget(self.copy_button)
        layout.addLayout(header)

        self.content_view = QTextBrowser()
        self.content_view.setFrameShape(QFrame.Shape.NoFrame)
        self.content_view.setOpenExternalLinks(True)
        self.content_view.setReadOnly(True)
        self.content_view.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        self.content_view.setMinimumHeight(24)
        self.content_view.setStyleSheet(
            "QTextBrowser { background: transparent; color: #e7e9ee; "
            "border: none; padding: 0; font-size: 10pt; }"
        )
        layout.addWidget(self.content_view)
        self.text = ""
        self.set_text(content)

    def set_language(self, language: str):
        self.language = language
        self.author_label.setText(
            tr(language, "user_label" if self.role == "user" else "assistant_label")
        )
        self.copy_button.setText(tr(language, "copy_message"))
        self.copy_button.setToolTip(tr(language, "copy_tooltip"))

    def set_text(self, content: str):
        self.text = content
        if content:
            self.content_view.setMarkdown(content)
        else:
            self.content_view.clear()
        document_height = int(self.content_view.document().size().height())
        self.content_view.setMinimumHeight(min(max(document_height + 4, 24), 4000))
        self.copy_button.setEnabled(bool(content.strip()))

    def set_error(self, content: str):
        self.text = content
        self.content_view.setPlainText(content)
        self.content_view.setStyleSheet(
            "QTextBrowser { background: transparent; color: #ff8585; "
            "border: none; padding: 0; font-size: 10pt; }"
        )


class SettingsDialog(QDialog):
    def __init__(
        self,
        settings: AppSettings,
        store: ChatHistoryStore,
        available_models: Optional[List[str]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.settings = settings
        self.store = store
        self.language = settings.language
        self.history_cleared = False
        self.allowed_folders = list(settings.allowed_folders)
        self.setWindowTitle(tr(self.language, "settings_title"))
        self.setObjectName("AssistantSettingsDialog")
        self.setMinimumWidth(600)
        self.resize(720, 820)
        self.setStyleSheet(
            """
            QDialog#AssistantSettingsDialog { background: #151820; }
            QDialog#AssistantSettingsDialog QLabel#SettingsSubtitle { color: #9299a8; }
            QDialog#AssistantSettingsDialog QGroupBox {
                border: 1px solid #343946;
                border-radius: 10px;
                margin-top: 12px;
                padding: 12px 10px 8px;
                color: #aa9cff;
                font-weight: 700;
            }
            QDialog#AssistantSettingsDialog QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 5px;
            }
            QDialog#AssistantSettingsDialog QLineEdit,
            QDialog#AssistantSettingsDialog QComboBox,
            QDialog#AssistantSettingsDialog QPlainTextEdit {
                background: #1c1f27;
                border: 1px solid #3b4050;
                border-radius: 6px;
                padding: 7px;
                selection-background-color: #7568b9;
            }
            QDialog#AssistantSettingsDialog QPushButton {
                min-height: 30px;
                padding: 3px 12px;
                border-radius: 6px;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(10)
        title = QLabel(tr(self.language, "settings_title"))
        title.setStyleSheet("font-size: 16pt; font-weight: 700;")
        layout.addWidget(title)
        subtitle = QLabel(tr(self.language, "settings_subtitle"))
        subtitle.setObjectName("SettingsSubtitle")
        layout.addWidget(subtitle)

        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QFrame.Shape.NoFrame)
        settings_content = QWidget()
        content_layout = QVBoxLayout(settings_content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)

        model_group = QGroupBox(tr(self.language, "settings_model_section"))
        model_form = QFormLayout(model_group)
        model_form.setSpacing(10)

        self.model_input = QComboBox()
        self.model_input.setEditable(True)
        self.model_input.addItems(available_models or [])
        self.model_input.setCurrentText(settings.model)
        self.model_input.setToolTip(tr(self.language, "model_hint"))
        model_form.addRow(tr(self.language, "model"), self.model_input)

        self.language_input = QComboBox()
        for code, name in LANGUAGE_NAMES.items():
            self.language_input.addItem(name, code)
        language_index = self.language_input.findData(settings.language)
        if language_index >= 0:
            self.language_input.setCurrentIndex(language_index)
        model_form.addRow(tr(self.language, "language"), self.language_input)
        content_layout.addWidget(model_group)

        folders_group = QGroupBox(tr(self.language, "settings_folders_section"))
        folders_layout = QVBoxLayout(folders_group)
        folders_note = QLabel(tr(self.language, "settings_folders_hint"))
        folders_note.setObjectName("SettingsSubtitle")
        folders_note.setWordWrap(True)
        folders_layout.addWidget(folders_note)
        self.allowed_folders_list = QListWidget()
        self.allowed_folders_list.setMaximumHeight(130)
        self._refresh_allowed_folders()
        folders_layout.addWidget(self.allowed_folders_list)
        folder_buttons = QHBoxLayout()
        self.add_folder_button = QPushButton(tr(self.language, "settings_folder_add"))
        self.add_folder_button.clicked.connect(self._add_allowed_folder)
        folder_buttons.addWidget(self.add_folder_button)
        self.remove_folder_button = QPushButton(
            tr(self.language, "settings_folder_remove")
        )
        self.remove_folder_button.clicked.connect(self._remove_allowed_folder)
        folder_buttons.addWidget(self.remove_folder_button)
        folder_buttons.addStretch(1)
        folders_layout.addLayout(folder_buttons)
        content_layout.addWidget(folders_group)

        profile_group = QGroupBox(tr(self.language, "settings_profile_section"))
        profile_form = QFormLayout(profile_group)
        profile_form.setSpacing(10)
        self.name_input = QLineEdit(settings.user_name)
        profile_form.addRow(tr(self.language, "user_name"), self.name_input)
        self.personal_info_input = QPlainTextEdit(settings.personal_info)
        self.personal_info_input.setMaximumHeight(72)
        profile_form.addRow(tr(self.language, "personal_info"), self.personal_info_input)
        content_layout.addWidget(profile_group)

        personality_group = QGroupBox(
            tr(self.language, "settings_personality_section")
        )
        personality_form = QFormLayout(personality_group)
        personality_form.setSpacing(10)
        self.personality_input = QPlainTextEdit(settings.personality)
        self.personality_input.setPlaceholderText(tr(self.language, "personality_hint"))
        self.personality_input.setMaximumHeight(82)
        personality_form.addRow(tr(self.language, "personality"), self.personality_input)
        content_layout.addWidget(personality_group)

        history_row = QHBoxLayout()
        history_note = QLabel(tr(self.language, "clear_history_hint"))
        history_note.setObjectName("SettingsSubtitle")
        history_note.setWordWrap(True)
        history_row.addWidget(history_note, 1)
        self.delete_history_button = QPushButton(tr(self.language, "delete_history"))
        self.delete_history_button.setObjectName("DangerButton")
        self.delete_history_button.clicked.connect(self._delete_history)
        history_row.addWidget(self.delete_history_button)
        content_layout.addLayout(history_row)
        content_layout.addStretch(1)
        settings_scroll.setWidget(settings_content)
        layout.addWidget(settings_scroll, 1)

        self.buttons = QDialogButtonBox()
        self.save_button = self.buttons.addButton(
            tr(self.language, "save"),
            QDialogButtonBox.ButtonRole.AcceptRole,
        )
        self.cancel_button = self.buttons.addButton(
            tr(self.language, "cancel"),
            QDialogButtonBox.ButtonRole.RejectRole,
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def selected_settings(self) -> AppSettings:
        return AppSettings(
            model=self.model_input.currentText().strip(),
            language=str(self.language_input.currentData()),
            personality=self.personality_input.toPlainText().strip(),
            user_name=self.name_input.text().strip(),
            personal_info=self.personal_info_input.toPlainText().strip(),
            allowed_folders=list(self.allowed_folders),
        )

    def _refresh_allowed_folders(self):
        self.allowed_folders_list.clear()
        for folder in self.allowed_folders:
            self.allowed_folders_list.addItem(folder)

    def _add_allowed_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            tr(self.language, "settings_folder_add"),
            str(Path.home()),
            QFileDialog.Option.ShowDirsOnly,
        )
        if not folder:
            return
        resolved = str(Path(folder).resolve())
        if resolved in self.allowed_folders:
            QMessageBox.information(
                self,
                tr(self.language, "settings_title"),
                tr(self.language, "settings_folder_duplicate"),
            )
            return
        self.allowed_folders.append(resolved)
        self._refresh_allowed_folders()

    def _remove_allowed_folder(self):
        row = self.allowed_folders_list.currentRow()
        if 0 <= row < len(self.allowed_folders):
            self.allowed_folders.pop(row)
            self._refresh_allowed_folders()

    def accept(self):
        if not self.model_input.currentText().strip():
            QMessageBox.warning(self, tr(self.language, "settings_title"), tr(self.language, "model_hint"))
            self.model_input.setFocus()
            return
        if any(not Path(folder).is_dir() for folder in self.allowed_folders):
            QMessageBox.warning(
                self,
                tr(self.language, "settings_title"),
                tr(self.language, "settings_folder_invalid"),
            )
            return
        super().accept()

    def _delete_history(self):
        answer = QMessageBox.question(
            self,
            tr(self.language, "clear_all_title"),
            tr(self.language, "confirm_delete_history"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.store.delete_all_conversations()
            self.history_cleared = True


class OllamaDesktopApp(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Yerel Asistan")
        self.resize(1320, 860)
        self.setMinimumSize(940, 650)

        self.ollama_url = os.environ.get("OLLAMA_BASE", DEFAULT_OLLAMA_URL).rstrip("/")
        self.ollama_runtime = OllamaRuntime(self.ollama_url)
        self.store = ChatHistoryStore()
        self.settings = AppSettings.load(self.store)
        self.model = self.settings.model
        self.setWindowTitle(tr(self.settings.language, "app_name"))
        self.cancel_event: Optional[threading.Event] = None
        self.chat_worker: Optional[ChatWorker] = None
        self.connection_worker: Optional[ConnectionWorker] = None
        self.hardware_worker: Optional[HardwareWorker] = None
        self.shutdown_worker: Optional[ShutdownWorker] = None
        self.available_models: List[str] = []
        self._ollama_ready = False
        self.active_message: Optional[MessageCard] = None
        self.active_chunks: List[str] = []
        self.busy = False
        self.shutdown_event = threading.Event()
        self._closing_started = False
        self._close_ready = False
        self.current_conversation_id: Optional[str] = None
        self.message_cards: List[MessageCard] = []
        self._refreshing_history = False

        self.setStyleSheet(STYLESHEET)
        self._build_ui()
        self._refresh_history()
        self.setStatusBar(QStatusBar())
        self._new_conversation()
        self.statusBar().showMessage(
            tr(self.settings.language, "history_saved")
        )

        QTimer.singleShot(0, self._start_connection_check)
        QTimer.singleShot(300, self._start_hardware_check)
        self.hardware_timer = QTimer(self)
        self.hardware_timer.setInterval(3000)
        self.hardware_timer.timeout.connect(self._start_hardware_check)

    @staticmethod
    def _button(text: str, callback, primary: bool = False) -> QPushButton:
        button = QPushButton(text)
        if primary:
            button.setObjectName("PrimaryButton")
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(callback)
        return button

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        outer.addWidget(self.splitter)

        self._build_sidebar()
        self._build_chat_area()
        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(self.chat_panel)
        self.splitter.setSizes([285, 1035])

    def _build_sidebar(self):
        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setMinimumWidth(240)
        self.sidebar.setMaximumWidth(380)
        layout = QVBoxLayout(self.sidebar)
        layout.setContentsMargins(16, 20, 16, 14)
        layout.setSpacing(12)

        brand = QHBoxLayout()
        mark = QLabel("◈")
        mark.setObjectName("BrandMark")
        self.brand_name = QLabel(tr(self.settings.language, "app_name"))
        self.brand_name.setObjectName("BrandName")
        brand.addWidget(mark)
        brand.addWidget(self.brand_name)
        brand.addStretch(1)
        layout.addLayout(brand)

        self.new_chat_button = self._button(
            tr(self.settings.language, "new_chat"), self._new_conversation
        )
        layout.addWidget(self.new_chat_button)

        self.history_heading = QLabel(tr(self.settings.language, "history"))
        self.history_heading.setObjectName("SectionLabel")
        layout.addWidget(self.history_heading)
        self.history_search = QLineEdit()
        self.history_search.setPlaceholderText(tr(self.settings.language, "chat_search"))
        self.history_search.textChanged.connect(self._filter_history)
        layout.addWidget(self.history_search)

        self.history_list = QListWidget()
        self.history_list.currentItemChanged.connect(self._on_history_select)
        layout.addWidget(self.history_list, 1)

        self._build_hardware_card(layout)
        bottom = QHBoxLayout()
        self.delete_button = self._button(
            tr(self.settings.language, "delete_chat"), self._delete_current
        )
        bottom.addWidget(self.delete_button)
        bottom.addStretch(1)
        layout.addLayout(bottom)

    def _build_hardware_card(self, parent_layout: QVBoxLayout):
        card = QFrame()
        card.setObjectName("HardwareCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(5)

        header = QHBoxLayout()
        self.hardware_heading = QLabel(tr(self.settings.language, "system_status"))
        self.hardware_heading.setObjectName("SectionLabel")
        header.addWidget(self.hardware_heading)
        header.addStretch(1)
        self.hardware_status = QLabel(tr(self.settings.language, "measuring"))
        self.hardware_status.setObjectName("HardwareStatus")
        header.addWidget(self.hardware_status)
        layout.addLayout(header)

        self.cpu_text, self.cpu_bar = self._add_metric(
            layout, tr(self.settings.language, "cpu"), "CpuBar"
        )
        self.ram_text, self.ram_bar = self._add_metric(
            layout, tr(self.settings.language, "memory"), "RamBar"
        )
        self.gpu_text, self.gpu_bar = self._add_metric(
            layout, tr(self.settings.language, "gpu"), "GpuBar"
        )
        self.vram_text = QLabel(tr(self.settings.language, "vram_measuring"))
        self.vram_text.setObjectName("Muted")
        self.vram_text.setWordWrap(True)
        layout.addWidget(self.vram_text)
        parent_layout.addWidget(card)

    @staticmethod
    def _add_metric(parent_layout: QVBoxLayout, name: str, object_name: str):
        label = QLabel(f"{name}  ·  --")
        bar = QProgressBar()
        bar.setObjectName(object_name)
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(False)
        parent_layout.addWidget(label)
        parent_layout.addWidget(bar)
        return label, bar

    def _build_chat_area(self):
        self.chat_panel = QWidget()
        layout = QVBoxLayout(self.chat_panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("Header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 15, 22, 15)
        title_column = QVBoxLayout()
        title_column.setSpacing(2)
        self.chat_heading = QLabel(tr(self.settings.language, "chat_heading"))
        self.chat_heading.setStyleSheet("font-size: 13pt; font-weight: 700;")
        title_column.addWidget(self.chat_heading)
        self.model_status = QLabel(
            f"●  {self.model} · {tr(self.settings.language, 'connection_starting')}"
        )
        self.model_status.setObjectName("ModelStatus")
        title_column.addWidget(self.model_status)
        header_layout.addLayout(title_column)
        header_layout.addStretch(1)

        self.copy_latest_button = self._button(
            tr(self.settings.language, "copy_latest"),
            self._copy_latest_reply,
        )
        self.copy_latest_button.setEnabled(False)
        header_layout.addWidget(self.copy_latest_button)
        self.copy_chat_button = self._button(
            tr(self.settings.language, "copy_chat"), self._copy_conversation
        )
        header_layout.addWidget(self.copy_chat_button)
        self.retry_button = self._button(
            tr(self.settings.language, "start_ollama"), self._start_connection_check
        )
        header_layout.addWidget(self.retry_button)
        self.stop_button = self._button(tr(self.settings.language, "stop"), self._stop_generation)
        self.stop_button.setEnabled(False)
        header_layout.addWidget(self.stop_button)
        self.settings_button = self._button(
            tr(self.settings.language, "settings"), self._open_settings
        )
        header_layout.addWidget(self.settings_button)
        layout.addWidget(header)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.message_container = QWidget()
        self.messages_layout = QVBoxLayout(self.message_container)
        self.messages_layout.setContentsMargins(12, 22, 12, 20)
        self.messages_layout.setSpacing(12)
        self.messages_layout.addStretch(1)
        self.scroll.setWidget(self.message_container)
        layout.addWidget(self.scroll, 1)

        composer_wrap = QWidget()
        composer_outer = QVBoxLayout(composer_wrap)
        composer_outer.setContentsMargins(70, 12, 70, 4)
        composer_outer.setSpacing(5)
        composer = QFrame()
        composer.setObjectName("Composer")
        composer_layout = QHBoxLayout(composer)
        composer_layout.setContentsMargins(14, 8, 8, 8)
        composer_layout.setSpacing(10)
        self.input_box = MessageInput()
        self.input_box.setPlaceholderText(tr(self.settings.language, "message_placeholder"))
        self.input_box.setMinimumHeight(60)
        self.input_box.setMaximumHeight(150)
        self.input_box.setTabChangesFocus(False)
        self.input_box.textChanged.connect(self._resize_input)
        composer_layout.addWidget(self.input_box, 1)
        self.send_button = self._button("Gönder  ➤", self._send_message, primary=True)
        self.send_button.setMinimumHeight(44)
        composer_layout.addWidget(self.send_button, 0, Qt.AlignmentFlag.AlignBottom)
        self.input_box.send_requested.connect(self._send_message)
        composer_outer.addWidget(composer)
        self.send_hint = QLabel(tr(self.settings.language, "send_hint"))
        self.send_hint.setObjectName("Muted")
        self.send_hint.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        composer_outer.addWidget(self.send_hint)
        layout.addWidget(composer_wrap)
    def _resize_input(self):
        document_height = self.input_box.document().size().height()
        self.input_box.setFixedHeight(max(60, min(150, int(document_height + 26))))

    def _refresh_history(self, select_id: Optional[str] = None):
        self._refreshing_history = True
        self.history_list.clear()
        self._history_ids = []
        for conversation in self.store.list_conversations():
            item = QListWidgetItem(conversation["title"])
            item.setData(Qt.ItemDataRole.UserRole, conversation["id"])
            self.history_list.addItem(item)
            self._history_ids.append(conversation["id"])
            if conversation["id"] == select_id:
                self.history_list.setCurrentItem(item)
        if select_id is None:
            self.history_list.setCurrentRow(-1)
        self._refreshing_history = False
        self._filter_history(self.history_search.text())

    def _filter_history(self, query: str):
        query = query.casefold().strip()
        for index in range(self.history_list.count()):
            item = self.history_list.item(index)
            item.setHidden(bool(query) and query not in item.text().casefold())

    def _on_history_select(self, current, _previous=None):
        if self._refreshing_history or self.busy or current is None:
            return
        conversation_id = current.data(Qt.ItemDataRole.UserRole)
        if conversation_id and conversation_id != self.current_conversation_id:
            self._load_conversation(conversation_id)

    def _new_conversation(self):
        if self.busy:
            return
        self.current_conversation_id = None
        self._clear_messages()
        self._show_welcome()
        self._refresh_history()
        self.copy_latest_button.setEnabled(False)
        self.copy_chat_button.setEnabled(False)
        self.delete_button.setEnabled(False)
        self.statusBar().showMessage(tr(self.settings.language, "new_chat_status"))

    def _load_conversation(self, conversation_id: str):
        if self.busy:
            return
        self.current_conversation_id = conversation_id
        self._clear_messages()
        messages = self.store.get_messages(conversation_id)
        if messages:
            for message in messages:
                self._add_message(message["role"], message["content"])
        else:
            self._show_welcome()
        self._refresh_history(select_id=conversation_id)
        self.copy_latest_button.setEnabled(
            any(message["role"] == "assistant" for message in messages)
        )
        self.copy_chat_button.setEnabled(bool(messages))
        self.delete_button.setEnabled(True)

    def _clear_messages(self):
        while self.messages_layout.count():
            item = self.messages_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.message_cards.clear()
        self.messages_layout.addStretch(1)

    def _add_message(self, role: str, content: str = "") -> MessageCard:
        stretch = self.messages_layout.takeAt(self.messages_layout.count() - 1)
        card = MessageCard(role, content, self.settings.language, self.message_container)
        card.copy_requested.connect(self._copy_text)
        self.messages_layout.addWidget(card)
        self.message_cards.append(card)
        if stretch.spacerItem():
            self.messages_layout.addItem(stretch)
        self._scroll_to_bottom()
        return card

    def _show_welcome(self):
        welcome = QFrame()
        welcome.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(welcome)
        layout.setContentsMargins(0, 65, 0, 20)
        layout.setSpacing(10)
        title = QLabel(f"{tr(self.settings.language, 'greeting')} 👋")
        title.setObjectName("WelcomeTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        subtitle = QLabel(tr(self.settings.language, "welcome_hint"))
        subtitle.setObjectName("WelcomeSubtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)
        layout.addSpacing(10)
        for prompt_key in ("suggestion_python", "suggestion_idea", "suggestion_ai"):
            prompt = tr(self.settings.language, prompt_key)
            button = QPushButton(prompt)
            button.setMaximumWidth(500)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, text=prompt: self._send_prompt(text))
            layout.addWidget(button, 0, Qt.AlignmentFlag.AlignHCenter)
        stretch = self.messages_layout.takeAt(self.messages_layout.count() - 1)
        self.messages_layout.addWidget(welcome)
        self.messages_layout.addItem(stretch)

    def _send_prompt(self, prompt: str):
        self.input_box.setPlainText(prompt)
        self._send_message()

    def _scroll_to_bottom(self):
        QTimer.singleShot(
            0,
            lambda: self.scroll.verticalScrollBar().setValue(
                self.scroll.verticalScrollBar().maximum()
            ),
        )

    def _start_connection_check(self):
        if self.shutdown_event.is_set():
            return
        if self.connection_worker and self.connection_worker.isRunning():
            return
        self.model_status.setText(
            f"●  {self.model} · {tr(self.settings.language, 'connection_starting')}"
        )
        self.model_status.setStyleSheet("color: #f5bd66;")
        self.retry_button.setEnabled(False)
        self.connection_worker = ConnectionWorker(
            self.ollama_runtime,
            self.ollama_url,
            self.model,
            self.shutdown_event,
            self,
            language=self.settings.language,
        )
        self.connection_worker.succeeded.connect(self._connection_ready)
        self.connection_worker.models_found.connect(self._models_found)
        self.connection_worker.failed.connect(self._connection_failed)
        self.connection_worker.finished.connect(
            lambda: self.retry_button.setEnabled(not self.shutdown_event.is_set())
        )
        self.connection_worker.start()

    def _connection_ready(self):
        self._ollama_ready = True
        self.model_status.setText(f"●  {self.model} · {tr(self.settings.language, 'ready')}")
        self.model_status.setStyleSheet("color: #66d19e;")
        self.statusBar().showMessage(tr(self.settings.language, "connection_ready"))

    def _models_found(self, models: List[str]):
        self.available_models = models

    def _connection_failed(self, message: str):
        self._ollama_ready = False
        self.model_status.setText(
            tr(self.settings.language, "model_connection_issue").format(model=self.model)
        )
        self.model_status.setStyleSheet("color: #ff8585;")
        if not self.busy:
            label = tr(self.settings.language, "connection_failure_status")
            self._add_message("assistant", f"**{label}**\n\n{message}").set_error(
                f"{label}\n\n{message}"
            )
        self.statusBar().showMessage(
            tr(self.settings.language, "connection_failure_status")
        )

    def _start_hardware_check(self):
        if self.shutdown_event.is_set():
            return
        if self.hardware_worker and self.hardware_worker.isRunning():
            return
        self.hardware_worker = HardwareWorker(self.ollama_url, self)
        self.hardware_worker.succeeded.connect(self._update_hardware_stats)
        self.hardware_worker.failed.connect(
            lambda _message: self.hardware_status.setText(
                tr(self.settings.language, "metrics_unavailable")
            )
        )
        self.hardware_worker.start()

    def _update_hardware_stats(self, stats):
        cpu = max(0, min(100, float(stats["cpu_percent"])))
        ram = max(0, min(100, float(stats["ram_percent"])))
        self.cpu_text.setText(f"{tr(self.settings.language, 'cpu')}  ·  {cpu:.0f}%")
        self.cpu_bar.setValue(round(cpu))
        self.ram_text.setText(
            f"{tr(self.settings.language, 'memory')}  ·  "
            f"{stats['ram_used_gb']:.1f}/{stats['ram_total_gb']:.1f} GB · {ram:.0f}%"
        )
        self.ram_bar.setValue(round(ram))
        gpu = stats.get("gpu")
        if gpu:
            utilization = max(0, min(100, float(gpu["utilization"])))
            self.gpu_text.setText(
                f"{tr(self.settings.language, 'nvidia_gpu')}  ·  {utilization:.0f}% · "
                f"{gpu['memory_used_gb']:.1f}/{gpu['memory_total_gb']:.1f} GB"
            )
            self.gpu_bar.setValue(round(utilization))
        else:
            self.gpu_text.setText(
                f"{tr(self.settings.language, 'gpu')}  ·  "
                f"{tr(self.settings.language, 'gpu_unavailable')}"
            )
            self.gpu_bar.setValue(0)
        vram = stats.get("ollama_vram_gb")
        self.vram_text.setText(
            f"{tr(self.settings.language, 'ollama_vram')}  ·  {vram:.2f} GB" if vram is not None
            else tr(self.settings.language, "vram_unavailable")
        )
        self.hardware_status.setText(tr(self.settings.language, "live"))
        if not self.hardware_timer.isActive():
            self.hardware_timer.start()

    def _send_message(self):
        if self.busy:
            return
        content = self.input_box.toPlainText().strip()
        if not content:
            return
        if not self.current_conversation_id:
            self.current_conversation_id = self.store.create_conversation()

        self.input_box.clear()
        self.store.add_message(self.current_conversation_id, "user", content)
        self._load_conversation(self.current_conversation_id)
        self.active_message = self._add_message(
            "assistant", tr(self.settings.language, "thinking")
        )
        stored_messages = self.store.get_messages(self.current_conversation_id)
        prompt_messages = [{"role": "system", "content": self.settings.system_prompt()}]
        prompt_messages.extend(
            {"role": message["role"], "content": message["content"]}
            for message in stored_messages[-40:]
        )

        self.cancel_event = threading.Event()
        self.active_chunks = []
        self.busy = True
        self.send_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.new_chat_button.setEnabled(False)
        self.copy_latest_button.setEnabled(False)
        self.model_status.setText(
            f"●  {tr(self.settings.language, 'model_replying').format(model=self.model)}"
        )
        self.model_status.setStyleSheet("color: #aa9cff;")
        self.chat_worker = ChatWorker(
            self.ollama_url,
            self.model,
            prompt_messages,
            self.cancel_event,
            self,
            language=self.settings.language,
        )
        self.chat_worker.token.connect(self._receive_token)
        self.chat_worker.completed.connect(self._finish_reply)
        self.chat_worker.cancelled.connect(self._finish_cancelled)
        self.chat_worker.failed.connect(self._reply_failed)
        self.chat_worker.finished.connect(self._set_idle)
        self.chat_worker.start()

    def _receive_token(self, token: str):
        self.active_chunks.append(token)
        if self.active_message:
            self.active_message.set_text("".join(self.active_chunks))
            self._scroll_to_bottom()

    def _finish_reply(self, content: str):
        content = content.strip()
        if content and self.current_conversation_id:
            self.store.add_message(self.current_conversation_id, "assistant", content)
            self._refresh_history(select_id=self.current_conversation_id)
            if self.active_message:
                self.active_message.set_text(content)
            self.copy_latest_button.setEnabled(True)
        else:
            self._reply_failed(tr(self.settings.language, "model_empty"))
            return
        self.model_status.setText(
            f"●  {self.model} · {tr(self.settings.language, 'ready')}"
        )
        self.model_status.setStyleSheet("color: #66d19e;")

    def _finish_cancelled(self, content: str):
        content = content.strip()
        if content and self.current_conversation_id:
            self.store.add_message(self.current_conversation_id, "assistant", content)
            self._refresh_history(select_id=self.current_conversation_id)
            self.copy_latest_button.setEnabled(True)
        if self.active_message:
            self.active_message.set_text(
                content or tr(self.settings.language, "reply_stopped")
            )
        self.model_status.setText(
            f"●  {tr(self.settings.language, 'model_stopped').format(model=self.model)}"
        )
        self.model_status.setStyleSheet("color: #9299a8;")

    def _reply_failed(self, message: str):
        if self.active_message:
            self.active_message.set_error(message)
        self.model_status.setText(
            f"●  {tr(self.settings.language, 'model_connection_issue').format(model=self.model)}"
        )
        self.model_status.setStyleSheet("color: #ff8585;")
        self.statusBar().showMessage(message, 10000)

    def _set_idle(self):
        self.busy = False
        self.active_message = None
        self.cancel_event = None
        self.send_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.new_chat_button.setEnabled(True)

    def _stop_generation(self):
        if self.chat_worker and self.chat_worker.isRunning():
            self.chat_worker.stop()
            self.stop_button.setEnabled(False)

    def _copy_text(self, text: str):
        QApplication.clipboard().setText(text)
        self.statusBar().showMessage(tr(self.settings.language, "copy_done"), 2500)

    def _copy_latest_reply(self):
        if not self.current_conversation_id:
            return
        for message in reversed(self.store.get_messages(self.current_conversation_id)):
            if message["role"] == "assistant":
                self._copy_text(message["content"])
                return
        self.statusBar().showMessage(tr(self.settings.language, "no_answer"), 2500)

    def _copy_conversation(self):
        if not self.current_conversation_id:
            return
        messages = self.store.get_messages(self.current_conversation_id)
        if not messages:
            self.statusBar().showMessage(tr(self.settings.language, "no_messages"), 2500)
            return
        transcript = "\n\n".join(
            f"{tr(self.settings.language, 'user_label' if message['role'] == 'user' else 'assistant_label')}:\n"
            f"{message['content']}"
            for message in messages
        )
        self._copy_text(transcript)

    def _delete_current(self):
        if self.busy or not self.current_conversation_id:
            return
        answer = QMessageBox.question(
            self,
            tr(self.settings.language, "delete_chat_title"),
            tr(self.settings.language, "delete_chat_confirm"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.store.delete_conversation(self.current_conversation_id)
        remaining = self.store.list_conversations()
        if remaining:
            self._load_conversation(remaining[0]["id"])
        else:
            self._new_conversation()

    def _open_settings(self):
        if self.busy:
            self.statusBar().showMessage(tr(self.settings.language, "settings_wait"), 5000)
            return
        dialog = SettingsDialog(
            self.settings,
            self.store,
            self.available_models,
            self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            previous_model = self.model
            updated_settings = dialog.selected_settings()
            try:
                updated_settings.save(self.store)
            except ValueError as exc:
                QMessageBox.warning(
                    self,
                    tr(updated_settings.language, "settings_title"),
                    str(exc),
                )
                return
            self.settings = updated_settings
            self.model = self.settings.model
            self._retranslate_ui()
            if self.current_conversation_id:
                self._load_conversation(self.current_conversation_id)
            else:
                self._new_conversation()
            self.statusBar().showMessage(tr(self.settings.language, "settings_saved"), 4000)
            if previous_model != self.model:
                self.model_status.setText(
                    f"●  {self.model} · {tr(self.settings.language, 'connection_starting')}"
                )
                QTimer.singleShot(250, self._start_connection_check)
        if dialog.history_cleared:
            self.current_conversation_id = None
            self._new_conversation()
            self.statusBar().showMessage(
                tr(self.settings.language, "delete_history_result"), 4000
            )

    def _retranslate_ui(self):
        language = self.settings.language
        self.setWindowTitle(tr(language, "app_name"))
        self.brand_name.setText(tr(language, "app_name"))
        self.new_chat_button.setText(tr(language, "new_chat"))
        self.history_heading.setText(tr(language, "history"))
        self.history_search.setPlaceholderText(tr(language, "chat_search"))
        self.delete_button.setText(tr(language, "delete_chat"))
        self.chat_heading.setText(tr(language, "chat_heading"))
        self.copy_latest_button.setText(tr(language, "copy_latest"))
        self.copy_chat_button.setText(tr(language, "copy_chat"))
        self.retry_button.setText(tr(language, "start_ollama"))
        self.stop_button.setText(tr(language, "stop"))
        self.settings_button.setText(tr(language, "settings"))
        self.input_box.setPlaceholderText(tr(language, "message_placeholder"))
        self.send_button.setText(f"{tr(language, 'send')}  ➤")
        self.send_hint.setText(tr(language, "send_hint"))
        self.hardware_heading.setText(tr(language, "system_status"))
        self.hardware_status.setText(tr(language, "measuring"))
        self.vram_text.setText(tr(language, "vram_measuring"))
        for label, key in (
            (self.cpu_text, "cpu"),
            (self.ram_text, "memory"),
            (self.gpu_text, "gpu"),
        ):
            label.setText(f"{tr(language, key)}  ·  --")
        for card in self.message_cards:
            card.set_language(language)
        self.model_status.setText(
            f"●  {self.model} · "
            f"{tr(language, 'ready' if self._ollama_ready else 'connection_starting')}"
        )

    def closeEvent(self, event):
        if self._close_ready:
            event.accept()
            return
        event.ignore()
        if self._closing_started:
            return
        self._closing_started = True
        self.shutdown_event.set()
        self.hardware_timer.stop()
        if self.cancel_event:
            if self.chat_worker and self.chat_worker.isRunning():
                self.chat_worker.stop()
            else:
                self.cancel_event.set()
        QTimer.singleShot(50, self._wait_for_workers_before_close)

    def _wait_for_workers_before_close(self):
        workers = (self.chat_worker, self.connection_worker, self.hardware_worker)
        if any(worker and worker.isRunning() for worker in workers):
            QTimer.singleShot(100, self._wait_for_workers_before_close)
            return
        self.shutdown_worker = ShutdownWorker(self.ollama_runtime, self.model, self)
        self.shutdown_worker.finished.connect(self._finish_shutdown)
        self.statusBar().showMessage("Model VRAM'den boşaltılıyor…")
        self.shutdown_worker.start()

    def _finish_shutdown(self):
        self._close_ready = True
        self.close()


def main():
    application = QApplication([])
    application.setApplicationName("Yerel Asistan")
    application.setStyle("Fusion")
    window = OllamaDesktopApp()
    window.show()
    application.exec()


if __name__ == "__main__":
    main()
