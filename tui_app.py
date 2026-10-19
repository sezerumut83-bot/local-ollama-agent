#!/usr/bin/env python3
"""Terminal chat interface for the local Ollama assistant."""

import json
import os
import re
import threading
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import requests
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, Markdown, Select, Static, TextArea

from app_config import AppSettings, LANGUAGE_NAMES, tr
from chat_history import ChatHistoryStore
from local_ollama_agent import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    FILE_AGENT_TOOLS,
    ToolRunner,
    build_system_prompt,
    normalize_model_name,
    parse_ollama_tool_call,
    parse_response,
    requires_file_write,
)
from ollama_runtime import OllamaRuntime


class PromptTextArea(TextArea):
    BINDINGS = [
        Binding("enter", "submit_message", show=False, priority=True),
        Binding("ctrl+enter", "submit_message", show=False, priority=True),
    ]

    def action_submit_message(self):
        self.app.action_send_message()


class ConfirmScreen(ModalScreen[bool]):
    CSS = """
    ConfirmScreen { align: center middle; background: #000000 60%; }
    #confirm-dialog {
        width: 64;
        height: auto;
        padding: 2;
        border: tall #7568b9;
        background: #17191f;
    }
    #confirm-message { height: auto; margin-bottom: 2; }
    #confirm-buttons { height: 3; align: right middle; }
    """

    def __init__(self, language: str):
        super().__init__()
        self.language = language

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Static(tr(self.language, "confirm_delete_history"), id="confirm-message")
            with Horizontal(id="confirm-buttons"):
                yield Button(tr(self.language, "cancel"), id="cancel")
                yield Button(tr(self.language, "delete_history"), id="confirm-delete", variant="error")

    def on_button_pressed(self, event: Button.Pressed):
        self.dismiss(event.button.id == "confirm-delete")


class HistoryScreen(ModalScreen[Optional[str]]):
    CSS = """
    HistoryScreen { align: center middle; background: #000000 70%; }
    #history-dialog {
        width: 72;
        height: 80%;
        max-height: 90%;
        padding: 1 2;
        border: tall #315b46;
        background: #101816;
    }
    #history-title { height: 3; color: #63e6a3; text-style: bold; }
    #history-filter { margin-bottom: 1; }
    #history-results { height: 1fr; }
    #history-results Button {
        width: 100%;
        height: auto;
        min-height: 3;
        margin-bottom: 1;
        content-align: left middle;
        text-align: left;
        background: #17231d;
        border: tall #263a32;
    }
    #history-results Button:hover { border: tall #63e6a3; }
    #history-cancel { width: 18; margin-top: 1; }
    """

    def compose(self) -> ComposeResult:
        language = self.app.settings.language
        with Vertical(id="history-dialog"):
            yield Static(tr(language, "history_picker_title"), id="history-title")
            yield Input(placeholder=tr(language, "search_chats"), id="history-filter")
            yield VerticalScroll(id="history-results")
            yield Button(tr(language, "cancel"), id="history-cancel")

    async def on_mount(self):
        await self._refresh_results()

    async def on_input_changed(self, event: Input.Changed):
        if event.input.id == "history-filter":
            await self._refresh_results()

    async def _refresh_results(self):
        results = self.query_one("#history-results", VerticalScroll)
        await results.remove_children()
        search = self.query_one("#history-filter", Input).value.casefold().strip()
        conversations = self.app.store.list_conversations()
        matches = [
            item
            for item in conversations
            if not search or search in item["title"].casefold()
        ]
        if not matches:
            await results.mount(Static(tr(self.app.settings.language, "history_empty")))
            return
        for item in matches:
            await results.mount(
                Button(
                    item["title"],
                    id=f"history-select-{item['id']}",
                )
            )

    def on_button_pressed(self, event: Button.Pressed):
        event.stop()
        button_id = event.button.id or ""
        if button_id == "history-cancel":
            self.dismiss(None)
        elif button_id.startswith("history-select-"):
            self.dismiss(button_id.removeprefix("history-select-"))


class ApprovalScreen(ModalScreen[bool]):
    CSS = """
    ApprovalScreen { align: center middle; background: #000000 75%; }
    #approval-dialog {
        width: 88;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        border: tall #b68142;
        background: #171a18;
    }
    #approval-title { height: 3; color: #f0bd70; text-style: bold; }
    #approval-message { height: auto; max-height: 65%; overflow-y: auto; }
    #approval-actions { height: 3; align: right middle; margin-top: 1; }
    #approval-deny { margin-right: 1; }
    """

    def __init__(self, language: str, message: str):
        super().__init__()
        self.language = language
        self.message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-dialog"):
            yield Static(tr(self.language, "approval_title"), id="approval-title")
            yield Static(self.message, id="approval-message")
            with Horizontal(id="approval-actions"):
                yield Button(tr(self.language, "deny"), id="approval-deny")
                yield Button(
                    tr(self.language, "approve"),
                    id="approval-accept",
                    variant="warning",
                )

    def on_button_pressed(self, event: Button.Pressed):
        event.stop()
        self.dismiss(event.button.id == "approval-accept")


class SettingsScreen(ModalScreen[Optional[AppSettings]]):
    CSS = """
    SettingsScreen { align: center middle; background: #000000 70%; }
    #settings-dialog {
        width: 82;
        height: auto;
        max-height: 94%;
        padding: 1 3;
        border: tall #315b46;
        background: #101816;
    }
    #settings-title { height: 3; text-style: bold; color: #63e6a3; }
    #settings-subtitle { height: 2; color: #91aa9b; margin-bottom: 1; }
    .settings-group {
        height: auto;
        padding: 0 1;
        margin-bottom: 1;
        border: round #263a32;
        background: #121d18;
    }
    .settings-section { height: 2; color: #63e6a3; text-style: bold; }
    .settings-label { height: 2; color: #b4c8bc; }
    #settings-model, #settings-language, #settings-name, #settings-model-list,
    #settings-folder-path { margin-bottom: 1; }
    #settings-folder-list { height: 8; margin-bottom: 1; }
    #settings-personal, #settings-personality { height: 4; margin-bottom: 1; }
    #settings-actions { height: 3; align: right middle; margin-top: 1; }
    #settings-delete { color: #ff8585; }
    """

    def __init__(
        self,
        settings: AppSettings,
        available_models: Optional[List[str]] = None,
        workspace: Optional[Path] = None,
    ):
        super().__init__()
        self.settings = replace(settings, allowed_folders=list(settings.allowed_folders))
        self.language = settings.language
        self.available_models = available_models or []
        self.workspace = workspace or Path(".").resolve()

    def compose(self) -> ComposeResult:
        language = self.language
        with VerticalScroll(id="settings-dialog"):
            yield Static(tr(language, "settings_title"), id="settings-title")
            yield Static(tr(language, "settings_subtitle"), id="settings-subtitle")
            with Vertical(classes="settings-group"):
                yield Static(
                    tr(language, "settings_model_section"),
                    classes="settings-section",
                )
                yield Static(tr(language, "model"), classes="settings-label")
                yield Input(self.settings.model, id="settings-model")
                if self.available_models:
                    yield Select(
                        [(model, model) for model in self.available_models],
                        value=(
                            self.settings.model
                            if self.settings.model in self.available_models
                            else Select.NULL
                        ),
                        id="settings-model-list",
                        allow_blank=True,
                    )
                yield Static(tr(language, "model_hint"), classes="settings-label")
                yield Static(tr(language, "language"), classes="settings-label")
                yield Select(
                    [(name, code) for code, name in LANGUAGE_NAMES.items()],
                    value=language,
                    id="settings-language",
                    allow_blank=False,
                )
            with Vertical(classes="settings-group"):
                yield Static(
                    tr(language, "settings_folders_section"),
                    classes="settings-section",
                )
                yield Static(
                    tr(language, "settings_folders_hint"),
                    classes="settings-label",
                )
                with Horizontal():
                    yield Input(
                        placeholder=tr(language, "settings_folder_path"),
                        id="settings-folder-path",
                    )
                    yield Button(
                        tr(language, "settings_folder_add"),
                        id="settings-folder-add",
                        variant="primary",
                    )
                yield VerticalScroll(id="settings-folder-list")
                yield Static(
                    tr(language, "settings_folders_empty"),
                    id="settings-folder-empty",
                    classes="settings-label",
                )
            with Vertical(classes="settings-group"):
                yield Static(
                    tr(language, "settings_profile_section"),
                    classes="settings-section",
                )
                yield Static(tr(language, "user_name"), classes="settings-label")
                yield Input(self.settings.user_name, id="settings-name")
                yield Static(tr(language, "personal_info"), classes="settings-label")
                yield TextArea(self.settings.personal_info, id="settings-personal")
            with Vertical(classes="settings-group"):
                yield Static(
                    tr(language, "settings_personality_section"),
                    classes="settings-section",
                )
                yield Static(tr(language, "personality"), classes="settings-label")
                yield TextArea(
                    self.settings.personality,
                    id="settings-personality",
                    soft_wrap=True,
                )
            with Horizontal(id="settings-actions"):
                yield Button(tr(language, "delete_history"), id="settings-delete", variant="error")
                yield Button(tr(language, "cancel"), id="settings-cancel")
                yield Button(tr(language, "save"), id="settings-save", variant="primary")

    async def on_button_pressed(self, event: Button.Pressed):
        button_id = event.button.id or ""
        if button_id == "settings-folder-add":
            self._add_allowed_folder()
        elif button_id.startswith("settings-remove-folder-"):
            index = int(button_id.removeprefix("settings-remove-folder-"))
            if 0 <= index < len(self.settings.allowed_folders):
                self.settings.allowed_folders.pop(index)
                await self._refresh_allowed_folders()
        elif button_id == "settings-cancel":
            self.dismiss(None)
        elif button_id == "settings-save":
            model = self.query_one("#settings-model", Input).value.strip()
            if not model:
                self.notify(tr(self.language, "model_hint"), severity="error")
                return
            language = self.query_one("#settings-language", Select).value
            if not isinstance(language, str) or language not in LANGUAGE_NAMES:
                return
            self.dismiss(
                AppSettings(
                    model=model,
                    language=language,
                    user_name=self.query_one("#settings-name", Input).value.strip(),
                    personal_info=self.query_one("#settings-personal", TextArea).text.strip(),
                    personality=self.query_one("#settings-personality", TextArea).text.strip(),
                    allowed_folders=list(self.settings.allowed_folders),
                )
            )
        elif button_id == "settings-delete":
            self.app.push_screen(
                ConfirmScreen(self.language),
                callback=self._clear_history_confirmed,
            )

    def on_mount(self):
        self.run_worker(self._refresh_allowed_folders(), name="refresh-allowed-folders")

    def _add_allowed_folder(self):
        raw_path = self.query_one("#settings-folder-path", Input).value.strip()
        if not raw_path:
            return
        candidate = Path(raw_path).expanduser()
        if not candidate.is_dir():
            self.notify(
                tr(self.language, "settings_folder_invalid"),
                severity="error",
            )
            return
        resolved = candidate.resolve()
        if resolved == self.workspace:
            self.notify(
                tr(self.language, "settings_folder_workspace_duplicate"),
                severity="warning",
            )
            return
        normalized = str(resolved)
        if normalized in self.settings.allowed_folders:
            self.notify(
                tr(self.language, "settings_folder_duplicate"),
                severity="warning",
            )
            return
        self.settings.allowed_folders.append(normalized)
        self.query_one("#settings-folder-path", Input).value = ""
        self.run_worker(self._refresh_allowed_folders(), name="refresh-allowed-folders")

    async def _refresh_allowed_folders(self):
        if not self.is_mounted:
            return
        container = self.query_one("#settings-folder-list", VerticalScroll)
        await container.remove_children()
        empty = self.query_one("#settings-folder-empty", Static)
        empty.display = not self.settings.allowed_folders
        for index, folder in enumerate(self.settings.allowed_folders):
            await container.mount(
                Button(
                    f"{tr(self.language, 'settings_folder_remove')}: {folder}",
                    id=f"settings-remove-folder-{index}",
                    variant="error",
                )
            )

    def _clear_history_confirmed(self, confirmed: bool):
        if confirmed:
            self.app.store.delete_all_conversations()
            self.app.notify(tr(self.language, "clear_done"))
            self.app.run_worker(self.app.action_new_chat(), name="history-cleared")

    def on_select_changed(self, event: Select.Changed):
        if event.select.id == "settings-model-list" and isinstance(event.value, str):
            self.query_one("#settings-model", Input).value = event.value


class LocalAssistantTUI(App):
    TITLE = "Yerel Asistan"
    SUB_TITLE = "Ollama • Yerel ve özel"
    CSS = """
    Screen {
        background: #0b1110;
        color: #dce9e2;
    }
    #main { width: 1fr; height: 1fr; padding: 0 3; }
    #topline {
        height: 5;
        padding: 0 1;
        border-bottom: solid #263a32;
        background: #0d1411;
    }
    #identity { width: 1fr; height: 4; }
    #brand {
        height: 3;
        color: #63e6a3;
        text-style: bold;
        content-align: left middle;
    }
    #workspace-info { height: 1; color: #7caa96; }
    #chat-title {
        width: 1fr;
        content-align: left middle;
        color: #b6c9bd;
        text-style: bold;
    }
    #connection {
        width: auto;
        content-align: right middle;
        color: #f5bd66;
        padding: 0 1;
    }
    #messages {
        height: 1fr;
        padding: 1 4;
        scrollbar-color: #315340;
    }
    .message {
        height: auto;
        margin: 1 0;
        padding: 1 2;
        border-left: tall #263a32;
        background: #0f1713;
    }
    .user-message {
        border-left: tall #63e6a3;
        background: #101d17;
    }
    #welcome {
        height: 1fr;
        content-align: center middle;
        color: #dce9e2;
        text-align: center;
        padding: 2 4;
    }
    #composer {
        height: 9;
        border: round #385443;
        background: #0f1713;
        padding: 1 2;
        margin: 0 2 1 2;
    }
    #prompt-row { height: 1fr; }
    #prompt-prefix {
        width: 3;
        content-align: center middle;
        color: #63e6a3;
        text-style: bold;
    }
    #prompt {
        height: 1fr;
        border: none;
        background: #0f1713;
    }
    #composer-footer { height: 2; }
    #status {
        width: 1fr;
        content-align: left middle;
        color: #7caa96;
        padding-left: 1;
    }
    #mode-indicator {
        width: auto;
        content-align: center middle;
        color: #63e6a3;
        text-style: bold;
        padding: 0 1;
    }
    Footer { background: #101816; }
    """
    BINDINGS = [
        ("ctrl+n", "new_chat", "Yeni sohbet"),
        ("ctrl+enter", "send_message", "Gönder"),
        ("escape", "stop_generation", "Durdur"),
        ("ctrl+shift+c", "copy_latest", "Yanıtı kopyala"),
        ("ctrl+o", "settings", "Ayarlar"),
        ("ctrl+q", "quit", "Çıkış"),
    ]

    def __init__(self, workspace: Optional[str] = None):
        super().__init__()
        self.ollama_url = os.environ.get("OLLAMA_BASE", DEFAULT_OLLAMA_URL).rstrip("/")
        self.model = normalize_model_name(os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL))
        self.store = ChatHistoryStore()
        self.settings = AppSettings.load(self.store)
        self.model = self.settings.model
        self.workspace = Path(
            workspace or os.environ.get("LOCAL_OLLAMA_WORKSPACE", ".")
        ).resolve()
        self.runtime = OllamaRuntime(self.ollama_url)
        self.shutdown_event = threading.Event()
        self.cancel_event: Optional[threading.Event] = None
        self.response: Optional[requests.Response] = None
        self.conversation_id: Optional[str] = None
        self.last_answer = ""
        self.active_answer = ""
        self._active_message: Optional[Markdown] = None
        self.reply_worker = None
        self.connection_worker = None
        self.available_models: List[str] = []
        self._connection_ready = False
        self.thinking_mode = "fast"
        self._approval_lock = threading.Lock()
        self._pending_approval: Optional[tuple[threading.Event, Dict[str, bool]]] = None
        self._undo_in_progress = False
        self.tool_runner = ToolRunner(
            str(self.workspace),
            approval_handler=self._request_tool_approval,
            allowed_roots=self.settings.allowed_folders,
        )

    def compose(self) -> ComposeResult:
        yield Vertical(
            Horizontal(
                Vertical(
                    Static(f"$ {tr(self.settings.language, 'app_name').upper()}", id="brand"),
                    Static(
                        f"{self.model}  ·  {tr(self.settings.language, 'workspace_label')}: "
                        f"{self.workspace}",
                        id="workspace-info",
                    ),
                    id="identity",
                ),
                Static(
                    tr(self.settings.language, "new_chat").lstrip("+ "),
                    id="chat-title",
                ),
                Static(
                    tr(self.settings.language, "connection_starting"),
                    id="connection",
                ),
                id="topline",
            ),
            VerticalScroll(id="messages"),
            Vertical(
                Horizontal(
                    Static("$", id="prompt-prefix"),
                    PromptTextArea(
                        placeholder=tr(self.settings.language, "message_placeholder"),
                        soft_wrap=True,
                        id="prompt",
                    ),
                    id="prompt-row",
                ),
                Horizontal(
                    Static(tr(self.settings.language, "command_help"), id="status"),
                    Static(
                        tr(self.settings.language, "mode_fast_label"),
                        id="mode-indicator",
                    ),
                    id="composer-footer",
                ),
                id="composer",
            ),
            id="main",
        )
        yield Footer()

    async def on_mount(self):
        await self.action_new_chat()
        self.connection_worker = self._check_connection()

    @work(thread=True, exclusive=True, group="ollama")
    def _check_connection(self):
        try:
            self.runtime.ensure_running(self.shutdown_event)
            if self.shutdown_event.is_set():
                return
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=(3, 5))
            response.raise_for_status()
            names = {item.get("name", "") for item in response.json().get("models", [])}
            if not self.shutdown_event.is_set():
                self.call_from_thread(self._models_found, sorted(name for name in names if name))
            if self.model not in names:
                raise RuntimeError(
                    f"{tr(self.settings.language, 'model_missing_prefix')}: {self.model}. "
                    f"{tr(self.settings.language, 'pull_model_hint').format(model=self.model)}"
                )
            if not self.shutdown_event.is_set():
                self.call_from_thread(
                    self._set_connection,
                    True,
                    tr(self.settings.language, "connection_ready"),
                )
        except (requests.RequestException, RuntimeError, ValueError, AttributeError) as exc:
            if not self.shutdown_event.is_set():
                self.call_from_thread(self._set_connection, False, str(exc))

    def _set_connection(self, ready: bool, message: str):
        self._connection_ready = ready
        label = self.query_one("#connection", Static)
        label.update(message)
        label.styles.color = "#66d19e" if ready else "#ff8585"

    def _models_found(self, models: List[str]):
        self.available_models = models

    async def _load_conversation(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.last_answer = ""
        self._active_message = None
        title = next(
            (
                item["title"]
                for item in self.store.list_conversations()
                if item["id"] == conversation_id
            ),
            "Sohbet",
        )
        self.query_one("#chat-title", Static).update(title)
        messages = self.query_one("#messages", VerticalScroll)
        await messages.remove_children()
        stored = self.store.get_messages(conversation_id)
        if not stored:
            await messages.mount(
                Static(
                    f"✦\n\n{tr(self.settings.language, 'welcome')}\n\n"
                    f"{tr(self.settings.language, 'welcome_hint')}",
                    id="welcome",
                )
            )
            return
        for item in stored:
            role = tr(
                self.settings.language,
                "user_label" if item["role"] == "user" else "assistant_label",
            )
            classes = "message user-message" if item["role"] == "user" else "message"
            await messages.mount(
                Markdown(f"**{role}**\n\n{item['content']}", classes=classes)
            )
            if item["role"] == "assistant":
                self.last_answer = item["content"]
        messages.scroll_end(animate=False)

    async def action_new_chat(self):
        if self.cancel_event:
            self.action_stop_generation()
            self.query_one("#status", Static).update(
                tr(self.settings.language, "settings_wait")
            )
            return
        self.conversation_id = None
        self.last_answer = ""
        self._active_message = None
        self.query_one("#chat-title", Static).update(
            tr(self.settings.language, "new_chat").lstrip("+ ")
        )
        messages = self.query_one("#messages", VerticalScroll)
        await messages.remove_children()
        await messages.mount(
            Static(
                f"✦\n\n{tr(self.settings.language, 'welcome')}\n\n"
                f"{tr(self.settings.language, 'welcome_hint')}",
                id="welcome",
            )
        )
        self.query_one("#prompt", TextArea).focus()

    def action_send_message(self):
        if self.cancel_event is not None:
            return
        if self._undo_in_progress:
            self.query_one("#status", Static).update(
                tr(self.settings.language, "undo_wait")
            )
            return
        prompt = self.query_one("#prompt", TextArea)
        content = prompt.text.strip()
        if not content:
            return
        prompt.clear()
        if self._handle_command(content):
            return
        if not self._connection_ready:
            self.query_one("#status", Static).update(
                tr(self.settings.language, "waiting_connection")
            )
            return
        if not self.conversation_id:
            self.conversation_id = self.store.create_conversation()
        self.store.add_message(self.conversation_id, "user", content)
        self.query_one("#chat-title", Static).update(content[:80])
        self.run_worker(self._render_then_reply(content), name="send-message")

    def _handle_command(self, content: str) -> bool:
        commands = content.casefold()
        if commands == "--history":
            self.push_screen(
                HistoryScreen(),
                callback=self._history_selected,
            )
        elif commands == "--settings":
            self.action_settings()
        elif commands == "-clear":
            self.run_worker(self.action_new_chat(), name="clear-terminal-chat")
            self.query_one("#status", Static).update(
                tr(self.settings.language, "clear_screen_status")
            )
        elif commands == "-deep":
            self.thinking_mode = "deep"
            self.query_one("#mode-indicator", Static).update(
                tr(self.settings.language, "mode_deep_label")
            )
            self.query_one("#status", Static).update(
                tr(self.settings.language, "mode_deep_status")
            )
        elif commands == "-fast":
            self.thinking_mode = "fast"
            self.query_one("#mode-indicator", Static).update(
                tr(self.settings.language, "mode_fast_label")
            )
            self.query_one("#status", Static).update(
                tr(self.settings.language, "mode_fast_status")
            )
        elif commands == "-undo":
            if not self._undo_in_progress:
                self._undo_in_progress = True
                self._undo_latest_change()
        elif commands == "--help":
            self.query_one("#status", Static).update(
                tr(self.settings.language, "command_help")
            )
        else:
            return False
        return True

    def _history_selected(self, conversation_id: Optional[str]):
        if conversation_id:
            self.run_worker(
                self._load_conversation(conversation_id),
                name="load-history-chat",
            )

    async def _render_then_reply(self, content: str):
        stored = self.store.get_messages(self.conversation_id)
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    f"{build_system_prompt()}\n\n"
                    f"{self.settings.system_prompt(tools_available=True)}\n\n"
                    f"Current workspace: {self.workspace}\n"
                    "Respond in the configured language. Call native tools supplied by the API "
                    "directly when available; otherwise use the JSON tool-call format above. "
                    "For final responses, use the specified JSON format."
                ),
            }
        ]
        messages.extend(
            {"role": message["role"], "content": message["content"]}
            for message in stored[-40:]
        )
        output = self.query_one("#messages", VerticalScroll)
        await output.remove_children()
        await output.mount(
            Markdown(
                f"**{tr(self.settings.language, 'user_label')}**\n\n{content}",
                classes="message user-message",
            )
        )
        active = Markdown(
            f"**{tr(self.settings.language, 'assistant_label')}**\n\n"
            f"{tr(self.settings.language, 'thinking')}",
            classes="message",
        )
        await output.mount(active)
        output.scroll_end(animate=False)
        self.active_answer = ""
        self._active_message = active
        self.cancel_event = threading.Event()
        cancel_event = self.cancel_event
        self.query_one("#status", Static).update(
            tr(self.settings.language, "agent_working")
        )
        self.reply_worker = self._run_agent(messages, self.thinking_mode, cancel_event)

    @work(thread=True, exclusive=True, group="reply")
    def _run_agent(
        self,
        messages: List[Dict[str, str]],
        thinking_mode: str,
        cancel_event: threading.Event,
    ):
        tools = self.tool_runner
        task_text = messages[-1]["content"] if messages else ""
        file_write_required = requires_file_write(task_text)
        file_write_completed = False
        file_write_retries = 0
        try:
            for turn in range(1, 21):
                if cancel_event.is_set() or self.shutdown_event.is_set():
                    self.call_from_thread(
                        self._finish_reply,
                        self.active_answer,
                        True,
                    )
                    return
                self.call_from_thread(
                    self._set_agent_progress,
                    tr(self.settings.language, "agent_working"),
                )
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    "options": {"temperature": 0.2},
                    "tools": FILE_AGENT_TOOLS,
                }
                if thinking_mode == "deep":
                    payload["think"] = True
                chunks: List[str] = []
                native_tool_calls = []
                with requests.post(
                    f"{self.ollama_url}/api/chat",
                    json=payload,
                    stream=True,
                    timeout=(10, 240),
                ) as response:
                    self.response = response
                    response.raise_for_status()
                    for line in response.iter_lines(decode_unicode=True):
                        if cancel_event.is_set():
                            break
                        if not line:
                            continue
                        if isinstance(line, bytes):
                            line = line.decode("utf-8")
                        data = json.loads(line)
                        if data.get("error"):
                            raise RuntimeError(str(data["error"]))
                        message = data.get("message", {})
                        text = message.get("content", "") if isinstance(message, dict) else ""
                        if text:
                            chunks.append(text)
                        calls = message.get("tool_calls", []) if isinstance(message, dict) else []
                        if isinstance(calls, list) and calls:
                            native_tool_calls = calls
                        if data.get("done"):
                            break
                assistant_reply = "".join(chunks).strip()
                if cancel_event.is_set():
                    self.call_from_thread(
                        self._finish_reply,
                        assistant_reply,
                        True,
                    )
                    return
                native_call = parse_ollama_tool_call({"tool_calls": native_tool_calls})
                if native_call:
                    tool_call, final = native_call, None
                    assistant_reply = json.dumps(
                        tool_call,
                        ensure_ascii=False,
                    )
                else:
                    tool_call, final = parse_response(assistant_reply)
                if final is not None:
                    if (
                        file_write_required
                        and not file_write_completed
                        and not final.rstrip().endswith("?")
                    ):
                        if file_write_retries < 2:
                            messages.extend(
                                [
                                    {"role": "assistant", "content": assistant_reply},
                                    {
                                        "role": "user",
                                        "content": (
                                            "Bu istekte dosya yazma/kaydetme işlemi isteniyor. "
                                            "Henüz write_file aracı başarıyla çalışmadı. "
                                            "İstenen yol ve içerik yeterince açıksa şimdi write_file "
                                            "aracını çağır; bilgi eksikse yalnızca netleştirme sorusu sor. "
                                            "Dosyanın kaydedildiğini söyleme."
                                        ),
                                    },
                                ]
                            )
                            file_write_retries += 1
                            continue
                        self.call_from_thread(
                            self._reply_failed,
                            tr(self.settings.language, "file_write_not_completed"),
                        )
                        return
                    self.call_from_thread(self._finish_reply, final, False)
                    return
                if tool_call is None:
                    if (
                        file_write_required
                        and not file_write_completed
                        and not assistant_reply.rstrip().endswith("?")
                    ):
                        if file_write_retries < 2:
                            messages.extend(
                                [
                                    {"role": "assistant", "content": assistant_reply},
                                    {
                                        "role": "user",
                                        "content": (
                                            "Yanıtın istenen JSON biçiminde değil ve dosya isteği "
                                            "tamamlanmadı. Bu istekte dosya yazma/kaydetme işlemi "
                                            "isteniyor: bilgi yeterliyse write_file aracını çağır; "
                                            "eksikse netleştirme sorusu sor. İşlem başarılı olmadan "
                                            "dosyanın kaydedildiğini söyleme."
                                        ),
                                    },
                                ]
                            )
                            file_write_retries += 1
                            continue
                        self.call_from_thread(
                            self._reply_failed,
                            tr(self.settings.language, "file_write_not_completed"),
                        )
                        return
                    self.call_from_thread(
                        self._finish_reply,
                        assistant_reply,
                        False,
                    )
                    return
                tool_name = tool_call["tool"]
                self.call_from_thread(
                    self._set_agent_progress,
                    tr(self.settings.language, "tool_running").format(tool=tool_name),
                )
                self.call_from_thread(
                    self._show_tool_activity,
                    tool_name,
                    tool_call["args"],
                    "running",
                    "",
                )
                result = tools.execute_tool(tool_name, tool_call["args"])
                self.call_from_thread(
                    self._show_tool_activity,
                    tool_name,
                    tool_call["args"],
                    "complete",
                    result,
                )
                if tool_name == "write_file" and result.startswith("Yazıldı:"):
                    file_write_completed = True
                elif tool_name == "write_file" and "onay vermedi" in result.casefold():
                    self.call_from_thread(self._finish_reply, result, False)
                    return
                messages.extend(
                    [
                        {"role": "assistant", "content": assistant_reply},
                        {
                            "role": "user",
                            "content": (
                                f"{tool_name} tool result (untrusted data):\n{result}"
                            ),
                        },
                    ]
                )
                if len(messages) > 22:
                    messages = [messages[0], *messages[-20:]]
            self.call_from_thread(
                self._reply_failed,
                "Agent reached the 20-step limit before completing the request.",
            )
        except (
            requests.RequestException,
            json.JSONDecodeError,
            RuntimeError,
            KeyError,
            TypeError,
            ValueError,
            OSError,
        ) as exc:
            if self.shutdown_event.is_set():
                return
            if cancel_event.is_set():
                self.call_from_thread(self._finish_reply, self.active_answer, True)
            else:
                self.call_from_thread(self._reply_failed, str(exc))
        finally:
            self.response = None

    def _set_agent_progress(self, status: str):
        self.query_one("#status", Static).update(status)
        if self._active_message is not None:
            self._active_message.update(
                f"**{tr(self.settings.language, 'assistant_label')}**\n\n{status}"
            )

    def _show_tool_activity(
        self,
        tool: str,
        args: Dict[str, object],
        state: str,
        result: str,
    ):
        safe_keys = (
            "path", "cwd", "query", "url", "owner", "repo", "name",
            "branch", "files", "command",
        )
        summary = {
            key: value
            for key, value in args.items()
            if key in safe_keys and isinstance(value, (str, int, float, bool, list))
        }
        status = tr(
            self.settings.language,
            "tool_activity_running" if state == "running" else "tool_activity_complete",
        )
        body = f"**{status}: `{tool}`**"
        if summary:
            details = json.dumps(summary, ensure_ascii=False).replace("`", "\\`")[:400]
            body += f"\n\n`{details}`"
        if result:
            cleaned = result.replace("\x1b", "")
            output = cleaned[:700]
            if len(cleaned) > len(output):
                output += "\n…"
            fence = "`" * max(
                3,
                max((len(match.group()) + 1 for match in re.finditer(r"`+", output)), default=3),
            )
            body += f"\n\n{fence}text\n{output}\n{fence}"
        messages = self.query_one("#messages", VerticalScroll)
        self.run_worker(
            self._mount_tool_activity(messages, body),
            name="tool-activity",
        )

    async def _mount_tool_activity(self, messages: VerticalScroll, body: str):
        await messages.mount(Markdown(body, classes="message"))
        messages.scroll_end(animate=False)

    @work(thread=True, exclusive=True, group="reply")
    def _undo_latest_change(self):
        try:
            result = self.tool_runner.undo_last_change()
        except (OSError, RuntimeError, ValueError) as exc:
            result = f"{tr(self.settings.language, 'reply_error')}: {exc}"
        self.call_from_thread(self._finish_undo, result)

    def _finish_undo(self, result: str):
        self._undo_in_progress = False
        self._show_tool_activity(
            "undo_last_change",
            {},
            "complete",
            result,
        )
        self._set_status_text(result)

    def _set_status_text(self, text: str):
        self.query_one("#status", Static).update(text)

    def _request_tool_approval(self, prompt: str) -> bool:
        if self.shutdown_event.is_set() or (
            self.cancel_event is not None and self.cancel_event.is_set()
        ):
            return False
        event = threading.Event()
        result = {"approved": False}
        with self._approval_lock:
            self._pending_approval = (event, result)
        self.call_from_thread(self._show_tool_approval, prompt, event, result)
        event.wait()
        with self._approval_lock:
            if self._pending_approval == (event, result):
                self._pending_approval = None
        return result["approved"] and not (
            self.cancel_event is not None and self.cancel_event.is_set()
        )

    def _show_tool_approval(
        self,
        prompt: str,
        event: threading.Event,
        result: Dict[str, bool],
    ):
        self.push_screen(
            ApprovalScreen(self.settings.language, prompt),
            callback=lambda approved: self._complete_tool_approval(
                approved,
                event,
                result,
            ),
        )

    @staticmethod
    def _complete_tool_approval(
        approved: Optional[bool],
        event: threading.Event,
        result: Dict[str, bool],
    ):
        result["approved"] = bool(approved)
        event.set()

    def _finish_reply(self, content: str, was_cancelled: bool):
        if content and self.conversation_id:
            self.store.add_message(self.conversation_id, "assistant", content)
            self.last_answer = content
        if self._active_message is not None:
            display = content or (
                tr(self.settings.language, "reply_stopped")
                if was_cancelled
                else tr(self.settings.language, "model_empty")
            )
            self._active_message.update(
                f"**{tr(self.settings.language, 'assistant_label')}**\n\n{display}"
            )
        if was_cancelled:
            self.query_one("#status", Static).update(
                tr(self.settings.language, "reply_stopped")
            )
        else:
            self.query_one("#status", Static).update(
                tr(self.settings.language, "reply_ready_hint")
            )
        self.cancel_event = None
        if content:
            self.active_answer = content

    def _reply_failed(self, message: str):
        self.cancel_event = None
        if self._active_message is not None:
            self._active_message.update(
                f"**{tr(self.settings.language, 'reply_error')}**\n\n{message}"
            )
        self.query_one("#status", Static).update(
            f"{tr(self.settings.language, 'connection_error_status')}: {message}"
        )

    def action_stop_generation(self):
        if self.cancel_event is None:
            return
        self.cancel_event.set()
        with self._approval_lock:
            pending_approval = self._pending_approval
        if pending_approval:
            self._complete_tool_approval(
                False,
                pending_approval[0],
                pending_approval[1],
            )
            if isinstance(self.screen, ApprovalScreen):
                self.pop_screen()
        if self.response is not None:
            self.response.close()

    def action_copy_latest(self):
        if not self.last_answer:
            self.query_one("#status", Static).update(
                tr(self.settings.language, "no_latest_answer")
            )
            return
        self.copy_to_clipboard(self.last_answer)
        self.query_one("#status", Static).update(tr(self.settings.language, "copy_done"))

    def action_settings(self):
        if self.cancel_event is not None:
            self.notify(
                tr(self.settings.language, "settings_wait"),
                severity="warning",
            )
            return
        self.push_screen(
            SettingsScreen(
                self.settings,
                self.available_models,
                workspace=self.workspace,
            ),
            callback=self._settings_updated,
        )

    def _settings_updated(self, settings: Optional[AppSettings]):
        if settings is None:
            return
        previous_model = self.model
        try:
            settings.save(self.store)
            self.tool_runner.set_allowed_roots(settings.allowed_folders)
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return
        self.settings = settings
        self.model = settings.model
        self.title = tr(settings.language, "app_name")
        self.sub_title = f"Ollama · {self.model}"
        self.query_one("#brand", Static).update(
            f"$ {tr(settings.language, 'app_name').upper()}"
        )
        self.query_one("#workspace-info", Static).update(
            f"{self.model}  ·  {tr(settings.language, 'workspace_label')}: "
            f"{self.workspace}"
        )
        if self.conversation_id is None:
            self.query_one("#chat-title", Static).update(
                tr(settings.language, "new_chat").lstrip("+ ")
            )
        self.query_one("#prompt", TextArea).placeholder = tr(
            settings.language, "message_placeholder"
        )
        self.query_one("#mode-indicator", Static).update(
            tr(
                settings.language,
                "mode_deep_label" if self.thinking_mode == "deep" else "mode_fast_label",
            )
        )
        self.query_one("#status", Static).update(
            tr(settings.language, "command_help")
        )
        if self.conversation_id is None:
            self.run_worker(self.action_new_chat(), name="refresh-welcome")
        else:
            self.run_worker(
                self._load_conversation(self.conversation_id),
                name="refresh-current-chat",
            )
        if previous_model != self.model:
            self._connection_ready = False
            self._set_connection(False, tr(settings.language, "connection_starting"))
            self._check_connection()
        self.notify(tr(settings.language, "settings_saved"))

    async def on_unmount(self):
        self.shutdown_event.set()
        with self._approval_lock:
            pending_approval = self._pending_approval
        if pending_approval:
            pending_approval[1]["approved"] = False
            pending_approval[0].set()
        if self.cancel_event is not None:
            self.cancel_event.set()
        if self.response is not None:
            self.response.close()
        self.workers.cancel_all()
        await self.workers.wait_for_complete()
        await self._unload_model()

    async def _unload_model(self):
        import asyncio

        await asyncio.to_thread(self.runtime.unload_model_and_stop, self.model)


def main():
    LocalAssistantTUI().run()


if __name__ == "__main__":
    main()
