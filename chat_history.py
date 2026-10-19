"""Persistent SQLite storage for desktop chat conversations."""

import os
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional


def default_database_path() -> Path:
    local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    return local_app_data / "LocalOllamaAgent" / "chat_history.db"


class ChatHistoryStore:
    def __init__(self, database_path: Optional[Path] = None):
        self.database_path = Path(database_path or default_database_path())
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS messages_conversation_idx "
                "ON messages(conversation_id, id)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS app_preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def create_conversation(self, title: str = "Yeni sohbet") -> str:
        conversation_id = str(uuid.uuid4())
        now = self._now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversations(id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (conversation_id, title, now, now),
            )
        return conversation_id

    def list_conversations(self) -> List[Dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, title, created_at, updated_at FROM conversations "
                "ORDER BY updated_at DESC, created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_messages(self, conversation_id: str) -> List[Dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT role, content, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("Mesaj rolü user veya assistant olmalıdır.")
        if not content.strip():
            raise ValueError("Boş mesaj kaydedilemez.")
        now = self._now()
        with self._connect() as connection:
            conversation = connection.execute(
                "SELECT title FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if conversation is None:
                raise ValueError("Sohbet bulunamadı.")
            connection.execute(
                "INSERT INTO messages(conversation_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (conversation_id, role, content, now),
            )
            title = conversation["title"]
            if role == "user" and title == "Yeni sohbet":
                title = " ".join(content.split())[:48] or "Yeni sohbet"
            connection.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, conversation_id),
            )

    def delete_conversation(self, conversation_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,),
            )

    def delete_all_conversations(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM conversations")

    def get_preferences(self) -> Dict[str, str]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM app_preferences WHERE key = 'user_settings'"
            ).fetchone()
        if row is None:
            return {}
        try:
            value = json.loads(row["value"])
        except json.JSONDecodeError as exc:
            raise ValueError("Kayıtlı kullanıcı ayarları bozuk.") from exc
        return value if isinstance(value, dict) else {}

    def save_preferences(self, preferences: Dict[str, str]) -> None:
        serialized = json.dumps(preferences, ensure_ascii=False)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO app_preferences(key, value) VALUES ('user_settings', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (serialized,),
            )
