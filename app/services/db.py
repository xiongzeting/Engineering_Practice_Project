from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from app.config import settings


def _connect() -> sqlite3.Connection:
    db_file = Path(settings.db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              username TEXT NOT NULL UNIQUE,
              password_hash TEXT NOT NULL,
              role TEXT NOT NULL DEFAULT 'student',
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS ocr_records (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              engine TEXT NOT NULL,
              ocr_text TEXT NOT NULL,
              steps_count INTEGER NOT NULL DEFAULT 0,
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS grading_records (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              engine TEXT NOT NULL,
              total_score REAL NOT NULL,
              steps_count INTEGER NOT NULL DEFAULT 0,
              ocr_text TEXT NOT NULL,
              grading_meta_json TEXT NOT NULL DEFAULT '{}',
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(user_id) REFERENCES users(id)
            );
            """
        )
        conn.commit()


def create_user(username: str, password_hash: str, role: str) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO users(username, password_hash, role) VALUES(?,?,?)",
            (username, password_hash, role),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_user_by_username(username: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, role, created_at FROM users WHERE username=?",
            (username,),
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, role, created_at FROM users WHERE id=?",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def save_ocr_record(user_id: int, engine: str, ocr_text: str, steps_count: int) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO ocr_records(user_id, engine, ocr_text, steps_count) VALUES(?,?,?,?)",
            (user_id, engine, ocr_text, steps_count),
        )
        conn.commit()
        return int(cur.lastrowid)


def save_grading_record(
    user_id: int,
    engine: str,
    total_score: float,
    steps_count: int,
    ocr_text: str,
    grading_meta: dict[str, Any],
) -> int:
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO grading_records(user_id, engine, total_score, steps_count, ocr_text, grading_meta_json)
            VALUES(?,?,?,?,?,?)
            """,
            (user_id, engine, total_score, steps_count, ocr_text, json.dumps(grading_meta, ensure_ascii=False)),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_ocr_records(user_id: int, limit: int = 20) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, engine, steps_count, created_at
            FROM ocr_records
            WHERE user_id=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, max(1, min(limit, 200))),
        ).fetchall()
    return [dict(x) for x in rows]


def list_grading_records(user_id: int, limit: int = 20) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, engine, total_score, steps_count, created_at
            FROM grading_records
            WHERE user_id=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, max(1, min(limit, 200))),
        ).fetchall()
    return [dict(x) for x in rows]
