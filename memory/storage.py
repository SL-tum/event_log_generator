import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SQLiteManager:
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        #self._migrate_history_table()
        self._create_history_table()
        self._create_preference_table()
        self._create_data_table()
        self._create_constrain_table()
        self._create_requirement_table()
        self._create_intent_table()
        self._create_knowledge_table()
        self._create_messages_table()

    def _migrate_history_table(self) -> None:
        """
        If a pre-existing history table had the old group-chat columns,
        rename it, create the new schema, copy the intersecting data, then
        drop the old table.
        """
        with self._lock:
            try:
                # Start a transaction
                self.connection.execute("BEGIN")
                cur = self.connection.cursor()

                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='history'")
                if cur.fetchone() is None:
                    self.connection.execute("COMMIT")
                    return  # nothing to migrate

                cur.execute("PRAGMA table_info(history)")
                old_cols = {row[1] for row in cur.fetchall()}

                expected_cols = {
                    "id",
                    "session_id",
                    "event",
                    "created_at",
                    "is_deleted",
                    "role",
                }

                if old_cols == expected_cols:
                    self.connection.execute("COMMIT")
                    return

                logger.info("Migrating history table to new schema (no convo columns).")

                # Clean up any existing history_old table from previous failed migration
                cur.execute("DROP TABLE IF EXISTS history_old")

                # Rename the current history table
                cur.execute("ALTER TABLE history RENAME TO history_old")

                # Create the new history table with updated schema
                cur.execute(
                    """
                    CREATE TABLE history (
                        id           TEXT PRIMARY KEY,
                        session_id    TEXT,
                        event        TEXT,
                        created_at   DATETIME,
                        is_deleted   INTEGER,
                        role         TEXT
                    )
                """
                )
                cur.execute("CREATE INDEX IF NOT EXISTS idx_session_history ON history (session_id, created_at)")

                # Copy data from old table to new table
                intersecting = list(expected_cols & old_cols)
                if intersecting:
                    cols_csv = ", ".join(intersecting)
                    cur.execute(f"INSERT INTO history ({cols_csv}) SELECT {cols_csv} FROM history_old")

                # Drop the old table
                cur.execute("DROP TABLE history_old")

                # Commit the transaction
                self.connection.execute("COMMIT")
                logger.info("History table migration completed successfully.")

            except Exception as e:
                # Rollback the transaction on any error
                self.connection.execute("ROLLBACK")
                logger.error(f"History table migration failed: {e}")
                raise

    def _create_history_table(self) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS history (
                        id           TEXT PRIMARY KEY,
                        session_id    TEXT,
                        event        TEXT,
                        created_at   DATETIME,
                        is_deleted   INTEGER,
                        role         TEXT
                    )
                """
                )
                #self.connection.execute("CREATE INDEX IF NOT EXISTS idx_session_history ON history (session_id, created_at)")
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to create history table: {e}")
                raise

    def _create_requirement_table(self) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS requirement (
                        id           TEXT PRIMARY KEY,
                        session_id    TEXT,
                        event        TEXT,
                        stored_at   DATETIME
                    )
                """
                )
                #self.connection.execute("CREATE INDEX IF NOT EXISTS idx_session_requirement ON requirement (session_id, created_at)")
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to create requirement table: {e}")
                raise

    def _create_intent_table(self) -> None:
            with self._lock:
                try:
                    self.connection.execute("BEGIN")
                    self.connection.execute(
                        """
                        CREATE TABLE IF NOT EXISTS intent (
                            id           TEXT PRIMARY KEY,
                            session_id    TEXT,
                            event        TEXT,
                            stored_at   DATETIME
                        )
                    """
                    )
                    #self.connection.execute("CREATE INDEX IF NOT EXISTS idx_session_requirement ON requirement (session_id, created_at)")
                    self.connection.execute("COMMIT")
                except Exception as e:
                    self.connection.execute("ROLLBACK")
                    logger.error(f"Failed to create intent table: {e}")
                    raise

    def _create_knowledge_table(self) -> None:
            with self._lock:
                try:
                    self.connection.execute("BEGIN")
                    self.connection.execute(
                        """
                        CREATE TABLE IF NOT EXISTS knowledge (
                            id           TEXT PRIMARY KEY,
                            session_id    TEXT,
                            event        TEXT,
                            stored_at   DATETIME
                        )
                    """
                    )
                    #self.connection.execute("CREATE INDEX IF NOT EXISTS idx_session_requirement ON requirement (session_id, created_at)")
                    self.connection.execute("COMMIT")
                except Exception as e:
                    self.connection.execute("ROLLBACK")
                    logger.error(f"Failed to create knowledge table: {e}")
                    raise

    def _create_preference_table(self) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS preference (
                        id           TEXT PRIMARY KEY,
                        session_id    TEXT,
                        event        TEXT,
                        stored_at   DATETIME
                    )
                """
                )
                #self.connection.execute("CREATE INDEX IF NOT EXISTS idx_session_preference ON preference (preference, created_at)")
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to create preference table: {e}")
                raise

    def _create_data_table(self) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS "data" (
                        id           TEXT PRIMARY KEY,
                        session_id    TEXT,
                        event        TEXT,
                        stored_at   DATETIME
                    )
                """
                )
                #self.connection.execute("CREATE INDEX IF NOT EXISTS idx_session_data ON data (session_id, created_at)")
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to create data table: {e}")
                raise

    def _create_constrain_table(self) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS constrain (
                        id           TEXT PRIMARY KEY,
                        session_id    TEXT,
                        event        TEXT,
                        stored_at   DATETIME
                    )
                """
                )
                #self.connection.execute("CREATE INDEX IF NOT EXISTS idx_session_history ON history (session_id, created_at)")
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to create constrain table: {e}")
                raise            

    def _create_messages_table(self) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id TEXT PRIMARY KEY,
                        session_scope TEXT,
                        role TEXT,
                        content TEXT,
                        name TEXT,
                        created_at DATETIME
                    )
                """
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to create messages table: {e}")
                raise

    def add_history(
        self,
        session_id: str,
        event: str,
        *,
        created_at: Optional[str] = None,
        is_deleted: int = 0,
        role: Optional[str] = None,
    ) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    """
                    INSERT INTO history (
                        id, session_id, event,
                        created_at, is_deleted, role
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        str(uuid.uuid4()),
                        session_id,
                        event,
                        created_at,
                        is_deleted,
                        role,
                    ),
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to add history record: {e}")
                raise
    
    """ 
    def add_item(
        self,
        session_id: str,
        table: str,
        event: str,
        *,
        stored_at: Optional[str] = None,
    ) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                sql = f"""
                    #INSERT INTO "{table}" (
                    #    id, session_id, event,
                    #    stored_at
                    #)
                    #VALUES (?, ?, ?, ?)
    """
                self.connection.execute(
                    sql,
                    (
                        str(uuid.uuid4()),
                        session_id,
                        event,
                        stored_at,
                    ),
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to add history record: {e}")
                raise

    """
    
    def batch_add_history(self, records: List[Dict[str, Any]]) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.executemany(
                    """
                    INSERT INTO history (
                        id, session_id, event,
                        created_at, is_deleted, role
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    [
                        (
                            str(uuid.uuid4()),
                            record.get("session_id"),
                            record.get("event"),
                            record.get("created_at"),
                            record.get("is_deleted", 0),
                            record.get("role"),
                        )
                        for record in records
                    ],
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to batch add history records: {e}")
                raise

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self.connection.execute(
                """
                SELECT id, session_id, event, created_at, is_deleted, role
                FROM history
                WHERE session_id = ?
                ORDER BY created_at ASC
            """,
                (session_id,),
            )
            rows = cur.fetchall()

        return [
            {
                "id": r[0],
                "session_id": r[1],
                "event": r[2],
                "created_at": r[3],
                "is_deleted": bool(r[4]),
                "role": r[5],
            }
            for r in rows
        ]

    def get_table(self, key: str, session_id: str) -> Optional[List[str]]:
        with self._lock:
            sql = f"""
            SELECT event
            FROM "{key}"
            WHERE session_id = ?
            ORDER BY stored_at ASC
        """
            try:
                cur = self.connection.execute(sql, (session_id,))
                rows = cur.fetchall()
                return [r[0]
                    for r in rows
                ]
            except sqlite3.OperationalError as e:
                logger.error(f"SQL Error in get_table for key [{key}]: {e}")
                return []

    def update_table(self, table, session_id, content) -> None:
        #self.clear_table(table)
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                #print(type(content))
                if not content:
                    print(f"In update_table function, content is None")
                else:
                    stored_at = datetime.now(timezone.utc).isoformat()
                    self.connection.execute(
                        f"""
                        INSERT INTO "{table}" (
                            id, session_id, event, stored_at
                        )
                        VALUES (?, ?, ?, ?)
                    """,
                        (
                            str(uuid.uuid4()),
                            session_id,
                            content,
                            stored_at
                        ),
                    )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to create constrain table: {e}")
                raise

    def clear_table(self, table) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    f"""
                    DELETE FROM "{table}"
                """
                )
                #self.connection.execute("CREATE INDEX IF NOT EXISTS idx_session_history ON history (session_id, created_at)")
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to create constrain table: {e}")
                raise

    def save_messages(self, messages: List[Dict[str, Any]], session_scope: str) -> None:
        if not messages:
            return
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                now = datetime.now(timezone.utc).isoformat()
                for message in messages:
                    self.connection.execute(
                        """
                        INSERT INTO messages (id, session_scope, role, content, name, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (
                            str(uuid.uuid4()),
                            session_scope,
                            message.get("role"),
                            message.get("content"),
                            message.get("name"),
                            now,
                        ),
                    )
                # Evict old messages beyond the most recent 10 for this scope.
                # Wrapped in a derived table to force SQLite to materialize the
                # ORDER BY before the outer NOT IN evaluates it.
                self.connection.execute(
                    """
                    DELETE FROM messages WHERE session_scope = ? AND id NOT IN (
                        SELECT id FROM (
                            SELECT id FROM messages WHERE session_scope = ? ORDER BY created_at DESC LIMIT 10
                        )
                    )
                """,
                    (session_scope, session_scope),
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to save messages: {e}")
                raise

    def get_last_messages(self, session_scope: str, limit: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            # Subquery picks the latest N rows (DESC + LIMIT), outer query
            # re-sorts them chronologically (ASC) for the caller.
            cur = self.connection.execute(
                """
                SELECT role, content, name, created_at FROM (
                    SELECT role, content, name, created_at
                    FROM messages
                    WHERE session_scope = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ) ORDER BY created_at ASC
            """,
                (session_scope, limit),
            )
            rows = cur.fetchall()

        return [
            {
                "role": r[0],
                "content": r[1],
                "name": r[2],
                "created_at": r[3],
            }
            for r in rows
        ]

    def reset(self) -> None:
        """Drop and recreate the history and messages tables."""
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute("DROP TABLE IF EXISTS history")
                self.connection.execute("DROP TABLE IF EXISTS messages")
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to reset tables: {e}")
                raise
        self._create_history_table()
        self._create_messages_table()

    def close(self) -> None:
        if self.connection:
            self.connection.close()
            self.connection = None

    def __del__(self):
        self.close()
