import logging
import math
import re
import sqlite3
import struct
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple

from openai import OpenAI

from invo_lens.config import settings
from invo_lens.schemas import InvoiceItem, NomenclatureItem

logger = logging.getLogger(__name__)


def _normalize(text: Optional[str]) -> str:
    if not text:
        return ""
    return " ".join(text.strip().lower().split())


def _extract_numbers(text: str) -> Set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)?", text))


def _numbers_compatible(a: str, b: str) -> bool:
    nums_a = _extract_numbers(a)
    nums_b = _extract_numbers(b)
    if not nums_a or not nums_b:
        return True
    return nums_a == nums_b


def _pack_embedding(vec: List[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack_embedding(blob: bytes) -> List[float]:
    count = len(blob) // 4
    return list(struct.unpack(f"{count}f", blob))


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _get_local_embedding(text: str) -> Optional[List[float]]:
    """Compute embedding via local LM Studio /v1/embeddings endpoint if configured."""
    if not settings.lm_studio_embedding_model or not text.strip():
        return None
    try:
        client = OpenAI(
            base_url=settings.lm_studio_base_url,
            api_key=settings.lm_studio_api_key,
            timeout=15.0,
        )
        resp = client.embeddings.create(
            model=settings.lm_studio_embedding_model,
            input=text,
        )
        return resp.data[0].embedding
    except Exception as exc:
        logger.warning("Local embedding request failed: %s", exc)
        return None


class SQLiteNomenclatureStore:
    """SQLite-backed store for client nomenclature and multi-level item matching."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS nomenclature (
                    client_id TEXT NOT NULL,
                    code TEXT NOT NULL,
                    name TEXT NOT NULL,
                    sku TEXT,
                    barcode TEXT,
                    unit TEXT,
                    embedding BLOB,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (client_id, code)
                )
                """
            )
            for col_def in ("barcode TEXT", "embedding BLOB"):
                try:
                    conn.execute(f"ALTER TABLE nomenclature ADD COLUMN {col_def}")
                except sqlite3.OperationalError:
                    pass

    def upsert_items(self, client_id: str, items: List[NomenclatureItem]) -> int:
        if not items:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        rows = []
        for item in items:
            vec = _get_local_embedding(item.name)
            emb_blob = _pack_embedding(vec) if vec else None
            rows.append((client_id, item.code, item.name, item.sku, item.barcode, item.unit, emb_blob, now))

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO nomenclature (client_id, code, name, sku, barcode, unit, embedding, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(client_id, code) DO UPDATE SET
                    name = excluded.name,
                    sku = excluded.sku,
                    barcode = excluded.barcode,
                    unit = excluded.unit,
                    embedding = COALESCE(excluded.embedding, nomenclature.embedding),
                    updated_at = excluded.updated_at
                """,
                rows,
            )
        return len(rows)

    def list_items(self, client_id: str) -> List[NomenclatureItem]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT code, name, sku, barcode, unit
                FROM nomenclature
                WHERE client_id = ?
                ORDER BY code ASC
                """,
                (client_id,),
            ).fetchall()
        return [
            NomenclatureItem(
                code=row["code"],
                name=row["name"],
                sku=row["sku"],
                barcode=row["barcode"],
                unit=row["unit"],
            )
            for row in rows
        ]

    def _list_items_with_embeddings(
        self, client_id: str
    ) -> List[Tuple[NomenclatureItem, Optional[List[float]]]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT code, name, sku, barcode, unit, embedding
                FROM nomenclature
                WHERE client_id = ?
                ORDER BY code ASC
                """,
                (client_id,),
            ).fetchall()
        result: List[Tuple[NomenclatureItem, Optional[List[float]]]] = []
        for row in rows:
            item = NomenclatureItem(
                code=row["code"],
                name=row["name"],
                sku=row["sku"],
                barcode=row["barcode"],
                unit=row["unit"],
            )
            vec = _unpack_embedding(row["embedding"]) if row["embedding"] else None
            result.append((item, vec))
        return result

    def delete_client_items(self, client_id: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM nomenclature WHERE client_id = ?",
                (client_id,),
            )
            return cur.rowcount

    def match_invoice_items(
        self,
        client_id: Optional[str],
        invoice_items: List[InvoiceItem],
    ) -> None:
        """Match recognized invoice items against client nomenclature.

        Cascade:
          1. Exact match: barcode -> sku -> normalized name.
          2. Fuzzy lexical match (SequenceMatcher >= 0.85) with number-consistency guard.
          3. Local embedding cosine similarity (LM Studio /v1/embeddings) with number guard.
        """
        if not client_id or not invoice_items:
            return

        catalog = self._list_items_with_embeddings(client_id)
        if not catalog:
            return

        by_barcode: Dict[str, str] = {}
        by_sku: Dict[str, str] = {}
        by_name: Dict[str, str] = {}
        for entry, _ in catalog:
            norm_barcode = _normalize(entry.barcode)
            if norm_barcode and norm_barcode not in by_barcode:
                by_barcode[norm_barcode] = entry.code
            norm_sku = _normalize(entry.sku)
            if norm_sku and norm_sku not in by_sku:
                by_sku[norm_sku] = entry.code
            norm_name = _normalize(entry.name)
            if norm_name and norm_name not in by_name:
                by_name[norm_name] = entry.code

        for item in invoice_items:
            # Level 1: Exact match
            norm_barcode = _normalize(item.barcode)
            if norm_barcode and norm_barcode in by_barcode:
                item.matched_code = by_barcode[norm_barcode]
                continue

            norm_sku = _normalize(item.sku)
            if norm_sku and norm_sku in by_sku:
                item.matched_code = by_sku[norm_sku]
                continue

            norm_name = _normalize(item.name)
            if norm_name and norm_name in by_name:
                item.matched_code = by_name[norm_name]
                continue

            # Level 2: Fuzzy lexical match with number guard
            best_fuzzy_code: Optional[str] = None
            best_fuzzy_score = 0.0
            for entry, _ in catalog:
                entry_norm = _normalize(entry.name)
                if not _numbers_compatible(norm_name, entry_norm):
                    continue
                score = SequenceMatcher(None, norm_name, entry_norm).ratio()
                if score > best_fuzzy_score:
                    best_fuzzy_score = score
                    best_fuzzy_code = entry.code

            if best_fuzzy_code and best_fuzzy_score >= 0.85:
                item.matched_code = best_fuzzy_code
                continue

            # Level 3: Local embedding similarity with number guard
            if settings.lm_studio_embedding_model:
                query_vec = _get_local_embedding(item.name)
                if query_vec:
                    best_emb_code: Optional[str] = None
                    best_emb_score = 0.0
                    for entry, entry_vec in catalog:
                        if not entry_vec or not _numbers_compatible(norm_name, _normalize(entry.name)):
                            continue
                        sim = _cosine_similarity(query_vec, entry_vec)
                        if sim > best_emb_score:
                            best_emb_score = sim
                            best_emb_code = entry.code

                    if best_emb_code and best_emb_score >= settings.nomenclature_match_threshold:
                        item.matched_code = best_emb_code


nomenclature_store = SQLiteNomenclatureStore(settings.db_path)
