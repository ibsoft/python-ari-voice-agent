"""Elastic vector search helper for the ARI voice agent."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from collections import OrderedDict
import threading

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ApiError, TransportError
from openai import OpenAI

log = logging.getLogger("elastic_retriever")


class ElasticVectorRetriever:
    """Thin wrapper that embeds user text and runs an Elastic kNN query."""

    def __init__(self, cfg: Dict[str, Any], oa_client: OpenAI):
        self.cfg = cfg
        self.oa = oa_client
        self.enabled = bool(cfg.get("enabled"))
        if not self.enabled:
            raise ValueError("ElasticVectorRetriever requires elasticsearch.enabled=true")

        hosts = cfg.get("hosts") or [cfg.get("host") or "http://localhost:9200"]
        if isinstance(hosts, str):
            hosts = [hosts]

        username = cfg.get("username")
        password = cfg.get("password")
        api_key = cfg.get("api_key")
        verify_certs = cfg.get("verify_certs", True)
        ca_certs = cfg.get("ca_certs")
        request_timeout = float(cfg.get("request_timeout", 10.0))

        es_kwargs: Dict[str, Any] = {
            "hosts": hosts,
            "verify_certs": verify_certs,
            "request_timeout": request_timeout,
        }

        if ca_certs:
            es_kwargs["ca_certs"] = ca_certs

        if api_key:
            es_kwargs["api_key"] = api_key
        elif username and password:
            es_kwargs["basic_auth"] = (username, password)

        self.client = Elasticsearch(**es_kwargs)

        self.vector_field = cfg.get("vector_field") or "embedding"
        self.text_field = cfg.get("text_field") or "content"
        self.metadata_fields = cfg.get("metadata_fields") or []
        self.top_k = int(cfg.get("top_k", 5))
        self.num_candidates = int(cfg.get("num_candidates", max(self.top_k * 4, 40)))
        self.min_score = float(cfg.get("min_score", 0.0))
        self.default_embedding_model = cfg.get("embedding_model", "text-embedding-3-large")
        self.conversation_logging_enabled = bool(cfg.get("conversation_log_enabled"))
        self.embedding_cache_size = int(cfg.get("embedding_cache_size", 32))
        self._embedding_cache: "OrderedDict[str, List[float]]" = OrderedDict()
        self._cache_lock = threading.Lock()

        self.index_configs = self._normalize_indexes(cfg)
        self.searchable_indexes = [info for info in self.index_configs if info["searchable"]]
        self.conversation_index = next(
            (info for info in self.index_configs if info.get("role") == "conversation_memory"),
            None
        )

        if not self.searchable_indexes:
            raise ValueError("ElasticVectorRetriever requires at least one searchable index in elasticsearch.indexes")

        # Keep first searchable index for backwards compatibility/logging
        self.index = self.searchable_indexes[0]["name"]

    def embed(self, text: str, model_override: Optional[str] = None) -> List[float]:
        if not text.strip():
            return []
        model = model_override or self.default_embedding_model
        key = f"{model}:{text.strip()}"
        with self._cache_lock:
            if key in self._embedding_cache:
                self._embedding_cache.move_to_end(key)
                return list(self._embedding_cache[key])
        resp = self.oa.embeddings.create(model=model, input=text)
        embedding = resp.data[0].embedding  # type: ignore[attr-defined]
        with self._cache_lock:
            self._embedding_cache[key] = embedding
            if len(self._embedding_cache) > self.embedding_cache_size:
                self._embedding_cache.popitem(last=False)
        return embedding

    def _normalize_indexes(self, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        seen = set()

        def _coerce_field(value):
            if value is None:
                return None
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, (list, tuple)):
                return [v for v in value if isinstance(v, str) and v.strip()]
            return None

        def _append(name: str, meta: Dict[str, Any]):
            name = name.strip()
            if not name or name in seen:
                return
            seen.add(name)
            normalized.append({
                "name": name,
                "embedding_model": meta.get("embedding_model"),
                "searchable": meta.get("searchable", True),
                "role": meta.get("role") or "knowledge",
                "text_field": _coerce_field(meta.get("text_field")),
                "metadata_fields": _coerce_field(meta.get("metadata_fields"))
            })

        indexes_cfg = cfg.get("indexes") or cfg.get("indices")
        if isinstance(indexes_cfg, str):
            indexes_cfg = [indexes_cfg]

        if isinstance(indexes_cfg, list):
            for entry in indexes_cfg:
                if isinstance(entry, str):
                    _append(entry, {})
                elif isinstance(entry, dict):
                    name = entry.get("name") or entry.get("index")
                    if name:
                        _append(name, entry)

        legacy_index = cfg.get("index")
        if isinstance(legacy_index, str):
            _append(legacy_index, {})

        if not normalized:
            raise ValueError("No elasticsearch indexes configured")
        return normalized

    def search(self, query: str, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not query.strip():
            return []

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for info in self.searchable_indexes:
            model = info.get("embedding_model") or self.default_embedding_model
            grouped.setdefault(model, []).append(info)

        source_fields = None

        aggregated_hits: List[Dict[str, Any]] = []

        for model, indexes in grouped.items():
            try:
                vector = self.embed(query, model_override=model)
            except Exception as exc:
                log.error("Embedding failed for model %s: %s", model, exc)
                continue

            knn_query: Dict[str, Any] = {
                "field": self.vector_field,
                "query_vector": vector,
                "k": self.top_k,
                "num_candidates": self.num_candidates,
            }
            if metadata_filter:
                knn_query["filter"] = metadata_filter

            for idx in indexes:
                name = idx["name"]
                txt_field = idx.get("text_field") or self.text_field
                meta_fields = idx.get("metadata_fields") or self.metadata_fields
                try:
                    search_kwargs = {
                        "index": name,
                        "knn": knn_query,
                        "size": self.top_k,
                    }
                    if source_fields is not None:
                        search_kwargs["source"] = {"includes": list(source_fields)}
                    resp = self.client.search(**search_kwargs)
                except (ApiError, TransportError) as exc:
                    log.error("Elastic search failed for index %s: %s", name, exc)
                    continue

                for hit in resp.get("hits", {}).get("hits", []):
                    hit_copy = dict(hit)
                    hit_copy["_index"] = name
                    hit_copy["_embedding_model"] = model
                    hit_copy["_text_field"] = txt_field
                    hit_copy["_metadata_fields"] = meta_fields
                    aggregated_hits.append(hit_copy)

        aggregated_hits.sort(key=lambda h: float(h.get("_score") or 0.0), reverse=True)

        results: List[Dict[str, Any]] = []
        for hit in aggregated_hits[: self.top_k]:
            score = float(hit.get("_score") or 0.0)
            if self.min_score and score < self.min_score:
                continue
            source = hit.get("_source") or {}
            txt_field = hit.get("_text_field") or self.text_field
            meta_fields = hit.get("_metadata_fields") or self.metadata_fields
            text = self._extract_text(source, txt_field)
            metadata = self._extract_metadata(source, meta_fields)
            metadata.setdefault("index", hit.get("_index"))
            if hit.get("_embedding_model"):
                metadata.setdefault("embedding_model", hit.get("_embedding_model"))
            results.append({"text": text, "score": score, "metadata": metadata})
        return results

    def log_conversation_turn(self, document: Dict[str, Any]) -> bool:
        if not self.conversation_logging_enabled:
            return False
        if not self.conversation_index:
            log.debug("Conversation logging requested but no conversation index configured")
            return False
        if not document:
            return False

        doc = dict(document)
        if "text" in doc:
            txt_field = self.conversation_index.get("text_field") or "content"
            doc.setdefault(txt_field, doc["text"])

        doc.setdefault("@timestamp", datetime.now(timezone.utc).isoformat())
        try:
            self.client.index(index=self.conversation_index["name"], document=doc)
            return True
        except (ApiError, TransportError) as exc:
            log.error("Elastic conversation log failed: %s", exc)
        except Exception as exc:
            log.error("Unexpected error during conversation log: %s", exc)
        return False

    def _extract_text(self, source: Dict[str, Any], field_spec: Optional[Any]) -> str:
        if not source:
            return ""
        candidates: List[str] = []
        if isinstance(field_spec, str) and field_spec:
            candidates = [field_spec]
        elif isinstance(field_spec, (list, tuple)):
            candidates = [f for f in field_spec if isinstance(f, str) and f]

        for field in candidates:
            val = source.get(field)
            if isinstance(val, str) and val.strip():
                return val.strip()

        # fallback: first non-empty string value
        for val in source.values():
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""

    def _extract_metadata(self, source: Dict[str, Any], field_spec: Optional[Any]) -> Dict[str, Any]:
        if not source:
            return {}
        fields: List[str] = []
        if isinstance(field_spec, str):
            fields = [field_spec]
        elif isinstance(field_spec, (list, tuple)):
            fields = [f for f in field_spec if isinstance(f, str)]

        if not fields:
            return {}
        meta = {}
        for field in fields:
            val = source.get(field)
            if val is not None:
                meta[field] = val
        return meta

    @staticmethod
    def format_results(results: List[Dict[str, Any]]) -> str:
        if not results:
            return ""
        blocks: List[str] = []
        for idx, doc in enumerate(results, 1):
            meta = doc.get("metadata") or {}
            meta_parts = [str(v) for v in meta.values() if v]
            meta_line = f" ({' | '.join(meta_parts)})" if meta_parts else ""
            text = doc.get("text") or ""
            score = doc.get("score")
            score_txt = f"score={score:.2f}" if isinstance(score, (int, float)) else ""
            blocks.append(f"Τεκμήριο {idx} {score_txt}{meta_line}:\n{text}".strip())
        return "\n\n".join(blocks)
