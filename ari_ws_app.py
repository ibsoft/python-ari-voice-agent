#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Asterisk ARI IVR — With Fixed Call Transfer Tool Implementation
Forces retrieval via file_search for every user question (when vector stores are available).
Plays a short human "ack"/exclamation and a waiter phrase on EVERY user turn.
If no answer is found from file_search, it says so and suggests transfer to a technician.
Keeps short-term memory of the last 3 user/assistant turns.
If caller declares their name, addresses them as “Κύριε/Κυρία <Επώνυμο>” when feasible; otherwise “Κ. <Επώνυμο>”.

Update (2025-10-12):
- Added pending transfer offer detection + affirmative follow-up handling.
  If assistant previously offered transfer and the caller replies "ναι", "παρακαλώ", "οκ", etc.,
  we immediately proceed to transfer; on failure we speak a clear fallback and continue.
"""

import asyncio
import socket
import struct
import time
import yaml
import logging
import threading
import os
import signal
import io
import wave
import tempfile
import re
import random
import contextlib
import json
from collections import deque
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
from pydub import AudioSegment
from openai import OpenAI, NotFoundError
from ari_client import AriClient
from elastic_retriever import ElasticVectorRetriever

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ai_agent")

# ───────────── Defaults ───────────── #

DEFAULTS = {
    "openai": {
        "project": None,
        "organization": None
    },
    "greeting_text": "Καλημέρα! Πώς μπορώ να βοηθήσω;",
    "goodbye_text": "Ευχαριστώ για την επικοινωνία! Αντίο.",
    "agent": {
        "agent_id": None,
        "model": "gpt-4.1-mini",
        "instructions": (
            "Μίλα Ελληνικά, σύντομα και ξεκάθαρα. Απάντησε απευθείας στο ερώτημα. "
            "Μην χρησιμοποιείς SSML/markup. Απάντησε με σύντομες παραγράφους, χωρίς bullets.\n\n"
            "ΚΑΝΟΝΕΣ ΜΕΤΑΦΟΡΑΣ:\n"
            "- Χρησιμοποίησε το transfer_to_technician tool ΟΠΟΤΕ ο χρήστης ζητά ρητά ανθρώπινη βοήθεια\n"
            "- Χρησιμοποίησε το transfer_to_technician tool ΟΠΟΤΕ δεν μπορείς να λύσεις το πρόβλημα\n"
            "- Χρησιμοποίησε το transfer_to_technician tool ΟΠΠΟΤΕ ο χρήστης λέει 'τεχνικό', 'άνθρωπο', 'ανθρώπινη βοήθεια'\n\n"
            "Παραδείγματα που πρέπει να μεταφέρεις:\n"
            "- 'Θέλω να μιλήσω με τεχνικό' → TRANSFER\n"
            "- 'Θέλω να μιλήσω με εκπρόσωπο' → TRANSFER\n"
            "- 'Μπορώ να μιλήσω με άνθρωπο;' → TRANSFER\n"
            "- 'Δεν μπορείτε να με βοηθήσετε' → TRANSFER\n"
            "- 'Αυτό είναι πολύ περίπλοκο' → TRANSFER\n\n"
            "ΑΠΑΝΤΑ ΠΑΝΤΑ ΒΑΣΙΣΜΕΝΟΣ ΣΤΟ ΜΗΧΑΝΟΓΡΑΦΗΜΕΝΟ ΥΛΙΚΟ (file_search). "
            "Αν το file_search δεν επιστρέψει επαρκή στοιχεία, πες το ρητά και πρότεινε μεταφορά σε τεχνικό.\n\n"
            "Αν γνωρίζεις το επώνυμο του καλούντος, να τον/την προσφωνείς ως «Κύριε/Κυρία <Επώνυμο>». "
            "Αν δεν γνωρίζεις φύλο, χρησιμοποίησε το «Κ. <Επώνυμο>»."
        ),
        # ── Human acks + waiters on every turn ──
        "ack_enabled": True,
        "ack_exclamations": [
            "Μάλιστα.",
            "ΟΚ.",
            "Μμμ…",
            "Ωραία.",
            "Μισό λεπτό…"
        ],
        "ack_trailing_ms": 200,
        "wait_phrases": [
            "Μια στιγμή, ψάχνω τις πληροφορίες.",
            "Περιμένετε λίγο, επεξεργάζομαι το αίτημά σας.",
            "Ένα δευτερόλεπτο να το ελέγξω."
        ],
        "wait_first_delay_ms": 500,
        "wait_only_with_file_search": True,
        "waiter_trailing_ms": 600,
        "waiter_drain_pad_ms": 120,
        "wait_every_turn": True,
        "single_wait_phrase": False
    },
    "tts": {
        "model": "gpt-4o-mini-tts",
        "voice": "verse",
        "trailing_silence_ms": 300,
        "gain_db": 3.0,
        "timeout": 30.0,
        "max_text_length": 1000
    },
    "stt": {
        "primary_model": "gpt-4o-transcribe",
        "fallback_model": "whisper-1",
        "language": "el",
        "prompt": (
            "Ελληνικός προφορικός λόγος, τηλεφωνική κλήση. "
            "Γράψε ακριβώς ό,τι ακούς. "
            "Προσοχή σε παρόμοιες λέξεις: 'οθόνη' ≠ 'τόνοι', 'μαύρη' ≠ 'μαύροι'. "
            "Διάκρισε αρσενικά/θηλυκά/ουδέτερα γένη. "
            "Σωστή στίξη, τόνωση και γραμματική. "
            "Χρήστης μιλάει για τεχνικά προβλήματα, υπολογιστές, συσκευές."
        ),
        "temperature": 0.0,
        "context_aware": True,
        "common_corrections": {
            "τόνοι": "οθόνη",
            "μαύροι": "μαύρη",
            "μαύρες": "μαύρη",
        }
    },
    "dialog": {
        "max_turns": 0,
        "leading_silence_ms": 80,
        "goodbye_phrases": ["αντίο", "ευχαριστώ", "τέλος", "bye", "thank you", "thanks", "σταμάτα", "τέλεια"]
    },
    "transfer": {
        "enabled": True,
        "technician_extension": "7669",
        "transfer_prompt": "Θέλετε να σας μεταφέρω στον τεχνικό για περαιτέρω βοήθεια;",
        "transfer_confirmation": "Σας μεταφέρω στον τεχνικό. Παρακαλώ περιμένετε.",
        "transfer_timeout_sec": 30
    },
    "vad": {
        "frame_ms": 10,
        "start_hang_ms": 260,
        "end_silence_ms": 1100,
        "pause_grace_ms": 1400,
        "min_speech_ms": 700,
        "max_utterance_ms": 30000,
        "auto_calibrate_ms": 1200,
        "energy_factor": 1.8,
        "energy_floor": 0.02,
        "max_zcr": 0.35,
        "adaptive_silence": True,
        "speech_extension_ms": 2000,
        "min_speech_between_pauses": 500,
        "pre_roll_ms": 240
    },
    "barge_in": {
        "enabled": True,
        "pre_calibrate_ms": 400,
        "start_hang_ms": 180,
        "hold_ms": 160,
        "energy_factor": 3.0,
        "energy_floor": 0.03,
        "max_zcr": 0.30,
        "min_snr_db": 8.0,
        "echo_corr_thresh": 0.85,
        "frame_ms": 20
    },
    "retrieval": {
        "enabled": True,
        "mode": "openai_file_search",
        "vector_store_ids": [],
        "max_num_results": 8,
        "instructions": (
            "Χρησιμοποίησε κατά προτεραιότητα το συνδεδεμένο υλικό για να απαντήσεις. "
            "Αν δεν επαρκεί, πες το ρητά και ζήτα διευκρίνιση. "
            "Μην παραθέτεις παραπομπές/ονόματα αρχείων."
        )
    },
    "elasticsearch": {
        "enabled": False,
        "hosts": ["http://localhost:9200"],
        "username": None,
        "password": None,
        "api_key": None,
        "index": "voice_kb",
        "indexes": [],
        "vector_field": "embedding",
        "text_field": "content",
        "metadata_fields": [],
        "top_k": 5,
        "num_candidates": 64,
        "min_score": 0.0,
        "embedding_model": "text-embedding-3-small",
        "conversation_log_enabled": False,
        "request_timeout": 12.0,
        "verify_certs": True,
        "ca_certs": None
    },
    "external_media": {
        "bind_ip": "0.0.0.0",
        "port_min": 30000,
        "port_max": 40050
    },
    "ari": {
        "host": "http://127.0.0.1",
        "port": 8088,
        "username": "asterisk",
        "password": "asterisk",
        "app": "ai_agent",
        "tenant_id_var": "X-Tenant-Id"
    }
}

# ───────────── Globals ───────────── #

_active_keys = set()
_active_lock = threading.Lock()
_external_channel_ids = set()
_external_lock = threading.Lock()
_shutdown = threading.Event()

oa: Optional[OpenAI] = None
_tts_cache = {}
elastic_rag: Optional[ElasticVectorRetriever] = None

# Track active calls for transfer
_active_calls: Dict[str, Dict[str, Any]] = {}
_active_calls_lock = threading.Lock()

# ───────────── Call Transfer Manager ───────────── #

class CallTransferManager:
    """Διαχείριση μεταφοράς κλήσεων προς τεχνικό, με ασφαλές teardown του bridge
    και blind transfer μέσω καθαρών ARI HTTP κλήσεων (continue/redirect) και, αν χρειαστεί, CLI.
    Απαιτεί το πακέτο `requests` (pip install requests).
    """

    def __init__(self, ari_client: AriClient, cfg: dict):
        import requests  # local import για να μην απαιτείται αν δεν χρησιμοποιηθεί
        self.requests = requests
        self.session = requests.Session()

        self.ari = ari_client
        self.cfg = cfg

        self.technician_extension = cfg["transfer"].get("technician_extension", "7669")
        self.transfer_prompt = cfg["transfer"].get(
            "transfer_prompt",
            "Θέλετε να σας μεταφέρω στον τεχνικό για περαιτέρω βοήθεια;"
        )
        self.transfer_confirmation = cfg["transfer"].get(
            "transfer_confirmation",
            "Σας μεταφέρω στον τεχνικό. Παρακαλώ περιμένετε."
        )
        self.transfer_timeout = float(cfg["transfer"].get("transfer_timeout_sec", 30))

        # Ρυθμίσεις ARI για απευθείας HTTP
        raw_host = (cfg.get("ari", {}).get("host") or "http://127.0.0.1").strip().rstrip("/")
        if not raw_host.startswith(("http://", "https://")):
            raw_host = f"http://{raw_host}"
        port = int(cfg.get("ari", {}).get("port", 8088))
        self.base = f"{raw_host}:{port}"
        self.ari_base = f"{self.base}/ari"
        self.auth = (
            cfg.get("ari", {}).get("username", "asterisk"),
            cfg.get("ari", {}).get("password", "asterisk"),
        )
        self.verify_ssl = bool(cfg.get("ari", {}).get("verify_ssl", True))
        self.timeout = float(cfg.get("ari", {}).get("http_timeout_sec", 5.0))

    async def ask_for_transfer(
        self,
        channel_id: str,
        rtp_sock: socket.socket,
        rtp_addr: tuple,
        rtp_manager,
        context: dict
    ) -> bool:
        prompt = self.transfer_prompt
        log.info("Asking user for transfer confirmation: %s", prompt)

        tts_audio = await tts_wav16k_cached(prompt, self.cfg)
        tts8 = resample_pcm16(tts_audio, 16000, 8000)
        out_ulaw = pcm16_to_ulaw(tts8)

        bytes_per_packet = 160
        prompt_packets = (len(out_ulaw) + bytes_per_packet - 1) // bytes_per_packet
        prompt_seq, prompt_ts = await rtp_manager.get_next_sequence(prompt_packets)

        loop = asyncio.get_running_loop()
        started = time.monotonic()
        sent = 0
        for pkt in rtp_packetize_ulaw(out_ulaw, rtp_manager.ssrc, prompt_seq, prompt_ts, 8000, 20, 0):
            await loop.sock_sendto(rtp_sock, pkt, rtp_addr)
            sent += 1
            target = started + sent * 0.020
            rem = target - time.monotonic()
            if rem > 0:
                await asyncio.sleep(rem)

        log.info("Transfer prompt played, waiting for user response")

        vad_cfg = self.cfg["vad"]
        in_ulaw, _ = await capture_utterance_ulaw(rtp_sock, rtp_addr, vad_cfg, pt_expect=0)
        if not in_ulaw:
            log.info("No response to transfer prompt")
            return False

        pcm8_in = ulaw_to_pcm16(in_ulaw)
        pcm16_in = resample_pcm16(pcm8_in, 8000, 16000)
        user_response = transcribe_pcm16_16k_enhanced(
            pcm16_in, self.cfg, "απάντηση για μεταφορά κλήσης"
        )

        log.info("User transfer response: %s", user_response)

        affirmative_keywords = ["ναι", "ok", "οκ", "εντάξει", "παρακαλώ", "βέβαια", "ναι παρακαλώ", "προχώρα", "προχωράμε", "κάν'το"]
        is_affirmative = any(k in (user_response or "").lower() for k in affirmative_keywords)
        return bool(is_affirmative)

    def _post(self, path: str, params: dict = None, json: dict = None) -> bool:
        url = f"{self.ari_base}{path}"
        try:
            r = self.session.post(
                url,
                params=params or {},
                json=json,
                auth=self.auth,
                timeout=self.timeout,
                verify=self.verify_ssl
            )
            if 200 <= r.status_code < 300:
                return True
            log.error("ARI POST %s failed: %s %s", url, r.status_code, r.text[:300])
            return False
        except Exception as e:
            log.error("ARI POST %s exception: %s", url, e)
            return False

    def _exec_cli(self, command: str) -> bool:
        url = f"{self.ari_base}/asterisk/commands"
        payload = {"command": command}
        try:
            r = self.session.post(
                url,
                json=payload,
                auth=self.auth,
                timeout=self.timeout,
                verify=self.verify_ssl
            )
            if 200 <= r.status_code < 300:
                return True
            log.error("ARI CLI failed: %s %s", r.status_code, r.text[:300])
            return False
        except Exception as e:
            log.error("ARI CLI exception: %s", e)
            return False

    def _teardown_bridge(self, channel_id: str):
        try:
            info = None
            with _active_calls_lock:
                info = _active_calls.get(channel_id)
            bid = (info or {}).get("bridge_id")
            ext = (info or {}).get("external_id")
            if not bid:
                return
            try:
                if ext:
                    self.ari.bridges_remove_channel(bid, ext)
            except Exception:
                pass
            try:
                self.ari.bridges_remove_channel(bid, channel_id)
            except Exception:
                pass
            try:
                self.ari.bridges_destroy(bid)
            except Exception:
                pass
        except Exception as e:
            log.warning("Bridge teardown issue: %s", e)

    def transfer_call(self, channel_id: str) -> bool:
        try:
            self._teardown_bridge(channel_id)

            ctx = "from-internal"
            ex = self.technician_extension
            prio = 1

            ok = self._post(
                f"/channels/{channel_id}/continue",
                params={"context": ctx, "extension": ex, "priority": prio}
            )
            if ok:
                log.info("channels/continue issued for %s → %s/%s/%d", channel_id, ctx, ex, prio)
                return True

            endpoint = f"Local/{ex}@{ctx}"
            ok = self._post(
                f"/channels/{channel_id}/redirect",
                params={"endpoint": endpoint}
            )
            if ok:
                log.info("channels/redirect issued for %s → %s", channel_id, endpoint)
                return True

            cmd = f"channel redirect {channel_id} {ctx} {ex} {prio}"
            if self._exec_cli(cmd):
                log.info("CLI blind transfer executed: %s", cmd)
                return True

            log.error("Transfer failed on all methods for channel %s", channel_id)
            return False

        except Exception as e:
            log.exception("Error during call transfer: %s", e)
            return False

# ───────────── Greek Language Processing ───────────── #

GREEK_CORRECTIONS = {
    "τόνοι": "οθόνη",
    "μαύροι": "μαύρη",
    "μαύρες": "μαύρη",
    "πρόγραμμα": "πρόγραμμα",
    "προγράμματα": "προγράμματα",
    "συσκευή": "συσκευή",
    "συσκευές": "συσκευές",
}

GREEK_PUNCTUATION_RULES = [
    (re.compile(r'\b(πως|πώς|γιατί|πότε|πού)\b.*[^.;!?]$', re.IGNORECASE), lambda m: m.group(0) + ';'),
    (re.compile(r'\b(εντάξει|οκ|ok|καλά|τέλεια)\b[^.!?]*$', re.IGNORECASE), lambda m: m.group(0) + '.'),
]

_CITATION_PATTERNS = [
    (re.compile(r'【[^】]*】'), ''),
    (re.compile(r'\[\s*\d+(?:\s*(?:[-–,]\s*\d+))*\s*\]'), ''),
    (re.compile(r'\(?\s*(?:source|sources|πηγή|πηγές)\s*:\s*[^)\n]+?\)?(?=$|\n)', re.IGNORECASE), ''),
    (re.compile(r'\s*(?:source|sources|πηγή|πηγές)\s*:\*\s*\S+', re.IGNORECASE), ''),
    (re.compile(r'\b\w+\.(pdf|doc|docx|xls|xlsx|txt)\b', re.IGNORECASE), ''),
    (re.compile(r'【.*?】'), ''),
    (re.compile(r'\[.*?\]'), ''),
]

def strip_citations(text: str) -> str:
    if not text:
        return text
    cleaned = text
    for pattern, replacement in _CITATION_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def clamp_response_length(text: str, cfg: dict) -> str:
    if not text:
        return text
    limit = int(cfg.get("agent", {}).get("max_response_chars", 650) or 0)
    if limit <= 0 or len(text) <= limit:
        return text
    sentences = re.split(r"(?<=[.!;?])\s+", text)
    out = []
    total = 0
    for sent in sentences:
        clean = sent.strip()
        if not clean:
            continue
        candidate_len = total + (1 if out else 0) + len(clean)
        if candidate_len > limit:
            break
        out.append(clean)
        total = candidate_len
    if out:
        return " ".join(out)
    return text[:limit].rsplit(" ", 1)[0] if " " in text[:limit] else text[:limit]


async def log_conversation_event(
    cfg: dict,
    speaker: str,
    text: Optional[str],
    conversation_id: Optional[str],
    turn_number: int,
    tenant_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None
):
    if not text:
        return
    if not cfg.get("elasticsearch", {}).get("conversation_log_enabled"):
        return
    global elastic_rag
    if not elastic_rag or not hasattr(elastic_rag, "log_conversation_turn"):
        return
    doc = {
        "speaker": speaker,
        "text": text,
        "conversation_id": conversation_id,
        "channel_id": conversation_id,
        "turn": turn_number,
        "tenant_id": tenant_id,
        "agent_id": cfg.get("agent", {}).get("agent_id"),
        "app": cfg.get("ari", {}).get("app")
    }
    if extra:
        doc.update({k: v for k, v in extra.items() if v is not None})
    try:
        await asyncio.to_thread(elastic_rag.log_conversation_turn, doc)
    except Exception as exc:
        log.error("Conversation logging failed: %s", exc)

def apply_greek_corrections(text: str, cfg: dict) -> str:
    if not text:
        return text
    
    custom_corrections = cfg.get("stt", {}).get("common_corrections", {})
    all_corrections = {**GREEK_CORRECTIONS, **custom_corrections}
    
    words = text.split()
    corrected_words = []
    
    for word in words:
        clean_word = re.sub(r'[.,;!?]', '', word.lower())
        
        if clean_word in all_corrections:
            corrected = all_corrections[clean_word]
            if word.istitle():
                corrected = corrected.capitalize()
            elif word.isupper():
                corrected = corrected.upper()
            if len(word) > 0 and word[-1] in '.,;!?':
                corrected += word[-1]
            corrected_words.append(corrected)
        else:
            corrected_words.append(word)
    
    result = ' '.join(corrected_words)
    
    if cfg.get("stt", {}).get("context_aware", True):
        for pattern, replacement in GREEK_PUNCTUATION_RULES:
            if pattern.search(result) and not result.endswith(('.', ';', '!', '?')):
                result = replacement(pattern.search(result))
    
    return result

def enhance_stt_prompt(cfg: dict, context: str = "") -> str:
    base_prompt = cfg["stt"].get("prompt", DEFAULTS["stt"]["prompt"])
    if context:
        enhanced_prompt = f"{base_prompt} Συζήτηση για: {context}"
    else:
        enhanced_prompt = base_prompt
    return enhanced_prompt.strip()

# ── Name extraction + gender/salutation helpers ── #

_GREEK_NAME_RE = re.compile(
    r"(?:\bμε\s+λένε\b|\bονομάζομαι\b|\bείμαι\b)\s+([A-Za-zΑ-Ω Ά-ΏΪΫά-ώϊϋΐΰ]+)(?:\s+([A-Za-zΑ-Ω Ά-ΏΪΫά-ώϊϋΐΰ]+))?",
    re.IGNORECASE
)

def extract_last_name_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    m = _GREEK_NAME_RE.search(text)
    if not m:
        return None
    first = (m.group(1) or "").strip()
    last = (m.group(2) or "").strip()
    if last and len(last) >= 2:
        return last[:1].upper() + last[1:].lower()
    return None

def detect_gender_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip().lower()

    if re.search(r'\bείμαι\s+ο\b', t) or re.search(r'\bκύριος\b', t):
        return "male"
    if re.search(r'\bείμαι\s+η\b', t) or re.search(r'\bκυρία\b', t):
        return "female"

    m = _GREEK_NAME_RE.search(t)
    if m:
        first = (m.group(1) or "").strip()
        if len(first) >= 3:
            if re.search(r'(α|η|ω)$', first):
                return "female"
            if re.search(r'(ος|ης|ας|άς)$', first):
                return "male"

    return None

def salutation_prefix(caller_profile: Dict[str, Optional[str]]) -> Optional[str]:
    last = caller_profile.get("last_name")
    if not last:
        return None
    gender = caller_profile.get("gender")
    if gender == "male":
        return f"Κύριε {last}"
    if gender == "female":
        return f"Κυρία {last}"
    return f"Κ. {last}"

def apply_salutation(text: str, caller_profile: Dict[str, Optional[str]]) -> str:
    last = caller_profile.get("last_name")
    if not last:
        return text or ""
    t = text or ""
    if len(t.strip()) <= 3:
        return t
    prefix = salutation_prefix(caller_profile)
    if not prefix:
        return t
    if t.strip().lower().startswith(prefix.lower()):
        return t
    return f"{prefix}, {t}"

# ───────────── Helpers ───────────── #

_GOODBYE_RE = re.compile(r'(?:^|\b)(αντίο|τέλος|σταμάτα|bye|thank you|thanks)\b[.!;…]*$', re.IGNORECASE)

# (μόνο το σχετικό τμήμα - το υπόλοιπο αρχείο σου μένει απαράλλακτο)

_GOODBYE_RE = re.compile(
    r'(?:^|\b)(αντίο|τέλος|σταμάτα|bye|thank you|thanks|ευχαριστώ)\b[.!;…]*$',
    re.IGNORECASE
)

def is_goodbye_message(text: str, cfg: dict) -> bool:
    """
    Ανίχνευση φράσεων που δηλώνουν τερματισμό συνομιλίας.
    Περιλαμβάνει 'όχι', 'όχι ευχαριστώ', 'δεν χρειάζεται', 'όλα καλά', 'τίποτα άλλο' κ.ά.
    """
    if not text:
        return False
    t = text.strip().lower()

    # Βασικές φράσεις τερματισμού
    if _GOODBYE_RE.search(t):
        return True

    # Μονές αρνήσεις όπως «όχι.»
    normalized = re.sub(r"[\s.,!;]+$", "", t)
    if normalized in {"όχι", "οχι"}:
        return True

    # Επεκτάσεις για αρνήσεις και φυσικό τέλος
    end_patterns = [
        r"όχι\s*ευχαριστώ",
        r"όχι\s+όλα\s+καλά",
        r"δεν\s+χρειάζεται",
        r"όλα\s+καλά",
        r"τίποτα\s+άλλο",
        r"όχι\s+τίποτα",
        r"δε\s+χρειάζετ",
        r"είμαι\s+εντάξει",
        r"όχι\s+δε\s+χρειάζεται",
        r"όχι\s+δε\s+θέλω",
        r"εντάξει\s+ευχαριστώ",
        r"ευχαριστώ\s+όχι"
    ]
    for pat in end_patterns:
        if re.search(pat, t, re.IGNORECASE):
            return True

    return False



def get_tenant_id_from_evt(evt: dict, cfg: dict) -> Optional[str]:
    try:
        ch = evt.get("channel") or {}
        vars_ = (ch.get("channelvars") or {}) if isinstance(ch.get("channelvars"), dict) else {}
        key = cfg["ari"].get("tenant_id_var") or "X-Tenant-Id"
        tid = vars_.get(key)
        if tid:
            return str(tid)
        linked = ch.get("linkedid") or ""
        if isinstance(linked, str) and linked.startswith("tenant:"):
            return linked.split(":", 1)[-1]
    except Exception:
        pass
    return None

def build_openai_client(cfg: dict) -> OpenAI:
    params: Dict[str, Any] = {}
    op = cfg.get("openai", {})
    if op.get("project"):
        params["project"] = op["project"]
    if op.get("organization"):
        params["organization"] = op["organization"]
    client = OpenAI(**params)
    return client

def validate_vector_stores(client: OpenAI, ids: List[str]) -> List[str]:
    valid = []
    for vid in ids or []:
        try:
            client.vector_stores.retrieve(vid)
            valid.append(vid)
        except Exception as e:
            log.error("Vector store %s not accessible: %s", vid, e)
    return valid

# ───────────── Optimized TTS Functions ───────────── #

async def tts_wav16k_async(text: str, cfg: dict) -> bytes:
    start_time = time.time()
    
    tts_model = cfg["tts"]["model"]
    voice = cfg["tts"]["voice"]
    trailing_ms = int(cfg["tts"]["trailing_silence_ms"])
    gain_db = float(cfg["tts"]["gain_db"])
    timeout = float(cfg["tts"].get("timeout", 30.0))
    max_length = int(cfg["tts"].get("max_text_length", 1000))
    
    safe_text = (text or "").strip() or "..."
    if len(safe_text) > max_length:
        log.warning(f"Truncating TTS text from {len(safe_text)} to {max_length} characters")
        safe_text = safe_text[:max_length] + "..."
    
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                oa.audio.speech.create,
                model=tts_model,
                voice=voice,
                input=safe_text,
                response_format="wav"
            ),
            timeout=timeout
        )
        
        tts_api_time = time.time() - start_time
        log.info(f"TTS API call completed in {tts_api_time:.2f}s")
        
        blob = response.read() if hasattr(response, "read") else bytes(response)
        if not isinstance(blob, (bytes, bytearray)):
            blob = bytes(blob)
        
        if len(blob) < 44 or blob[:4] != b"RIFF":
            seg = AudioSegment.from_file(io.BytesIO(blob)).set_channels(1).set_frame_rate(16000).set_sample_width(2)
            out = io.BytesIO()
            seg.export(out, format="wav")
            blob = out.getvalue()
        
        pcm16, sr = wav_bytes_to_pcm16_mono(blob)
        if sr != 16000:
            pcm16 = resample_pcm16(pcm16, sr, 16000)
        if abs(gain_db) > 0.1:
            pcm16 = apply_gain_db(pcm16, 16000, gain_db)
        if trailing_ms > 0:
            pad = np.zeros(int(16000 * trailing_ms / 1000), dtype=np.int16).tobytes()
            pcm16 += pad
        
        total_time = time.time() - start_time
        log.info(f"TTS total processing completed in {total_time:.2f}s")
        return pcm16
    
    except asyncio.TimeoutError:
        log.error(f"TTS timeout after {timeout}s, returning empty audio")
        return b""
    except Exception as e:
        log.error(f"TTS failed: {e}, returning empty audio")
        return b""

async def tts_wav16k_cached(text: str, cfg: dict) -> bytes:
    cache_key = hash(text[:200])
    
    if cache_key in _tts_cache:
        log.info("TTS cache hit")
        return _tts_cache[cache_key]
    
    audio = await tts_wav16k_async(text, cfg)
    if audio:
        _tts_cache[cache_key] = audio
    return audio

# ───────────── Enhanced STT Function ───────────── #

def transcribe_pcm16_16k_enhanced(pcm16_16k: bytes, cfg: dict, context: str = "") -> str:
    language = cfg["stt"]["language"]
    primary = cfg["stt"]["primary_model"]
    fallback = cfg["stt"]["fallback_model"]
    stt_temp = float(cfg["stt"].get("temperature", 0.0))
    
    enhanced_prompt = enhance_stt_prompt(cfg, context)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tf:
        with wave.open(tf.name, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm16_16k)
        
        try:
            log.info("Using primary STT model: %s", primary)
            r = oa.audio.transcriptions.create(
                model=primary, 
                file=open(tf.name, "rb"),
                language=language, 
                prompt=enhanced_prompt, 
                temperature=stt_temp
            )
            raw_text = (getattr(r, "text", "") or str(r)).strip()
            log.info("STT raw output: %s", raw_text)
        
        except Exception as e1:
            log.warning("Primary STT failed (%s), fallback to %s", e1, fallback)
            try:
                r = oa.audio.transcriptions.create(
                    model=fallback, 
                    file=open(tf.name, "rb"),
                    language=language, 
                    prompt=enhanced_prompt, 
                    temperature=stt_temp
                )
                raw_text = (getattr(r, "text", "") or str(r)).strip()
                log.info("STT fallback output: %s", raw_text)
            except Exception as e2:
                log.exception("Fallback STT failed: %s", e2)
                return ""

    if not raw_text:
        return ""

    corrected_text = apply_greek_corrections(raw_text, cfg)
    
    if corrected_text != raw_text:
        log.info("STT corrected: '%s' → '%s'", raw_text, corrected_text)
    
    return corrected_text

# ───────────── RTP Sequence Management ───────────── #

class RTPStateManager:
    def __init__(self, initial_seq: int, initial_ts: int, ssrc: int):
        self.seq = initial_seq
        self.ts = initial_ts
        self.ssrc = ssrc
        self.lock = asyncio.Lock()
    
    async def get_next_sequence(self, packet_count: int) -> Tuple[int, int]:
        async with self.lock:
            current_seq = self.seq
            current_ts = self.ts
            self.seq = (self.seq + packet_count) & 0xFFFF
            self.ts = (self.ts + packet_count * 160) & 0xFFFFFFFF
            return current_seq, current_ts
    
    async def advance_sequence(self, packet_count: int):
        async with self.lock:
            self.seq = (self.seq + packet_count) & 0xFFFF
            self.ts = (self.ts + packet_count * 160) & 0xFFFFFFFF

# ───────────── FIXED Chat with Retrieval-First + Human Acks/Waiter per Turn ───────────── #

# ───────────── FIXED Chat with Retrieval-First + Human Acks/Waiter per Turn ───────────── #

# Επέκταση ανίχνευσης αιτημάτων για ανθρώπινη βοήθεια ή μεταφορά σε τεχνικό
# ───────────── Ανίχνευση αιτημάτων μεταφοράς σε τεχνικό ───────────── #

_TECH_REGEX = re.compile(
    r"(?:"
    r"θέλω\s+να\s+μιλήσω\s+(?:με\s+)?(τεχνικ(?:ό|ό|ό|ος)?|ανθρώπ(?:ο|ινο)?|εκπροσώπ(?:ο|η)?|αντιπρόσωπ(?:ο|η)?)"
    r"|μπορώ\s+να\s+μιλήσω\s+(?:με\s+)?(τεχνικ(?:ό|ό|ό|ος)?|ανθρώπ(?:ο|ινο)?|εκπροσώπ(?:ο|η)?|αντιπρόσωπ(?:ο|η)?)"
    r"|να\s+μιλήσω\s+(?:με\s+)?(τεχνικ(?:ό|ό|ό|ος)?|ανθρώπ(?:ο|ινο)?|εκπροσώπ(?:ο|η)?|αντιπρόσωπ(?:ο|η)?)"
    r"|μίλησέ\s+μου\s+(?:με|σε)\s+(τεχνικ|ανθρώπ|εκπροσώπ|αντιπροσώπ)"
    r"|συνδέσ(?:έ|ε|ω|εις|ει|ουμε|ετε|ουν)\s+με\s+(?:κάποιον\s+)?(τεχνικ|ανθρώπ|εκπροσώπ|αντιπροσώπ)"
    r"|θέλω\s+να\s+με\s+μεταφέρεις\s+(?:στον|σε\s+έναν)?\s*(τεχνικ|ανθρώπ|εκπροσώπ|αντιπροσώπ)"
    r"|μεταφέρ(?:ε|σέ)\s+με"
    r"|με\s+τον\s+τεχνικ(?:ό|ό|ό|ό|ό)"
    r"|θέλω\s+άνθρωπο"
    r"|θέλω\s+να\s+μιλήσω\s+με\s+κάποιον\s+άνθρωπο"
    r"|δεν\s+μπορείς\s+να\s+με\s+βοηθή"
    r"|δεν\s+μπορείτε\s+να\s+με\s+βοηθή"
    r"|είναι\s+πολύ\s+περίπλοκο"
    r"|είναι\s+δύσκολο\s+να\s+το\s+κάνω"
    r"|μιλήσω\s+(?:σε|με)\s+(τεχνικ|ανθρώπ|εκπροσώπ|αντιπροσώπ)"
    r")",
    re.IGNORECASE
)



# Detect assistant offers for transfer in previous turn
_TRANSFER_OFFER_RE = re.compile(
    r"(?:να\s+σας\s+(?:μεταφ(?:έρ|ε)ρω|συνδέσω)\s+.*τεχνικ)|(?:μεταφορά\s+σε\s+τεχνικ)|(?:συνδέσω\s+με\s+τεχνικ)|(?:να\s+προχωρήσω.*μεταφορά)",
    re.IGNORECASE
)

def _looks_like_no_answer(txt: str) -> bool:
    if not txt:
        return True
    low = txt.strip().lower()
    patterns = [
        "δεν εντόπισα", "δεν βρήκα", "δεν μπορώ να βρω", "δεν υπάρχουν επαρκείς πληροφορίες",
        "δεν έχω αρκετές πληροφορίες", "δεν είμαι σίγ", "no relevant information",
        "i don't have enough", "i cannot find", "insufficient"
    ]
    return any(p in low for p in patterns) or len(low) < 8

def _truncate_history(history: List[Dict[str, str]], max_pairs: int = 3) -> List[Dict[str, str]]:
    if not history:
        return []
    return history[-(max_pairs*2):]

def _is_affirmative_short(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    normalized = re.sub(r"[.!?,;]+", " ", t)
    tokens = [tok for tok in normalized.split() if tok]
    single_words = {
        "ναι", "οκ", "ok", "εντάξει", "παρακαλώ", "βεβαίως", "βέβαια"
    }
    phrases = [
        "ναι παρακαλώ",
        "ναι ευχαριστώ",
        "προχώρα",
        "προχωράμε",
        "κάν'το",
        "προχώρησε",
        "συνέχισε"
    ]
    if any(tok in single_words for tok in tokens):
        return True
    norm_joined = " ".join(tokens)
    return any(phrase in norm_joined for phrase in phrases)

async def chat_with_model(
    cfg: dict,
    user_text: str,
    channel_id: str,
    tenant_id: Optional[str],
    transfer_manager: CallTransferManager,
    rtp_sock: socket.socket,
    rtp_addr: tuple,
    rtp_manager: RTPStateManager,
    waiter_used_flag: Dict[str, bool],
    history: List[Dict[str, str]],
    caller_profile: Dict[str, Optional[str]],
    dialog_state: Dict[str, Any]
) -> Tuple[str, bool, bytes, bool, bool]:
    """
    Returns: (response_text, should_end, tts_audio, did_transfer, offered_transfer_this_turn)
    """
    global elastic_rag
    u = (user_text or "").strip()
    log.info("AGENT → user_text: %s", u)
    
    # Κανονικοποίηση για regex (καθαρισμός τόνων και στίξης)
    normalized_u = re.sub(r"[.,;!?]", "", u.lower().strip())


    # ── NEW: If the previous assistant turn offered transfer and user now answers affirmatively, transfer immediately ──
    pending_offer = dialog_state.get("transfer_offer_pending") or {}
    if pending_offer and _is_affirmative_short(u):
        log.info("Affirmative reply to previous transfer offer detected → proceeding to transfer without re-asking.")
        transfer_msg = cfg["transfer"].get("transfer_confirmation", "Σας μεταφέρω στον τεχνικό. Παρακαλώ περιμένετε.")
        transfer_msg = apply_salutation(transfer_msg, caller_profile)
        tts_audio = await tts_wav16k_cached(transfer_msg, cfg)
        success = transfer_manager.transfer_call(channel_id)
        dialog_state["transfer_offer_pending"] = False
        if success:
            log.info("Call transfer initiated successfully (affirmative follow-up).")
            return transfer_msg, True, tts_audio, True, False
        else:
            log.error("Call transfer failed after affirmative follow-up.")
            fallback = "Δυστυχώς η μεταφορά απέτυχε. Μπορούμε να συνεχίσουμε εδώ ή να δοκιμάσω ξανά;"
            fallback = apply_salutation(fallback, caller_profile)
            tts_audio = await tts_wav16k_cached(fallback, cfg)
            return fallback, False, tts_audio, False, False

    # ── Immediate human "ack/exclamation" before reasoning ──
    async def _play_ack_if_enabled():
        try:
            if not cfg["agent"].get("ack_enabled", True):
                return
            excls = list(cfg["agent"].get("ack_exclamations") or [])
            if not excls:
                return
            phrase = random.choice(excls).strip()
            log.info("ACK → %s", phrase)
            w16 = await tts_wav16k_cached(phrase, cfg)
            w8 = resample_pcm16(w16, 16000, 8000)
            ul = pcm16_to_ulaw(w8)
            trail_ms = int(cfg["agent"].get("ack_trailing_ms", 200))
            if trail_ms > 0:
                ul += b'\xff' * int(8000 * trail_ms / 1000)
            packets = (len(ul) + 159) // 160
            seq, ts = await rtp_manager.get_next_sequence(packets)
            loop = asyncio.get_running_loop()
            started = time.monotonic(); sent = 0
            for pkt in rtp_packetize_ulaw(ul, rtp_manager.ssrc, seq, ts, 8000, 20, 0):
                await loop.sock_sendto(rtp_sock, pkt, rtp_addr)
                sent += 1
                target = started + sent * 0.020
                rem = target - time.monotonic()
                if rem > 0:
                    await asyncio.sleep(rem)
        except Exception as e:
            log.error("ACK play failed: %s", e)

    # ── Waiter per turn (not single-use) ──
    async def _play_waiter_until_stopped(stop_event: asyncio.Event):
        try:
            first_delay = int(cfg["agent"].get("wait_first_delay_ms", 500)) / 1000.0
            await asyncio.sleep(max(0.0, first_delay))
            if stop_event.is_set():
                return
            phrases = list(cfg["agent"].get("wait_phrases") or [])
            if not phrases:
                return
            phrase = random.choice(phrases).strip()
            log.info("WAITER → %s", phrase)
            w16 = await tts_wav16k_cached(phrase, cfg)
            w8 = resample_pcm16(w16, 16000, 8000)
            ul = pcm16_to_ulaw(w8)
            waiter_pad_ms = int(cfg["agent"].get("waiter_trailing_ms", 600))
            if waiter_pad_ms > 0:
                ul += b'\xff' * int(8000 * waiter_pad_ms / 1000)
            packet_count = (len(ul) + 159) // 160
            waiter_seq, waiter_ts = await rtp_manager.get_next_sequence(packet_count)
            loop = asyncio.get_running_loop()
            started = time.monotonic(); sent = 0
            for pkt in rtp_packetize_ulaw(ul, rtp_manager.ssrc, waiter_seq, waiter_ts, 8000, 20, 0):
                if stop_event.is_set():
                    break
                await loop.sock_sendto(rtp_sock, pkt, rtp_addr)
                sent += 1
                target = started + sent * 0.020
                rem = target - time.monotonic()
                if rem > 0:
                    await asyncio.sleep(rem)
        except asyncio.CancelledError:
            log.info("Waiter cancelled")
        except Exception as e:
            log.error("Waiter error: %s", e)

    # Αν ο χρήστης αναφέρει τεχνικό ρητά, ακολουθούμε το υπάρχον flow επιβεβαίωσης
    if _TECH_REGEX.search(normalized_u):
        log.info("Heuristic trigger: user explicitly asked for technician → transfer flow")
        user_confirmed = await transfer_manager.ask_for_transfer(
            channel_id, rtp_sock, rtp_addr, rtp_manager, {"user_text": u}
        )
        if user_confirmed:
            transfer_msg = cfg["transfer"].get("transfer_confirmation", "Σας μεταφέρω στον τεχνικό. Παρακαλώ περιμένετε.")
            transfer_msg = apply_salutation(transfer_msg, caller_profile)
            tts_audio = await tts_wav16k_cached(transfer_msg, cfg)
            success = transfer_manager.transfer_call(channel_id)
            if success:
                return transfer_msg, True, tts_audio, True, False
            else:
                fallback = "Η μεταφορά απέτυχε. Θέλετε να δοκιμάσω ξανά ή να συνεχίσουμε εδώ;"
                fallback = apply_salutation(fallback, caller_profile)
                tts_audio = await tts_wav16k_cached(fallback, cfg)
                return fallback, False, tts_audio, False, False
        else:
            reply = "Εντάξει, ας συνεχίσουμε εδώ. Πώς μπορώ να βοηθήσω;"
            reply = apply_salutation(reply, caller_profile)
            tts_audio = await tts_wav16k_cached(reply, cfg)
            return reply, False, tts_audio, False, False

    if is_goodbye_message(u, cfg):
        goodbye_text = cfg.get("goodbye_text") or DEFAULTS["goodbye_text"]
        goodbye_text = apply_salutation(goodbye_text, caller_profile)
        tts_audio = await tts_wav16k_cached(goodbye_text, cfg)
        return goodbye_text, True, tts_audio, False, False

    model = cfg["agent"].get("model", DEFAULTS["agent"]["model"])
    base_instructions = cfg["agent"].get("instructions", DEFAULTS["agent"]["instructions"])
    agent_id = cfg["agent"].get("agent_id")
    system_prefix = f"[agent_id={agent_id}] " if agent_id else ""

    retrieval_conf = cfg.get("retrieval", {})
    retrieval_mode = retrieval_conf.get("mode")
    vs_ids: List[str] = retrieval_conf.get("_validated_vector_store_ids") or []
    file_search_enabled = bool(retrieval_mode == "openai_file_search" and retrieval_conf.get("enabled") and vs_ids)
    elastic_cfg = cfg.get("elasticsearch", {})
    elastic_enabled = bool(retrieval_mode == "elastic" and elastic_cfg.get("enabled") and elastic_rag)
    rag_active = file_search_enabled or elastic_enabled

    # ── Pre-fetch Elastic context concurrently ──
    elastic_results_task: Optional[asyncio.Task] = None
    if elastic_enabled and elastic_rag:
        async def _fetch_elastic_context():
            return await asyncio.to_thread(elastic_rag.search, u)
        elastic_results_task = asyncio.create_task(_fetch_elastic_context())

    # ── Start ACK immediately ──
    await _play_ack_if_enabled()

    retrieval_rules = retrieval_conf.get("instructions", DEFAULTS["retrieval"]["instructions"])

    if file_search_enabled:
        force_retrieval_clause = (
            "ΧΡΗΣΙΜΟΠΟΙΗΣΕ ΥΠΟΧΡΕΩΤΙΚΑ file_search για να απαντήσεις. "
            "Αν δεν βρεις σχετικό περιεχόμενο, δήλωσε ρητά ότι δεν βρέθηκαν επαρκείς πληροφορίες."
        )
    elif elastic_enabled:
        force_retrieval_clause = (
            "Χρησιμοποίησε αποκλειστικά τα αποσπάσματα που παρέχονται από την Elasticsearch βάση γνώσης. "
            "Αν δεν επαρκούν, ενημέρωσε τον χρήστη και πρότεινε μεταφορά σε τεχνικό."
        )
    else:
        force_retrieval_clause = ""

    last_name_hint = ""
    if caller_profile.get("last_name"):
        form = salutation_prefix(caller_profile) or f"Κ. {caller_profile['last_name']}"
        last_name_hint = f"Ο/Η καλών/καλούσα έχει δηλώσει επώνυμο και επιθυμητή προσφώνηση: «{form}»."

    sys_parts = [f"{system_prefix}{base_instructions}", retrieval_rules, force_retrieval_clause, last_name_hint]
    system_instruction = "\n\n".join([part.strip() for part in sys_parts if part]).strip()

    tools: List[Dict[str, Any]] = []

    transfer_tool = {
        "type": "function",
        "name": "transfer_to_technician",
        "description": "Transfer the call to a human technician when you cannot help the user or when they request human assistance. Use this when user asks for human help, technician, or when the problem is too complex.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "The reason for transferring the call"}
            },
            "required": ["reason"]
        }
    }
    tools.append(transfer_tool)

    if file_search_enabled:
        tools.append({
            "type": "file_search",
            "vector_store_ids": vs_ids,
            "max_num_results": int(cfg["retrieval"].get("max_num_results", 8))
        })

    # ── Start waiter per-turn if applicable ──
    turn_stop_event = asyncio.Event()
    waiter_task = None
    start_wait = cfg["agent"].get("wait_every_turn", True)
    if start_wait and (rag_active or not cfg["agent"].get("wait_only_with_file_search", True)):
        waiter_task = asyncio.create_task(_play_waiter_until_stopped(turn_stop_event))
        log.info("Started per-turn waiter")
    else:
        log.info("Per-turn waiter skipped by config")

    def _call_responses_blocking(kwargs: dict):
        try:
            return oa.responses.create(**kwargs)
        except NotFoundError as e:
            msg = str(e)
            if "Vector store" in msg and "not found" in msg:
                log.error("Vector store not found during chat; retrying without file_search.")
                kwargs = dict(kwargs)
                kwargs.pop("tools", None)
                return oa.responses.create(**kwargs)
            raise

    trimmed_history = _truncate_history(history, max_pairs=3)
    base_input: List[Dict[str, Any]] = [{"role": "system", "content": system_instruction}]
    for h in trimmed_history:
        role = h.get("role") or "user"
        content = h.get("content") or ""
        if content:
            base_input.append({"role": role, "content": content})

    elastic_hits_found = False
    elastic_results: List[Dict[str, Any]] = []
    if elastic_results_task:
        try:
            elastic_results = await elastic_results_task
        except Exception as exc:
            log.error("Elastic retrieval task failed: %s", exc)
            elastic_results = []

    if elastic_results:
        elastic_hits_found = True
        if elastic_hits_found:
            context_payload = ElasticVectorRetriever.format_results(elastic_results)
            log.info("Elastic retrieval returned %d hits", len(elastic_results))
            base_input.append({
                "role": "assistant",
                "content": "[Βάση γνώσης]\n" + context_payload
            })
    elif elastic_enabled and elastic_rag:
        log.info("Elastic retrieval returned no matches")
        base_input.append({
            "role": "system",
            "content": (
                "Η αναζήτηση στην Elasticsearch βάση δεν επέστρεψε σχετικά στοιχεία. "
                "Ενημέρωσε τον χρήστη ότι δεν υπάρχουν διαθέσιμες πληροφορίες και πρότεινε μεταφορά σε τεχνικό όταν χρειάζεται."
            )
        })

    # track turn counter for dialog diagnostics
    dialog_state["turn_counter"] = dialog_state.get("turn_counter", 0) + 1

    base_input.append({"role": "user", "content": u or "Απάντησε σύντομα και καθαρά."})

    agent_temp = float(cfg["agent"].get("temperature", 0.2))
    agent_max_tokens = int(cfg["agent"].get("max_output_tokens", 0) or 0)

    kwargs = {
        "model": model,
        "input": base_input,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": agent_temp
    }
    if agent_max_tokens > 0:
        kwargs["max_output_tokens"] = agent_max_tokens

    offered_transfer_this_turn = False

    try:
        chat_start = time.time()
        resp = await asyncio.to_thread(_call_responses_blocking, kwargs)
        chat_time = time.time() - chat_start
        log.info(f"Chat API completed in {chat_time:.2f}s")
        
        should_transfer = False
        response_text = ""

        try:
            if hasattr(resp, 'output') and resp.output:
                for item in resp.output:
                    if getattr(item, 'type', None) == 'tool_calls' and getattr(item, 'tool_calls', None):
                        for tool_call in item.tool_calls:
                            if getattr(getattr(tool_call, 'function', None), 'name', '') == 'transfer_to_technician':
                                log.info("✅ AI requested call transfer to technician")
                                should_transfer = True
                    if getattr(item, 'type', None) == 'message' and getattr(item, 'content', None):
                        for content_item in item.content:
                            if getattr(content_item, 'type', '') == 'output_text' and hasattr(content_item, 'text'):
                                response_text += content_item.text or ""
        except Exception as e:
            log.error(f"Error processing response output: {e}")

        if not should_transfer and hasattr(resp, 'tool_calls') and resp.tool_calls:
            for tool_call in resp.tool_calls:
                if getattr(getattr(tool_call, 'function', None), 'name', '') == 'transfer_to_technician':
                    log.info("✅ AI requested call transfer to technician")
                    should_transfer = True

        if not response_text and hasattr(resp, 'output_text') and resp.output_text:
            response_text = resp.output_text

        if not response_text:
            response_text = str(resp)

        response_text = strip_citations((response_text or "").strip())
        log.info("AGENT ← response: %s", response_text)
        log.info("Transfer requested: %s", should_transfer)

        # Stop per-turn waiter immediately
        turn_stop_event.set()
        if waiter_task and not waiter_task.done():
            waiter_task.cancel()
            try:
                await waiter_task
            except asyncio.CancelledError:
                pass

        if should_transfer and channel_id:
            log.info("AI requested transfer, asking user for confirmation")
            transfer_prompt_ok = await transfer_manager.ask_for_transfer(
                channel_id, rtp_sock, rtp_addr, rtp_manager, {"user_text": u}
            )
            if transfer_prompt_ok:
                transfer_msg = cfg["transfer"].get("transfer_confirmation", "Σας μεταφέρω στον τεχνικό. Παρακαλώ περιμένετε.")
                transfer_msg = apply_salutation(transfer_msg, caller_profile)
                tts_audio = await tts_wav16k_cached(transfer_msg, cfg)
                success = transfer_manager.transfer_call(channel_id)
                if success:
                    log.info("Call transfer initiated successfully")
                    return transfer_msg, True, tts_audio, True, False
                else:
                    log.error("Call transfer failed")
                    transfer_msg = "Δυστυχώς η μεταφορά απέτυχε. Πώς μπορώ να σας βοηθήσω;"
                    transfer_msg = apply_salutation(transfer_msg, caller_profile)
                    tts_audio = await tts_wav16k_cached(transfer_msg, cfg)
                    return transfer_msg, False, tts_audio, False, False
            else:
                if "δεν" not in response_text.lower() and "declined" not in response_text.lower():
                    response_text += " Εντάξει, συνεχίζουμε. Πώς μπορώ να βοηθήσω;"

        # ── NEW: mark if this turn contains a transfer offer so we can catch "Ναι" next turn ──
        match_offer = _TRANSFER_OFFER_RE.search(response_text)
        if match_offer:
            offered_transfer_this_turn = True
            dialog_state["transfer_offer_pending"] = {
                "trigger": response_text,
                "timestamp": time.time(),
                "turn": dialog_state.get("turn_counter", 0)
            }
        else:
            dialog_state["transfer_offer_pending"] = False

        if _looks_like_no_answer(response_text) and rag_active:
            response_text = (
                "Δεν εντόπισα επαρκείς πληροφορίες στα έγγραφά μας για αυτό το θέμα. "
                "Μπορώ να σας μεταφέρω τώρα σε τεχνικό για άμεση βοήθεια. Θέλετε να προχωρήσω;"
            )
            offered_transfer_this_turn = True
            dialog_state["transfer_offer_pending"] = {
                "trigger": response_text,
                "timestamp": time.time(),
                "turn": dialog_state.get("turn_counter", 0)
            }

        response_text = apply_salutation(response_text, caller_profile)
        response_text = clamp_response_length(response_text, cfg)

        tts_audio = await tts_wav16k_cached(response_text, cfg)
        should_end = is_goodbye_message(response_text, cfg)
        return (response_text or "Δεν κατάλαβα. Μπορείτε να επαναλάβετε;"), should_end, tts_audio, False, offered_transfer_this_turn

    except Exception as e:
        turn_stop_event.set()
        if waiter_task and not waiter_task.done():
            waiter_task.cancel()
        log.exception("Responses API failed: %s", e)
        error_text = "Παρουσιάστηκε σφάλμα. Δοκιμάστε ξανά."
        error_text = apply_salutation(error_text, caller_profile)
        tts_audio = await tts_wav16k_cached(error_text, cfg)
        dialog_state["transfer_offer_pending"] = False
        return error_text, False, tts_audio, False, False

# ───────────── RTP helpers ───────────── #

def pick_free_udp_port(bind_ip: str, port_min: int, port_max: int) -> int:
    for port in range(port_min, port_max + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.bind((bind_ip, port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free UDP port in {port_min}-{port_max}")

def parse_rtp_header(pkt: bytes) -> Optional[Tuple[int, int, int, int]]:
    if len(pkt) < 12: return None
    if (pkt[0] >> 6) != 2: return None
    pt = pkt[1] & 0x7F
    seq = struct.unpack_from("!H", pkt, 2)[0]
    ts = struct.unpack_from("!I", pkt, 4)[0]
    ssrc= struct.unpack_from("!I", pkt, 8)[0]
    return pt, seq, ts, ssrc

def rtp_packetize_ulaw(ulaw: bytes, ssrc: int, seq_start: int, ts_start: int,
                       sample_rate: int = 8000, frame_ms: int = 20, pt: int = 0):
    samples_per_packet = sample_rate * frame_ms // 1000
    bytes_per_packet = samples_per_packet
    seq = seq_start & 0xFFFF
    ts = ts_start & 0xFFFFFFFF
    off = 0
    while off < len(ulaw):
        chunk = ulaw[off: off + bytes_per_packet]
        if len(chunk) < bytes_per_packet:
            chunk += b"\xff" * (bytes_per_packet - len(chunk))
        header = struct.pack("!BBHII", 0x80, pt, seq, ts, ssrc)
        yield header + chunk
        seq = (seq + 1) & 0xFFFF
        ts = (ts + samples_per_packet) & 0xFFFFFFFF
        off += bytes_per_packet

# ───────────── μ-law (G.711) ───────────── #

_ULAW_DECODE = np.empty(256, dtype=np.int16)
for code in range(256):
    u = (~code) & 0xFF
    sign = u & 0x80
    seg = (u >> 4) & 0x07
    mant = u & 0x0F
    mag = ((mant << 3) + 0x84) << seg
    val = (mag - 0x84) if mag > 0x84 else 0
    _ULAW_DECODE[code] = -val if sign else val

def ulaw_to_pcm16(ulaw_bytes: bytes) -> bytes:
    if not ulaw_bytes: return b""
    b = np.frombuffer(ulaw_bytes, dtype=np.uint8)
    y = _ULAW_DECODE[b]
    return y.tobytes()

def pcm16_to_ulaw(pcm16_bytes: bytes) -> bytes:
    if not pcm16_bytes: return b""
    x = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.int32)
    BIAS = 0x84
    CLIP = 32635
    sign = (x >> 8) & 0x80
    mag = np.abs(x)
    mag = np.where(mag > CLIP, CLIP, np.abs(mag)).astype(np.int32)
    mag = mag + BIAS
    seg = np.zeros_like(mag)
    seg[mag >= 0x4000] = 7
    seg[(mag >= 0x2000) & (mag < 0x4000)] = 6
    seg[(mag >= 0x1000) & (mag < 0x2000)] = 5
    seg[(mag >= 0x0800) & (mag < 0x1000)] = 4
    seg[(mag >= 0x0400) & (mag < 0x0800)] = 3
    seg[(mag >= 0x0200) & (mag < 0x0400)] = 2
    seg[(mag >= 0x0100) & (mag < 0x0200)] = 1
    mant = (mag >> (seg + 3)) & 0x0F
    ulaw = ~(sign | (seg << 4) | mant)
    return ulaw.astype(np.uint8).tobytes()

# ───────────── Audio utils ───────────── #

def wav_bytes_to_pcm16_mono(wav_bytes: bytes) -> Tuple[bytes, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        nch = w.getnchannels(); sw = w.getsampwidth(); fr = w.getframerate(); nframes = w.getnframes()
        raw = w.readframes(nframes)
        if sw == 2:
            x = np.frombuffer(raw, dtype=np.int16)
        elif sw == 1:
            x = (np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128) << 8
        elif sw == 3:
            a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            x32 = (a[:,0].astype(np.int32) | (a[:,1].astype(np.int32) << 8) | (a[:,2].astype(np.int32) << 16))
            neg = (a[:,2] & 0x80) != 0
            x32[neg] -= 1 << 24
            x = (x32 >> 16).astype(np.int16)
        elif sw == 4:
            x = (np.frombuffer(raw, dtype=np.int32) >> 16).astype(np.int16)
        else:
            raise RuntimeError(f"Unsupported WAV sample width: {sw}")
        if nch > 1:
            x = x.reshape(-1, nch).mean(axis=1).astype(np.int16)
        return x.tobytes(), fr

def resample_pcm16(pcm16: bytes, sr_from: int, sr_to: int) -> bytes:
    if not pcm16 or sr_from == sr_to: return pcm16
    seg = AudioSegment(pcm16, frame_rate=sr_from, sample_width=2, channels=1).set_frame_rate(sr_to)
    out = io.BytesIO(); seg.export(out, format="wav")
    return wav_bytes_to_pcm16_mono(out.getvalue())[0]

def apply_gain_db(pcm16: bytes, sr: int, db: float) -> bytes:
    if not pcm16 or abs(db) < 0.1: return pcm16
    seg = AudioSegment(pcm16, frame_rate=sr, sample_width=2, channels=1).apply_gain(db)
    out = io.BytesIO(); seg.export(out, format="wav")
    return wav_bytes_to_pcm16_mono(out.getvalue())[0]

# ───────────── VAD / Endpointing ───────────── #

def _ulaw_energy(frame: bytes) -> float:
    if not frame: return 0.0
    s = np.frombuffer(ulaw_to_pcm16(frame), dtype=np.int16).astype(np.float32) / 32768.0
    if s.size == 0: return 0.0
    return float(np.sqrt(np.mean(s*s)))

def _zcr(frame: bytes) -> float:
    if not frame: return 0.0
    x = np.frombuffer(ulaw_to_pcm16(frame), dtype=np.int16)
    if x.size < 2: return 0.0
    return float(np.mean(np.abs(np.diff(np.sign(x)))) / 2.0)

async def _calibrate_noise(sock: socket.socket, remote_addr, frame_ms: int, dur_ms: int, pt_expect: int = 0):
    loop = asyncio.get_running_loop()
    fb = int(8000 * frame_ms / 1000)
    end = time.monotonic() + dur_ms/1000.0
    energies, zcrs = [], []
    while time.monotonic() < end and not _shutdown.is_set():
        try:
            pkt, addr = await asyncio.wait_for(loop.sock_recvfrom(sock, 2048), timeout=0.25)
        except asyncio.TimeoutError:
            continue
        if addr != remote_addr: continue
        hdr = parse_rtp_header(pkt)
        if not hdr: continue
        pt, _, _, _ = hdr
        if pt != pt_expect: continue
        payload = pkt[12:]; i = 0
        while i < len(payload):
            fr = payload[i:i+fb]
            if len(fr) < fb: break
            i += fb
            energies.append(_ulaw_energy(fr))
            zcrs.append(_zcr(fr))
    if not energies: return 0.01, 0.1
    return float(np.median(energies)), float(np.median(zcrs) if zcrs else 0.1)

async def capture_utterance_ulaw(sock: socket.socket,
                                 remote_addr,
                                 vad_cfg: dict,
                                 pt_expect: int = 0) -> Tuple[bytes, float]:
    loop = asyncio.get_running_loop()
    frame_ms = int(vad_cfg["frame_ms"])
    fb = int(8000 * frame_ms / 1000)
    start_hang = int(vad_cfg["start_hang_ms"])
    end_sil = int(vad_cfg["end_silence_ms"])
    pause_gr = int(vad_cfg["pause_grace_ms"])
    min_speech = int(vad_cfg["min_speech_ms"])
    max_utt = int(vad_cfg["max_utterance_ms"])
    auto_ms = int(vad_cfg["auto_calibrate_ms"])
    factor = float(vad_cfg["energy_factor"])
    floor_min = float(vad_cfg["energy_floor"])
    max_zcr = float(vad_cfg["max_zcr"])
    adaptive_silence = vad_cfg.get("adaptive_silence", True)
    speech_extension = int(vad_cfg.get("speech_extension_ms", 2000))
    min_between = int(vad_cfg.get("min_speech_between_pauses", 500))
    pre_roll_ms = int(vad_cfg.get("pre_roll_ms", 240))
    pre_frames = max(0, pre_roll_ms // frame_ms)
    preroll_buf = deque(maxlen=pre_frames)

    noise_rms, noise_z = await _calibrate_noise(sock, remote_addr, frame_ms, auto_ms, pt_expect)
    thr = max(floor_min, noise_rms * factor)
    log.info("VAD calib: noise_rms=%.4f, noise_zcr=%.3f, thr=%.4f (factor=%.2f)", noise_rms, noise_z, thr, factor)

    started = False
    ok_ms = 0; cont_sil = 0; sum_sil = 0; speech_ms = 0; consec_speech = 0
    out = bytearray()
    t0 = time.monotonic()
    current_end_sil = end_sil

    while not _shutdown.is_set():
        if time.monotonic() - t0 > 60.0:
            log.warning("VAD timeout reached, returning captured audio")
            break

        try:
            pkt, addr = await asyncio.wait_for(loop.sock_recvfrom(sock, 2048), timeout=0.25)
        except asyncio.TimeoutError:
            if started and cont_sil >= current_end_sil:
                break
            continue

        if addr != remote_addr:
            continue
        hdr = parse_rtp_header(pkt)
        if not hdr:
            continue
        pt, _, _, _ = hdr
        if pt != pt_expect:
            continue

        payload = pkt[12:]
        i = 0
        while i < len(payload):
            fr = payload[i:i+fb]
            if len(fr) < fb: break
            i += fb

            e = _ulaw_energy(fr)
            z = _zcr(fr)

            if not started:
                preroll_buf.append(fr)
                if (e > thr) and (z <= max_zcr * 1.5):
                    ok_ms += frame_ms
                    if ok_ms >= start_hang:
                        started = True
                        for pfr in preroll_buf: out.extend(pfr)
                        cont_sil = 0; sum_sil = 0; speech_ms = len(preroll_buf) * frame_ms; consec_speech = speech_ms
                        current_end_sil = end_sil
                else:
                    ok_ms = max(0, ok_ms - frame_ms)
            else:
                out.extend(fr)
                speech_ms += frame_ms
                is_speech = (e > thr * 0.75) and (z <= max_zcr * 1.4)
                if is_speech:
                    consec_speech += frame_ms
                    cont_sil = 0
                    if adaptive_silence and consec_speech > min_between:
                        current_end_sil = min(end_sil + speech_extension, 5000)
                else:
                    cont_sil += frame_ms
                    sum_sil += frame_ms
                    if cont_sil > 1000:
                        consec_speech = 0
                        current_end_sil = end_sil

                if cont_sil >= current_end_sil or sum_sil >= pause_gr or speech_ms >= max_utt:
                    break

        if started and (cont_sil >= current_end_sil or sum_sil >= pause_gr or speech_ms >= max_utt):
            break

    if not started or speech_ms < min_speech:
        log.info("VAD: no valid utterance (started=%s, speech_ms=%d)", started, speech_ms)
        return b"", noise_rms

    duration_sec = speech_ms / 1000.0
    log.info("VAD: captured utterance %.1f seconds (speech: %d ms, total silence: %d ms, adaptive: %d ms, pre_roll: %d ms)",
             duration_sec, speech_ms, sum_sil, current_end_sil, pre_roll_ms)
    return bytes(out), noise_rms

# ───────────── Barge-In helpers ───────────── #

def _pcm16_from_ulaw(ulaw_chunk: bytes) -> np.ndarray:
    if not ulaw_chunk: return np.zeros(0, dtype=np.int16)
    return np.frombuffer(ulaw_to_pcm16(ulaw_chunk), dtype=np.int16)

def _corr_coeff(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0: return 0.0
    n = min(a.size, b.size)
    ax = a[:n].astype(np.float32); bx = b[:n].astype(np.float32)
    sa = ax.std(); sb = bx.std()
    if sa < 1e-6 or sb < 1e-6: return 0.0
    return float(((ax - ax.mean()) * (bx - bx.mean())).mean() / (sa * sb))

class BargeInDetector:
    def __init__(self, noise_rms: float, cfg: dict):
        self.noise_rms = max(1e-6, float(noise_rms))
        self.cfg = cfg
        self.frame_ms = int(cfg["frame_ms"])
        self.energy_thr = max(cfg["energy_floor"], self.noise_rms * cfg["energy_factor"])
        self.max_zcr = float(cfg["max_zcr"])
        self.min_snr = float(cfg["min_snr_db"])
        self.start_hang_ms = int(cfg["start_hang_ms"])
        self.hold_ms = int(cfg["hold_ms"])
        self.run_ms = 0
        self.active = False
        self.hold_acc = 0

    def step(self, in_ulaw_frame: bytes, out_ref_pcm_8k: Optional[np.ndarray], echo_corr_thresh: float) -> bool:
        pcm = _pcm16_from_ulaw(in_ulaw_frame).astype(np.float32) / 32768.0
        if pcm.size == 0: return False
        rms = float(np.sqrt(np.mean(pcm * pcm)) + 1e-9)
        snr_db = 20.0 * np.log10(max(rms,1e-8)/max(self.noise_rms,1e-8))
        x = (pcm > 0).astype(np.int8); zcr = float(np.mean(np.abs(np.diff(x))))
        corr = 0.0
        if out_ref_pcm_8k is not None and out_ref_pcm_8k.size > 0:
            win = int(0.02 * 8000)
            ref_tail = out_ref_pcm_8k[-win:].astype(np.float32)
            in_tail = (pcm[-win:] * 32768.0).astype(np.float32)
            corr = _corr_coeff(ref_tail, in_tail)
        voiced = (rms > self.energy_thr) and (snr_db >= self.min_snr) and (zcr <= self.max_zcr) and (corr < echo_corr_thresh)

        if not self.active:
            if voiced:
                self.run_ms += self.frame_ms
                if self.run_ms >= self.start_hang_ms:
                    self.active = True
                    self.hold_acc = 0
            else:
                self.run_ms = max(0, self.run_ms - self.frame_ms)
            return False
        else:
            if voiced:
                self.hold_acc += self.frame_ms
                if self.hold_acc >= self.hold_ms:
                    return True
            else:
                self.active = False; self.run_ms = 0; self.hold_acc = 0
            return False

async def barge_in_precalibrate(sock, addr, cfg_bi: dict) -> float:
    loop = asyncio.get_running_loop()
    frame_ms = int(cfg_bi["frame_ms"]); fb = int(8000 * frame_ms / 1000)
    dur_ms = int(cfg_bi["pre_calibrate_ms"]); end = time.monotonic() + dur_ms / 1000.0
    energies = []
    while time.monotonic() < end and not _shutdown.is_set():
        try:
            pkt, a = await asyncio.wait_for(loop.sock_recvfrom(sock, 2048), timeout=0.02)
        except asyncio.TimeoutError:
            continue
        if a != addr: continue
        hdr = parse_rtp_header(pkt)
        if not hdr: continue
        pt, _, _, _ = hdr
        if pt != 0: continue
        payload = pkt[12:]; i = 0
        while i < len(payload):
            fr = payload[i:i + fb]
            if len(fr) < fb: break
            i += fb
            energies.append(_ulaw_energy(fr))
    if not energies: return 0.01
    return float(np.median(energies))

async def play_tts_with_barge_in(sock, addr, ssrc, seq, ts, ulaw_bytes: bytes, out_ref_pcm8k: np.ndarray,
                                 cfg: dict, noise_rms_override: Optional[float] = None) -> Tuple[int, int, bool]:
    loop = asyncio.get_running_loop()
    cfg_bi = cfg.get("barge_in", DEFAULTS["barge_in"])
    if not cfg_bi.get("enabled", True):
        started = time.monotonic(); sent = 0
        for pkt in rtp_packetize_ulaw(ulaw_bytes, ssrc, seq, ts, 8000, 20, 0):
            await loop.sock_sendto(sock, pkt, addr)
            sent += 1
            target = started + sent * 0.020
            rem = target - time.monotonic()
            if rem > 0: await asyncio.sleep(rem)
        seq = (seq + sent) & 0xFFFF; ts = (ts + sent * 160) & 0xFFFFFFFF
        return seq, ts, False

    if noise_rms_override is None:
        noise_rms = await barge_in_precalibrate(sock, addr, cfg_bi)
    else:
        noise_rms = float(noise_rms_override)

    detector = BargeInDetector(noise_rms, cfg_bi)
    echo_thr = float(cfg_bi["echo_corr_thresh"])

    frame_ms = int(cfg_bi["frame_ms"]); fb = int(8000 * frame_ms / 1000)
    started = time.monotonic(); sent = 0
    out_ref = out_ref_pcm8k.astype(np.int16) if out_ref_pcm8k is not None else np.zeros(0, dtype=np.int16)
    ref_idx = 0

    for pkt in rtp_packetize_ulaw(ulaw_bytes, ssrc, seq, ts, 8000, 20, 0):
        await loop.sock_sendto(sock, pkt, addr)
        sent += 1
        ref_idx = min(out_ref.size, ref_idx + fb)

        target = started + sent * 0.020
        while True:
            rem = target - time.monotonic()
            if rem <= 0: break
            try:
                pkt_in, a = await asyncio.wait_for(loop.sock_recvfrom(sock, 2048), timeout=min(0.004, rem))
            except asyncio.TimeoutError:
                await asyncio.sleep(min(0.003, rem))
                continue
            if a != addr: continue
            hdr = parse_rtp_header(pkt_in)
            if not hdr: continue
            pt, _, _, _ = hdr
            if pt != 0: continue

            payload = pkt_in[12:]; i = 0
            while i < len(payload):
                fr = payload[i:i + fb]
                if len(fr) < fb: break
                i += fb
                tail = None
                if ref_idx >= fb:
                    tail = out_ref[max(0, ref_idx - fb):ref_idx]
                if detector.step(fr, tail, echo_thr):
                    log.info("BARGE-IN detected: stopping TTS playback early.")
                    seq = (seq + sent) & 0xFFFF
                    ts = (ts + sent * 160) & 0xFFFFFFFF
                    return seq, ts, True

    seq = (seq + sent) & 0xFFFF
    ts = (ts + sent * 160) & 0xFFFFFFFF
    return seq, ts, False

# ───────────── Enhanced Pipeline with Transfer Support ───────────── #

async def dialog_pipeline(cfg: dict, bind_ip: str, dyn_port: int, greeting_text: str, 
                          tenant_id: Optional[str], channel_id: str, ari: AriClient):
    loop = asyncio.get_running_loop()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    transfer_manager = CallTransferManager(ari, cfg)

    # single-waiter flag kept for back-compat, no longer used with wait_every_turn=True
    waiter_used_flag = {"used": False}

    history: List[Dict[str, str]] = []
    caller_profile: Dict[str, Optional[str]] = {"last_name": None, "gender": None}

    # ── NEW: dialog state for pending transfer offers
    dialog_state: Dict[str, Any] = {"transfer_offer_pending": False}
    dialog_state.setdefault("conversation_id", channel_id)

    try:
        sock.bind((bind_ip, dyn_port))
        sock.setblocking(False)
        log.info("RTP socket bound to %s:%d, waiting for first inbound RTP...", bind_ip, dyn_port)

        try:
            first_pkt, addr = await asyncio.wait_for(loop.sock_recvfrom(sock, 2048), timeout=10.0)
            log.info("Received first RTP packet from %s:%d", addr[0], addr[1])
        except asyncio.TimeoutError:
            log.error("Timeout waiting for first RTP packet from Asterisk")
            return

        parsed = parse_rtp_header(first_pkt)
        if not parsed:
            log.error("First packet invalid RTP")
            return
        in_pt, in_seq, in_ts, in_ssrc = parsed
        if in_pt != 0:
            log.warning("Inbound PT=%d (expected PCMU=0)", in_pt)

        rtp_manager = RTPStateManager(in_seq, in_ts, in_ssrc)
        await rtp_manager.advance_sequence(1)

        greet = (greeting_text or DEFAULTS["greeting_text"]).strip()
        log.info("Starting greeting TTS...")
        g_pcm16 = await tts_wav16k_cached(greet, cfg)
        g_pcm8 = resample_pcm16(g_pcm16, 16000, 8000)
        g_ulaw = pcm16_to_ulaw(g_pcm8)
        
        bytes_per_packet = 160
        greeting_packets = (len(g_ulaw) + bytes_per_packet - 1) // bytes_per_packet
        greet_seq, greet_ts = await rtp_manager.get_next_sequence(greeting_packets)
        
        lead_ms = int(cfg["dialog"]["leading_silence_ms"])
        if lead_ms > 0:
            g_ulaw = (b'\xff' * (8000 * lead_ms // 1000)) + g_ulaw

        started = time.monotonic(); sent = 0
        for pkt in rtp_packetize_ulaw(g_ulaw, rtp_manager.ssrc, greet_seq, greet_ts, 8000, 20, 0):
            await loop.sock_sendto(sock, pkt, addr)
            sent += 1
            target = started + sent*0.020
            rem = target - time.monotonic()
            if rem > 0: await asyncio.sleep(rem)
        
        log.info("Greeting RTP stream ended: %d packets", sent)

        max_turns = int(cfg["dialog"].get("max_turns", 0) or 0)
        vad_cfg = cfg["vad"]
        turn = 0
        conversation_context = "τεχνικά προβλήματα, υπολογιστές, συσκευές"

        while True:
            if max_turns > 0 and turn >= max_turns:
                log.info("Reached configured max_turns=%d; ending dialog.", max_turns)
                break

            turn_label = turn + 1
            max_label = str(max_turns) if max_turns > 0 else "∞"
            log.info("Turn %s/%s - Listening...", turn_label, max_label)
            in_ulaw, last_noise_rms = await capture_utterance_ulaw(sock, addr, vad_cfg, pt_expect=0)
            if not in_ulaw:
                log.info("No inbound utterance (silence/too short); ending dialog.")
                break

            pcm8_in = ulaw_to_pcm16(in_ulaw)
            pcm16_in = resample_pcm16(pcm8_in, 8000, 16000)
            
            user_txt = transcribe_pcm16_16k_enhanced(pcm16_in, cfg, conversation_context)
            log.info("STT → %s", user_txt)

            try:
                last_name = extract_last_name_from_text(user_txt)
                if last_name:
                    caller_profile["last_name"] = last_name
                    log.info("Detected caller last name: %s", last_name)
                gender = detect_gender_from_text(user_txt)
                if gender:
                    caller_profile["gender"] = gender
                    log.info("Detected caller gender: %s", gender)
            except Exception:
                pass

            history.append({"role": "user", "content": user_txt})

            await log_conversation_event(
                cfg,
                "user",
                user_txt,
                dialog_state.get("conversation_id", channel_id),
                turn_label,
                tenant_id,
                {"direction": "inbound"}
            )

            reply, should_end, tts16, should_transfer, offered = await chat_with_model(
                cfg, user_txt, channel_id, tenant_id, transfer_manager, sock, addr, rtp_manager,
                waiter_used_flag, history, caller_profile, dialog_state
            )

            await log_conversation_event(
                cfg,
                "assistant",
                reply,
                dialog_state.get("conversation_id", channel_id),
                turn_label,
                tenant_id,
                {
                    "direction": "outbound",
                    "offered_transfer": offered,
                    "should_transfer": should_transfer
                }
            )

            if should_transfer:
                log.info("Call transfer confirmed, ending dialog pipeline")
                return

            try:
                history.append({"role": "assistant", "content": reply})
            except Exception:
                pass

            tts8 = resample_pcm16(tts16, 16000, 8000)
            out_ulaw = pcm16_to_ulaw(tts8)
            out_ref_pcm8k = np.frombuffer(tts8, dtype=np.int16)

            main_packets = (len(out_ulaw) + bytes_per_packet - 1) // bytes_per_packet
            main_seq, main_ts = await rtp_manager.get_next_sequence(main_packets)

            _, _, barged = await play_tts_with_barge_in(
                sock, addr, rtp_manager.ssrc, main_seq, main_ts, out_ulaw, out_ref_pcm8k, cfg,
                noise_rms_override=last_noise_rms
            )
            log.info("Main TTS RTP stream ended. barged=%s", barged)

            if should_end:
                log.info("Goodbye detected from user or response text, ending conversation.")
                break

            if barged:
                prompt = "Θα θέλατε κάτι άλλο;"
                prompt = apply_salutation(prompt, caller_profile)
                w16 = await tts_wav16k_cached(prompt, cfg)
                w8 = resample_pcm16(w16, 16000, 8000)
                out = pcm16_to_ulaw(w8)
                
                prompt_packets = (len(out) + bytes_per_packet - 1) // bytes_per_packet
                prompt_seq, prompt_ts = await rtp_manager.get_next_sequence(prompt_packets)
                
                startedp = time.monotonic(); sentp = 0
                for pkt in rtp_packetize_ulaw(out, rtp_manager.ssrc, prompt_seq, prompt_ts, 8000, 20, 0):
                    await loop.sock_sendto(sock, pkt, addr)
                    sentp += 1
                    targetp = startedp + sentp*0.020
                    remp = targetp - time.monotonic()
                    if remp > 0: await asyncio.sleep(remp)

                log.info("Barge-in: returning to listening without consuming a turn.")
                continue

            turn += 1

        log.info("Dialog completed naturally.")
    finally:
        sock.close()

# ───────────── ARI handlers ───────────── #

def make_handlers(cfg: dict, ari: AriClient):
    em = cfg["external_media"]; bind_ip = em["bind_ip"]
    pm = int(em.get("port_min", 30000)); px = int(em.get("port_max", 40050))

    def on_stasis_start(evt):
        if _shutdown.is_set(): return
        ch = evt.get("channel") or {}; ch_id = ch.get("id"); ch_name = ch.get("name","")
        linked = ch.get("linkedid") or ch_id
        if ch_name.startswith("External/"): return
        with _external_lock:
            if ch_id in _external_channel_ids: return
        with _active_lock:
            if linked in _active_keys: return
            _active_keys.add(linked)

        try:
            dyn = pick_free_udp_port(bind_ip, pm, px)
            external_host = f"{bind_ip}:{dyn}"
            log.info("New call: channel=%s, RTP port=%d", ch_id, dyn)

            ari.channels_answer(ch_id)
            ext = ari.channels_external_media(external_host, fmt="ulaw", direction="both")
            ext_id = ext.get("id")
            if ext_id:
                with _external_lock: _external_channel_ids.add(ext_id)

            br = ari.bridges_create(); bid = br.get("id")
            ari.bridges_add_channel(bid, ch_id); ari.bridges_add_channel(bid, ext_id)

            with _active_calls_lock:
                _active_calls[ch_id] = {
                    "bridge_id": bid,
                    "external_id": ext_id,
                    "caller_id": ch_id
                }

            log.info("Bridge created, starting dialog pipeline")
            greeting = cfg.get("greeting_text") or DEFAULTS["greeting_text"]
            tenant_id = get_tenant_id_from_evt(evt, cfg)

            def _run():
                try:
                    asyncio.run(dialog_pipeline(cfg, bind_ip, dyn, greeting, tenant_id, ch_id, ari))
                except Exception as e:
                    log.exception("dialog_pipeline failed: %s", e)
                finally:
                    with _active_lock: _active_keys.discard(linked)

            threading.Thread(target=_run, daemon=True).start()
        except Exception as e:
            log.exception("Error in StasisStart: %s", e)
            with _active_lock: _active_keys.discard(linked)

    def on_stasis_end(evt):
        ch = evt.get("channel") or {}; ch_id = ch.get("id")
        linked = ch.get("linkedid") or ch_id
        with _active_lock: _active_keys.discard(linked)
        with _external_lock: _external_channel_ids.discard(ch_id)
        with _active_calls_lock:
            _active_calls.pop(ch_id, None)
        log.info("Call ended: %s", linked)

    return on_stasis_start, on_stasis_end

# ───────────── Main ───────────── #

def main():
    global oa
    try:
        with open("config.yaml","r",encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        log.info("Configuration loaded successfully")
    except Exception as e:
        log.error("Configuration error: %s", e); return

    for k in ("openai","agent","tts","stt","dialog","transfer","vad","barge_in","retrieval","elasticsearch","external_media","ari"):
        cfg.setdefault(k, {})
        for key, val in DEFAULTS[k].items():
            cfg[k].setdefault(key, val)

    oa = build_openai_client(cfg)

    retrieval_conf = cfg.get("retrieval", {})
    mode = retrieval_conf.get("mode") or ("elastic" if cfg.get("elasticsearch", {}).get("enabled") else "openai_file_search")
    retrieval_conf["mode"] = mode

    use_file_search = bool(retrieval_conf.get("enabled") and mode == "openai_file_search")
    if use_file_search:
        raw_ids = retrieval_conf.get("vector_store_ids") or []
        if not isinstance(raw_ids, list):
            log.error("retrieval.vector_store_ids must be a list of strings (OpenAI vector store IDs)")
            return
        valid_ids = validate_vector_stores(oa, raw_ids)
        if not valid_ids:
            log.error("No accessible vector stores from config; disabling file_search retrieval.")
            retrieval_conf["enabled"] = False
        retrieval_conf["_validated_vector_store_ids"] = valid_ids
    else:
        retrieval_conf["_validated_vector_store_ids"] = []

    global elastic_rag
    elastic_rag = None
    elastic_cfg = cfg.get("elasticsearch", {})
    if mode == "elastic" and elastic_cfg.get("enabled"):
        try:
            elastic_rag = ElasticVectorRetriever(elastic_cfg, oa)
            log.info("Elastic retriever initialized for index: %s", elastic_cfg.get("index"))
        except Exception as e:
            log.error("Elastic retriever init failed: %s", e)
            elastic_cfg["enabled"] = False
            retrieval_conf["mode"] = "none"

    host = cfg["ari"]["host"].split("://")[-1]
    try:
        ari = AriClient(host, cfg["ari"]["port"], cfg["ari"]["username"], cfg["ari"]["password"], cfg["ari"]["app"])
        log.info("ARI client initialized for app: %s", cfg["ari"]["app"])
    except Exception as e:
        log.error("ARI client initialization failed: %s", e); return

    on_start, on_end = make_handlers(cfg, ari)
    ari.on("StasisStart", on_start); ari.on("StasisEnd", on_end)

    def signal_handler(signum, frame):
        log.info("Received signal %s, shutting down...", signum); _shutdown.set()
    signal.signal(signal.SIGINT, signal_handler); signal.signal(signal.SIGTERM, signal_handler)

    try:
        ari.start_events()
        log.info("ARI event listener started. Waiting for calls...")
        while not _shutdown.is_set(): time.sleep(0.1)
    except Exception as e:
        log.error("ARI event loop failed: %s", e)
    finally:
        log.info("Application shutdown complete")

if __name__ == "__main__":
    main()
