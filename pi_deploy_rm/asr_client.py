#!/usr/bin/env python3
"""
asr_client.py — Stage 1: Real ASR Streaming for VaaK / Dracarys KWS.

Replaces the {\"eof\": 1} stub in live_kws.py with genuine continuous PCM
streaming to the Vosk WebSocket server. Instruments four timestamps per
detection:

    T1 = keyword-end time (last sample of the window that crossed THRESHOLD)
    T2 = WebSocket connection established (server-side handshake complete)
    T3 = first audio byte confirmed received by server (via server ack/partial)
    T4 = final transcript received

Trailing-silence detection ends the utterance early; a hard max_duration_s
cap provides a safety ceiling. Both configurable.

Preserves the existing 30s Vosk backoff logic from the CPU fix — not removed.

Usage (standalone test mode):
    python3 asr_client.py --test --wav /path/to/utterance.wav
"""

import json
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np

try:
    import websocket
    _WEBSOCKET_AVAILABLE = True
except ImportError:
    _WEBSOCKET_AVAILABLE = False
    websocket = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000                   # Must match live_kws.py / features.py
CHUNK_FRAMES = 3200                   # 200 ms chunks — same as hop stride
CHUNK_BYTES = CHUNK_FRAMES * 2        # 16-bit PCM = 2 bytes per sample
DEFAULT_MAX_DURATION_S = 6.0          # Hard cap on post-wake recording window
DEFAULT_SILENCE_THRESHOLD_RMS = 0.01  # Normalised RMS (~-40 dBFS) for silence
DEFAULT_SILENCE_FRAMES = 8            # Consecutive silent 200 ms chunks → EOT
VOSK_RECONNECT_BACKOFF_S = 30         # Backoff from CPU-fix — do not remove

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class DetectionRecord:
    """Full timing record for one keyword + ASR cycle."""
    t1_keyword_end: float = 0.0         # epoch seconds — last KWS window sample
    t2_ws_connected: float = 0.0        # epoch — WebSocket handshake complete
    t3_server_ack: float = 0.0          # epoch — first partial from Vosk received
    t4_final_transcript: float = 0.0   # epoch — final transcript received
    transcript: str = ""
    spk_vector: Optional[List[float]] = None
    speaker_identity: str = "UNKNOWN"
    ws_latency_ms: float = 0.0          # T2 - T1
    first_byte_latency_ms: float = 0.0  # T3 - T1
    total_latency_ms: float = 0.0       # T4 - T1
    success: bool = False
    error: Optional[str] = None
    raw_final_json: str = "{}"

    def summary(self) -> str:
        return (
            f"T1={self.t1_keyword_end:.3f}  T2={self.t2_ws_connected:.3f}"
            f"  T3={self.t3_server_ack:.3f}  T4={self.t4_final_transcript:.3f}"
            f"  ws_lat={self.ws_latency_ms:.1f}ms"
            f"  first_byte_lat={self.first_byte_latency_ms:.1f}ms"
            f"  total_lat={self.total_latency_ms:.1f}ms"
            f'  transcript="{self.transcript}"'
        )


# ---------------------------------------------------------------------------
# Connection manager (stateful, thread-safe reconnect with backoff)
# ---------------------------------------------------------------------------
class VoskConnection:
    """
    Manages a single persistent WebSocket connection to the Vosk server.
    Re-opens on failure with the 30s backoff preserved from the CPU fix.
    Thread-safe: connect/close serialised by _lock.
    """

    def __init__(self, server_url: str):
        self._url = server_url
        self._ws: Optional[object] = None
        self._lock = threading.Lock()
        self._last_fail_time: float = 0.0

    # ------------------------------------------------------------------
    def connect(self) -> bool:
        """
        Attempt to (re)open the WebSocket to Vosk.
        Returns True on success, False if not available or in backoff window.
        Records T2 externally — caller reads _ws after this returns True.
        """
        with self._lock:
            if not _WEBSOCKET_AVAILABLE:
                log.warning("websocket-client not installed — ASR unavailable.")
                return False

            # 30-second backoff from CPU fix — preserved as-is
            if self._last_fail_time and (
                time.time() - self._last_fail_time < VOSK_RECONNECT_BACKOFF_S
            ):
                remaining = VOSK_RECONNECT_BACKOFF_S - (
                    time.time() - self._last_fail_time
                )
                log.info(
                    "Vosk backoff active — %.0fs remaining. Skipping ASR.", remaining
                )
                return False

            # Close stale connection if present
            if self._ws is not None:
                try:
                    self._ws.close()
                except Exception:
                    pass
                self._ws = None

            try:
                ws = websocket.create_connection(self._url, timeout=2.0)
                # Send initial config — Vosk requires sample_rate before audio
                ws.send(
                    json.dumps({"config": {"sample_rate": SAMPLE_RATE}})
                )
                self._ws = ws
                return True
            except Exception as exc:
                log.warning("Vosk connect failed: %s", exc)
                self._last_fail_time = time.time()
                return False

    # ------------------------------------------------------------------
    def get_ws(self):
        """Return the raw websocket object (may be None)."""
        return self._ws

    # ------------------------------------------------------------------
    def close(self):
        with self._lock:
            if self._ws is not None:
                try:
                    self._ws.close()
                except Exception:
                    pass
                self._ws = None


# ---------------------------------------------------------------------------
# Trailing-silence detector
# ---------------------------------------------------------------------------
def _rms(samples: np.ndarray) -> float:
    """RMS of normalised float32 PCM (−1.0 … +1.0)."""
    return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))


def _is_silent(
    chunk_int16: bytes,
    threshold: float = DEFAULT_SILENCE_THRESHOLD_RMS,
) -> bool:
    """True when the RMS of this PCM chunk is below the silence threshold."""
    samples = np.frombuffer(chunk_int16, dtype=np.int16).astype(np.float32)
    samples /= 32768.0
    return _rms(samples) < threshold


# ---------------------------------------------------------------------------
# Core streaming function
# ---------------------------------------------------------------------------
def stream_utterance(
    audio_queue: "queue.Queue[Optional[bytes]]",
    connection: VoskConnection,
    t1_keyword_end: float,
    max_duration_s: float = DEFAULT_MAX_DURATION_S,
    silence_threshold_rms: float = DEFAULT_SILENCE_THRESHOLD_RMS,
    silence_frames: int = DEFAULT_SILENCE_FRAMES,
) -> DetectionRecord:
    """
    Stream raw int16 PCM bytes from *audio_queue* to the Vosk WebSocket server
    and return a fully-populated DetectionRecord.

    Protocol
    --------
    1. Connect (→ T2).
    2. Send PCM in CHUNK_FRAMES-sized chunks continuously (NOT a single batch).
    3. Collect any partial results — first partial arrival → T3.
    4. End utterance on trailing silence OR max_duration_s, whichever comes first.
    5. Send {"eof": 1}, receive final result → T4.

    The caller feeds int16 PCM as raw bytes into *audio_queue*.
    Pushing ``None`` signals end-of-stream early (e.g., on Ctrl-C).
    """
    rec = DetectionRecord(t1_keyword_end=t1_keyword_end)

    # ------------------------------------------------------------------ T2
    ok = connection.connect()
    rec.t2_ws_connected = time.time()
    rec.ws_latency_ms = (rec.t2_ws_connected - rec.t1_keyword_end) * 1000.0

    if not ok:
        rec.error = "Vosk server not reachable (backoff or unavailable)."
        log.warning(rec.error)
        return rec

    ws = connection.get_ws()

    partials: List[str] = []
    t3_captured = False
    silent_count = 0
    deadline = time.time() + max_duration_s

    try:
        # Set socket to non-blocking receive so we can interleave send/recv
        ws.sock.setblocking(False)  # type: ignore[attr-defined]
    except Exception:
        pass  # Older websocket-client versions may not support this; continue

    # ------------------------------------------------------------------ stream
    while True:
        # Pull next PCM chunk from queue (timeout keeps us from hanging forever)
        try:
            chunk: Optional[bytes] = audio_queue.get(timeout=0.05)
        except queue.Empty:
            chunk = b""

        # Sentinel — upstream signalled end-of-stream
        if chunk is None:
            break

        if chunk:
            # Send PCM chunk to server (continuous streaming, not batch)
            try:
                ws.send_binary(chunk)
            except Exception as send_exc:
                log.error("WebSocket send error: %s", send_exc)
                rec.error = str(send_exc)
                break

            # Trailing-silence detection
            if _is_silent(chunk, silence_threshold_rms):
                silent_count += 1
            else:
                silent_count = 0

            if silent_count >= silence_frames:
                log.info("Trailing silence detected — ending utterance.")
                break

        # ---------------------------------------------------------------- T3
        # Poll for server partials (non-blocking)
        try:
            raw = ws.recv()
            if raw:
                msg = json.loads(raw)
                
                # Check for first byte ack
                if msg.get("ack") == "first_audio_byte" and not t3_captured:
                    rec.t3_server_ack = time.time()
                    rec.first_byte_latency_ms = (
                        rec.t3_server_ack - rec.t1_keyword_end
                    ) * 1000.0
                    t3_captured = True
                    log.debug("T3 captured (first audio byte ack from server)")

                # Collect intermediate final results (when silence breaks)
                if "text" in msg and msg["text"].strip():
                    rec.transcript += msg["text"] + " "
                    log.info("Intermediate final: %s", msg["text"])
                if "spk" in msg:
                    rec.spk_vector = msg["spk"]
                    
                partial_text = msg.get("partial", "")
                if partial_text:
                    partials.append(partial_text)
                    log.info("Partial: %s", partial_text)
        except Exception:
            pass  # Non-blocking — no data yet is normal

        # Hard deadline
        if time.time() >= deadline:
            log.info("Max duration reached — ending utterance.")
            break

    # ------------------------------------------------------------------ T4
    # Signal end-of-utterance and get final transcript
    try:
        ws.sock.setblocking(True)  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        ws.send('{"eof" : 1}')
        final_raw = ws.recv()
        rec.t4_final_transcript = time.time()
        rec.raw_final_json = final_raw
        final_msg = json.loads(final_raw)
        if "text" in final_msg and final_msg["text"].strip():
            rec.transcript += final_msg["text"]
        if "spk" in final_msg:
            rec.spk_vector = final_msg["spk"]
        rec.transcript = rec.transcript.strip()
        rec.total_latency_ms = (
            rec.t4_final_transcript - rec.t1_keyword_end
        ) * 1000.0
        rec.success = True
        log.info("Final transcript: %s", rec.transcript)
    except Exception as final_exc:
        log.error("Failed to get final transcript: %s", final_exc)
        rec.error = str(final_exc)
        rec.t4_final_transcript = time.time()
        rec.total_latency_ms = (
            rec.t4_final_transcript - rec.t1_keyword_end
        ) * 1000.0

    if not t3_captured:
        # No partial was received — server may have responded only with final.
        # Use T4 as a conservative T3 (worst-case — will be flagged in report).
        rec.t3_server_ack = rec.t4_final_transcript
        rec.first_byte_latency_ms = rec.total_latency_ms
        log.warning(
            "T3 not independently captured (no partials). "
            "Using T4 as conservative T3 estimate — flag this in gate report."
        )

    return rec


# ---------------------------------------------------------------------------
# Session statistics helper
# ---------------------------------------------------------------------------
def print_latency_summary(records: List[DetectionRecord]) -> None:
    """Print per-detection timestamps and aggregate percentile stats."""
    valid = [r for r in records if r.t3_server_ack > 0]
    print("\n" + "=" * 72)
    print("ASR CLIENT — DETECTION LATENCY SUMMARY")
    print("=" * 72)
    for i, r in enumerate(records, 1):
        status = "OK" if r.success else f"FAIL({r.error})"
        print(f"  [{i:02d}] {status}  {r.summary()}")
    if valid:
        t3_t1 = np.array([r.first_byte_latency_ms for r in valid])
        print(f"\n  n={len(valid)} valid detections (T3-T1):")
        print(f"    mean   = {np.mean(t3_t1):.2f} ms")
        print(f"    p95    = {np.percentile(t3_t1, 95):.2f} ms")
        print(f"    min    = {np.min(t3_t1):.2f} ms")
        print(f"    max    = {np.max(t3_t1):.2f} ms")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Standalone test mode
# ---------------------------------------------------------------------------
def _test_main():
    """
    Standalone test: streams a WAV file (or a generated tone) through the
    Vosk server and prints full T1–T4 results.

    Usage:
        python3 asr_client.py --test [--wav /path/to/file.wav]
                              [--url ws://localhost:2700]
    """
    import argparse
    import wave

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="asr_client Stage 1 test")
    parser.add_argument("--test", action="store_true", required=True)
    parser.add_argument("--wav", default=None, help="Path to 16kHz mono WAV file")
    parser.add_argument(
        "--url",
        default=os.environ.get("VOSK_SERVER_URL", "ws://localhost:2700"),
        help="Vosk WebSocket URL",
    )
    parser.add_argument(
        "--runs", type=int, default=1, help="Number of test runs (for multi-detection)"
    )
    args = parser.parse_args()

    conn = VoskConnection(args.url)
    all_records: List[DetectionRecord] = []

    for run_i in range(args.runs):
        print(f"\n--- Test run {run_i + 1}/{args.runs} ---")
        q: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=64)
        t1 = time.time()  # Simulate keyword-end time

        # Feed audio into queue in a background thread
        def _feed():
            if args.wav:
                with wave.open(args.wav, "rb") as wf:
                    assert wf.getframerate() == SAMPLE_RATE, (
                        f"WAV must be {SAMPLE_RATE} Hz, got {wf.getframerate()}"
                    )
                    assert wf.getnchannels() == 1, "WAV must be mono"
                    assert wf.getsampwidth() == 2, "WAV must be 16-bit"
                    while True:
                        frames = wf.readframes(CHUNK_FRAMES)
                        if not frames:
                            break
                        q.put(frames)
                        time.sleep(CHUNK_FRAMES / SAMPLE_RATE)  # real-time pacing
            else:
                # Generate 3 seconds of 440 Hz tone (sanity check without a real WAV)
                tone = (
                    np.sin(
                        2 * np.pi * 440 * np.arange(SAMPLE_RATE * 3) / SAMPLE_RATE
                    )
                    * 0.3
                    * 32767
                ).astype(np.int16)
                for start in range(0, len(tone), CHUNK_FRAMES):
                    chunk = tone[start : start + CHUNK_FRAMES].tobytes()
                    q.put(chunk)
                    time.sleep(CHUNK_FRAMES / SAMPLE_RATE)
            q.put(None)  # end sentinel

        feeder = threading.Thread(target=_feed, daemon=True)
        feeder.start()

        rec = stream_utterance(q, conn, t1_keyword_end=t1)
        all_records.append(rec)
        feeder.join()

    print_latency_summary(all_records)
    conn.close()


if __name__ == "__main__":
    if "--test" in sys.argv:
        _test_main()
    else:
        print("Run with --test flag. Example:")
        print("  python3 asr_client.py --test --wav utterance.wav")
        print("  python3 asr_client.py --test  # uses synthetic tone")
