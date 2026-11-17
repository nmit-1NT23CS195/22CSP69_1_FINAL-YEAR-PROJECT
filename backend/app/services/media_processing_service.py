"""
app/services/media_processing_service.py
=========================================
Video-Based Mock Interview — Media Processing Engine (Module D)

RESPONSIBILITIES
----------------
1. Audio Extraction via ffmpeg subprocess (16kHz mono WAV)
2. Vision Quality Checks via OpenCV (brightness, Laplacian blur)
3. Speech-to-Text Transcription via faster-whisper (CTranslate2)
4. Acoustic Metrics via librosa (WPM, filler words, pause detection)
5. Unified `process_video(video_path)` returning a consolidated dict

DESIGN RULES
------------
- WhisperModel is initialised ONCE at module load (singleton) so the model
  weights are loaded into memory only when the FastAPI server starts, not
  on every request.
- ffmpeg is invoked via subprocess — the binary must be on PATH.
- CPU-bound operations are run synchronously; callers must use asyncio.to_thread.
- Every step wraps exceptions individually so a single failure returns partial
  results instead of crashing the entire evaluation.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from typing import Any

# Suppress HuggingFace symlink cache warnings on Windows before any HF import
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SAMPLE_RATE_HZ: int = 16_000           # Whisper target sample rate
_FRAMES_SAMPLED_PER_SEC: int = 1        # Vision: frames to sample per second
_BRIGHTNESS_THRESHOLD: float = 40.0     # Below → too dark
_BLUR_THRESHOLD: float = 80.0           # Below → blurry (Laplacian variance)
_WHISPER_MODEL_SIZE: str = "tiny.en"    # tiny.en: fast CPU inference, English-only
_FILLER_WORDS: frozenset[str] = frozenset({
    "um", "uh", "like", "you know", "so", "basically",
    "actually", "literally", "right", "i mean",
})


# ---------------------------------------------------------------------------
# Singleton Whisper model — loaded ONCE at server startup, reused per request.
# Using "tiny.en" (English-only) with int8 quantisation for fast CPU inference.
# ---------------------------------------------------------------------------
_whisper_model = None  # type: ignore[var-annotated]

def _get_whisper_model():
    """Lazy singleton: initialises the model on first call, then caches it."""
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel  # type: ignore
            logger.info(
                "[MediaProcessor] Loading WhisperModel('%s') into memory (one-time)…",
                _WHISPER_MODEL_SIZE,
            )
            _whisper_model = WhisperModel(
                _WHISPER_MODEL_SIZE,
                device="cpu",
                compute_type="int8",    # int8 quantisation for fast CPU inference
            )
            logger.info("[MediaProcessor] WhisperModel ready.")
        except ImportError:
            logger.warning(
                "[MediaProcessor] faster-whisper not installed — transcription disabled."
            )
            _whisper_model = None
    return _whisper_model


class MediaProcessorError(Exception):
    """Raised when a critical (unrecoverable) step fails."""


class MediaProcessor:
    """
    Encapsulates all local media processing for one interview video.

    Usage (from async context):
        result = await asyncio.to_thread(MediaProcessor().process_video, video_path)
    """

    # ── 1. Audio Extraction ───────────────────────────────────────────────────

    def extract_audio(self, video_path: str, wav_path: str) -> None:
        """
        Use ffmpeg to extract 16kHz mono PCM WAV from the video.

        Parameters
        ----------
        video_path : Absolute path to the input video file (.webm/.mp4/.mov).
        wav_path   : Absolute path where the output .wav should be written.

        Raises
        ------
        MediaProcessorError : If ffmpeg is not on PATH or extraction fails.
        """
        cmd = [
            "ffmpeg",
            "-y",                       # overwrite output without asking
            "-i", video_path,
            "-vn",                      # no video stream in output
            "-acodec", "pcm_s16le",     # 16-bit signed PCM
            "-ar", str(_SAMPLE_RATE_HZ),
            "-ac", "1",                 # mono
            wav_path,
        ]
        logger.info("[MediaProcessor] Running ffmpeg: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,            # 2-minute hard limit per video
            )
            if result.returncode != 0:
                raise MediaProcessorError(
                    f"ffmpeg exited with code {result.returncode}. "
                    f"stderr: {result.stderr[:500]}"
                )
            logger.info("[MediaProcessor] Audio extracted → %s", wav_path)
        except FileNotFoundError:
            raise MediaProcessorError(
                "ffmpeg is not installed or not on PATH. "
                "Install via: https://ffmpeg.org/download.html"
            )
        except subprocess.TimeoutExpired:
            raise MediaProcessorError("ffmpeg timed out after 120 seconds.")

    # ── 2. Vision Quality Checks ──────────────────────────────────────────────

    def check_video_quality(self, video_path: str) -> dict[str, Any]:
        """
        Read the video and sample one frame per second.

        Returns
        -------
        dict with keys:
            is_video_clear  (bool)  : True when avg brightness AND avg blur pass thresholds.
            avg_brightness  (float) : Mean pixel intensity across sampled frames (0–255).
            avg_blur_score  (float) : Mean Laplacian variance (higher = sharper).
            frames_sampled  (int)   : Number of frames analysed.
            vision_error    (str|None): Error message if OpenCV failed.
        """
        result: dict[str, Any] = {
            "is_video_clear": True,
            "avg_brightness": 0.0,
            "avg_blur_score": 0.0,
            "frames_sampled": 0,
            "vision_error": None,
        }

        try:
            import cv2  # type: ignore  # lazy import — heavy dependency
            import numpy as np  # type: ignore

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                result["vision_error"] = f"OpenCV could not open video: {video_path}"
                result["is_video_clear"] = False
                return result

            fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_interval: int = max(1, int(fps / _FRAMES_SAMPLED_PER_SEC))

            brightness_vals: list[float] = []
            blur_vals: list[float] = []
            frame_idx: int = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_interval == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    brightness_vals.append(float(np.mean(gray)))
                    blur_vals.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
                frame_idx += 1

            cap.release()

            if not brightness_vals:
                result["vision_error"] = "No frames could be read from the video."
                result["is_video_clear"] = False
                return result

            avg_brightness = float(np.mean(brightness_vals))
            avg_blur = float(np.mean(blur_vals))

            result["avg_brightness"] = round(avg_brightness, 2)
            result["avg_blur_score"] = round(avg_blur, 2)
            result["frames_sampled"] = len(brightness_vals)
            result["is_video_clear"] = (
                avg_brightness >= _BRIGHTNESS_THRESHOLD
                and avg_blur >= _BLUR_THRESHOLD
            )

            logger.info(
                "[MediaProcessor] Vision — brightness=%.2f blur=%.2f clear=%s",
                avg_brightness, avg_blur, result["is_video_clear"],
            )

        except ImportError:
            result["vision_error"] = (
                "opencv-python not installed. "
                "Run: pip install opencv-python-headless"
            )
            logger.warning("[MediaProcessor] OpenCV not available — skipping vision checks.")
        except Exception as exc:
            result["vision_error"] = str(exc)
            logger.error("[MediaProcessor] Vision check failed: %s", exc, exc_info=True)

        return result

    # ── 3. Speech Transcription (Faster-Whisper) ──────────────────────────────

    def transcribe_audio(self, wav_path: str) -> dict[str, Any]:
        """
        Transcribe a WAV file using faster-whisper.

        Returns
        -------
        dict with keys:
            transcript      (str)       : Full transcription text.
            segments        (list[dict]): Per-segment start/end/text dicts.
            language        (str)       : Detected language code.
            transcribe_error(str|None)  : Error message if transcription failed.
        """
        result: dict[str, Any] = {
            "transcript": "",
            "segments": [],
            "language": "en",
            "transcribe_error": None,
        }

        try:
            model = _get_whisper_model()
            if model is None:
                result["transcribe_error"] = (
                    "faster-whisper not installed. "
                    "Run: pip install faster-whisper"
                )
                logger.warning("[MediaProcessor] faster-whisper not available — transcription skipped.")
                return result

            logger.info(
                "[MediaProcessor] Transcribing %s with singleton WhisperModel('%s')…",
                wav_path, _WHISPER_MODEL_SIZE,
            )

            segments_iter, info = model.transcribe(
                wav_path,
                beam_size=5,
                language="en",
                vad_filter=True,        # skip silence via VAD
                vad_parameters={"min_silence_duration_ms": 500},
            )

            segments_list: list[dict] = []
            full_text_parts: list[str] = []

            for seg in segments_iter:
                segments_list.append({
                    "start": round(seg.start, 2),
                    "end":   round(seg.end, 2),
                    "text":  seg.text.strip(),
                })
                full_text_parts.append(seg.text.strip())

            result["transcript"] = " ".join(full_text_parts).strip()
            result["segments"]   = segments_list
            result["language"]   = info.language

            logger.info(
                "[MediaProcessor] Transcribed %d segment(s), lang=%s, chars=%d",
                len(segments_list), info.language, len(result["transcript"]),
            )

        except Exception as exc:
            result["transcribe_error"] = str(exc)
            logger.error(
                "[MediaProcessor] Transcription failed for %s: %s",
                wav_path, exc, exc_info=True,
            )

        return result

    # ── 4. Acoustic & Speech Metrics ─────────────────────────────────────────

    def compute_acoustic_metrics(
        self,
        wav_path: str,
        transcript: str,
        segments: list[dict],
    ) -> dict[str, Any]:
        """
        Compute WPM, pause statistics, and filler word density.

        Uses librosa for total audio duration measurement.

        Returns
        -------
        dict with keys:
            wpm                 (float) : Words per minute.
            total_words         (int)   : Word count in transcript.
            audio_duration_secs (float) : Total audio duration.
            speaking_time_secs  (float) : Sum of spoken segment durations.
            pause_count         (int)   : Number of silences > 0.5s.
            total_pause_secs    (float) : Cumulative pause time.
            filler_word_count   (int)   : Occurrences of common filler words.
            filler_density      (float) : Filler words per 100 words.
            acoustic_error      (str|None)
        """
        result: dict[str, Any] = {
            "wpm": 0.0,
            "total_words": 0,
            "audio_duration_secs": 0.0,
            "speaking_time_secs": 0.0,
            "pause_count": 0,
            "total_pause_secs": 0.0,
            "filler_word_count": 0,
            "filler_density": 0.0,
            "acoustic_error": None,
        }

        # ── Word count + filler words from transcript ─────────────────────────
        if transcript:
            words = transcript.lower().split()
            result["total_words"] = len(words)

            joined = " ".join(words)
            filler_count = sum(
                len(re.findall(r"\b" + re.escape(fw) + r"\b", joined))
                for fw in _FILLER_WORDS
            )
            result["filler_word_count"] = filler_count

        # ── Audio duration via librosa ────────────────────────────────────────
        try:
            import librosa  # type: ignore  # lazy import

            duration = librosa.get_duration(path=wav_path)
            result["audio_duration_secs"] = round(duration, 2)
            logger.info("[MediaProcessor] librosa duration=%.2fs", duration)

        except ImportError:
            result["acoustic_error"] = "librosa not installed (pip install librosa)"
            logger.warning("[MediaProcessor] librosa not available — using segment fallback.")
            if segments:
                duration = segments[-1]["end"]
                result["audio_duration_secs"] = round(duration, 2)
        except Exception as exc:
            result["acoustic_error"] = str(exc)
            logger.error("[MediaProcessor] librosa duration failed: %s", exc)
            if segments:
                duration = segments[-1]["end"]
                result["audio_duration_secs"] = round(duration, 2)

        duration = result["audio_duration_secs"]

        # ── Pause analysis from segment gaps ─────────────────────────────────
        pause_threshold_secs = 0.5
        speaking_time = 0.0
        pause_count = 0
        total_pause = 0.0

        for i, seg in enumerate(segments):
            speaking_time += seg["end"] - seg["start"]
            if i > 0:
                gap = seg["start"] - segments[i - 1]["end"]
                if gap >= pause_threshold_secs:
                    pause_count += 1
                    total_pause += gap

        result["speaking_time_secs"] = round(speaking_time, 2)
        result["pause_count"] = pause_count
        result["total_pause_secs"] = round(total_pause, 2)

        # ── WPM computation ───────────────────────────────────────────────────
        effective_speaking_mins = speaking_time / 60.0
        total_words = result["total_words"]
        if effective_speaking_mins > 0 and total_words > 0:
            result["wpm"] = round(total_words / effective_speaking_mins, 1)

        # ── Filler density ────────────────────────────────────────────────────
        if total_words > 0:
            result["filler_density"] = round(
                (result["filler_word_count"] / total_words) * 100, 2
            )

        logger.info(
            "[MediaProcessor] Acoustics — wpm=%.1f pauses=%d total_pause=%.1fs fillers=%d",
            result["wpm"],
            result["pause_count"],
            result["total_pause_secs"],
            result["filler_word_count"],
        )
        return result

    # ── 5. Unified Entry Point ────────────────────────────────────────────────

    def process_video(self, video_path: str) -> dict[str, Any]:
        """
        Run the full media processing pipeline on a single video file.

        Steps
        -----
        1. Create a sibling temp WAV file.
        2. Extract audio (ffmpeg).
        3. Run vision quality checks (OpenCV).
        4. Transcribe audio (faster-whisper).
        5. Compute acoustic metrics (librosa).
        6. Delete the WAV file in a finally block.
        7. Return a unified result dict.

        This method is SYNCHRONOUS and CPU-bound.
        Callers MUST use ``asyncio.to_thread`` or a ThreadPoolExecutor.

        Parameters
        ----------
        video_path : Absolute path to the uploaded video file.

        Returns
        -------
        dict — always returned; individual step failures are captured as error
               keys and do not abort subsequent steps.
        """
        logger.info("[MediaProcessor] Starting pipeline for: %s", video_path)

        # Create a temporary WAV file next to (or alongside) the video
        wav_fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="placebuddy_audio_")
        os.close(wav_fd)  # close the fd; ffmpeg will write to the path

        pipeline_result: dict[str, Any] = {
            "video_path":   video_path,
            "wav_path":     wav_path,
            "audio_ok":     False,
            "transcript":   "",
            "segments":     [],
            "language":     "en",
            "vision":       {},
            "acoustics":    {},
            "errors":       [],
        }

        try:
            # ── Step 1: Extract audio ─────────────────────────────────────────
            try:
                self.extract_audio(video_path, wav_path)
                pipeline_result["audio_ok"] = True
            except MediaProcessorError as exc:
                pipeline_result["errors"].append(f"audio_extraction: {exc}")
                logger.error("[MediaProcessor] Audio extraction failed: %s", exc)
                # Cannot proceed without audio — vision check still possible
                # Transcription and acoustics will return empty / error results

            # ── Step 2: Vision quality check ──────────────────────────────────
            try:
                vision = self.check_video_quality(video_path)
                pipeline_result["vision"] = vision
                if vision.get("vision_error"):
                    pipeline_result["errors"].append(f"vision: {vision['vision_error']}")
            except Exception as exc:
                pipeline_result["errors"].append(f"vision: {exc}")
                pipeline_result["vision"] = {"is_video_clear": True, "vision_error": str(exc)}
                logger.error("[MediaProcessor] Vision check threw: %s", exc)

            # ── Step 3: Transcription (only if audio extracted successfully) ──
            transcription: dict[str, Any] = {
                "transcript": "",
                "segments": [],
                "language": "en",
                "transcribe_error": None,
            }
            if pipeline_result["audio_ok"]:
                try:
                    transcription = self.transcribe_audio(wav_path)
                    if transcription.get("transcribe_error"):
                        pipeline_result["errors"].append(
                            f"transcription: {transcription['transcribe_error']}"
                        )
                except Exception as exc:
                    transcription["transcribe_error"] = str(exc)
                    pipeline_result["errors"].append(f"transcription: {exc}")
                    logger.error("[MediaProcessor] Transcription threw: %s", exc)

            pipeline_result["transcript"] = transcription["transcript"]
            pipeline_result["segments"]   = transcription["segments"]
            pipeline_result["language"]   = transcription["language"]

            # ── Step 4: Acoustic metrics ──────────────────────────────────────
            try:
                acoustics = self.compute_acoustic_metrics(
                    wav_path=wav_path,
                    transcript=transcription["transcript"],
                    segments=transcription["segments"],
                )
                pipeline_result["acoustics"] = acoustics
                if acoustics.get("acoustic_error"):
                    pipeline_result["errors"].append(
                        f"acoustics: {acoustics['acoustic_error']}"
                    )
            except Exception as exc:
                pipeline_result["errors"].append(f"acoustics: {exc}")
                pipeline_result["acoustics"] = {"acoustic_error": str(exc)}
                logger.error("[MediaProcessor] Acoustics threw: %s", exc)

        finally:
            # ── Cleanup: always delete the temp WAV ──────────────────────────
            if os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                    logger.info("[MediaProcessor] Temp WAV deleted: %s", wav_path)
                except OSError as exc:
                    logger.warning(
                        "[MediaProcessor] Could not delete temp WAV %s: %s",
                        wav_path, exc,
                    )

        logger.info(
            "[MediaProcessor] Pipeline complete — transcript_len=%d errors=%d",
            len(pipeline_result["transcript"]),
            len(pipeline_result["errors"]),
        )
        return pipeline_result
