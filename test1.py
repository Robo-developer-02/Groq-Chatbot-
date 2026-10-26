"""

this is the code which was been used before the mic's diameter was fixed . 

============================================================
  🤖 Speech-to-Speech AI Chatbot — Powered by Groq (Free)
============================================================
  Stack:
    STT  → Groq Whisper (whisper-large-v3-turbo)
    LLM  → Groq gpt-oss-20b  (streamed)
    TTS  → Microsoft Edge TTS (edge-tts, 100% free)

  Language Support:
    → Speak English → AcroBot replies & speaks in English
    → Speak Hindi   → AcroBot replies & speaks in Hindi
    → Switches instantly every message — no confusion

  Language Detection (3-layer):
    1. Whisper language tag  (fast, sometimes wrong)
    2. Script scan of transcript  (ground truth — never lies)
    3. Default → English

  State Machine:
    IDLE  ──(wake word)──►  LISTENING  ──(speech)──►  SPEAKING
      ▲                          │                        │
      └────────(10s silence)─────┘◄───────────────────────┘

  Wake word: "Hello" (or just "Acrobot")

  Latency pipeline (new):
    LLM tokens ─► sentence chunker ─► TTS synth thread ─► player thread
    The first meaningful chunk is spoken while the rest of the
    reply is still being generated and synthesised.
============================================================
"""

import io
import os
import re
import asyncio
import tempfile
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf
from groq import Groq
import edge_tts
import pygame
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# turbo = faster + cheaper than whisper-large-v3. If Hindi accuracy ever
# drops, set STT_MODEL back to "whisper-large-v3".
STT_MODEL      = "whisper-large-v3-turbo"   # normal conversation
STT_MODEL_WAKE = "whisper-large-v3-turbo"   # wake-word check (short clips)
CHAT_MODEL     = "openai/gpt-oss-20b"

# gpt-oss is a reasoning model; "low" cuts the hidden "thinking" time before
# the first word. Set to None to disable.
REASONING_EFFORT = "low"

TTS_VOICE_HI = "hi-IN-SwaraNeural"  # change to hi-IN-MadhurNeural for male
TTS_VOICE_EN = TTS_VOICE_HI   # change to en-US-GuyNeural for male
TTS_VOICE_UR = "ur-PK-UzmaNeural"   # safety net only: used if a reply ever comes back in Urdu script

SAMPLE_RATE = 16000
CHANNELS    = 1
MAX_TOKENS  = 250            # includes hidden reasoning tokens; raise if replies come back empty
MAX_REPLY_CHUNKS = 3         # safety net: stop speaking after this many chunks (~2-3 sentences)

# ── VAD tuning ─────────────────────────────────────────────
ENERGY_THRESHOLD     = 0.10   # RMS level above which audio counts as "speech"
                             # ↑ raise (e.g. 0.025) if background noise triggers false starts
                               # ↓ lower (e.g. 0.010) if mic is quiet and misses your voice

SILENCE_AFTER_SPEECH = 0.8   # seconds of silence that marks end-of-turn (was 1.2 → faster response)
PRE_ROLL_CHUNKS      = 6      # chunks buffered before speech onset (avoids clipping first word)
MIN_SPEECH_SECS      = 0.5    # discard clips shorter than this (accidental noise / breath)
CHUNK_SECS           = 0.1    # size of each audio chunk in seconds
TRAILING_KEEP_CHUNKS = 3      # trailing silence chunks kept in the clip sent to Whisper

MAX_UTTERANCE_SECS   = 15.0   # hard cap on one question (long clips = slow upload + slow STT)

# Whisper often hears spoken Hindi as "Urdu" and writes it in Arabic script.
# When that happens, re-transcribe once with language="hi" to get Devanagari.
# (costs one extra STT call, only on those turns)
FORCE_DEVANAGARI_HINDI = True

# ── STT network behaviour ──────────────────────────────────
STT_TIMEOUT          = 8.0    # give up on a stuck request after this many seconds (SDK default is 60s!)
STT_MAX_RETRIES      = 1      # then retry once, quickly

IDLE_TIMEOUT         = 60.0   # secs of no speech in LISTENING → go IDLE
IDLE_POLL_TIMEOUT    = 50.0   # how long to wait for audio in IDLE before re-looping

# ── Wake-word tuning (IDLE state only) ─────────────────────
WAKE_SILENCE_SECS    = 0.5    # shorter end-of-speech silence → wake check starts sooner
WAKE_MAX_SECS        = 4.0    # never record more than this while idle (wake word is short)

# ── Streaming TTS chunking ─────────────────────────────────
MIN_CHUNK_CHARS      = 15     # don't speak tiny fragments like "Yes."
FIRST_CLAUSE_MIN     = 20     # first chunk may end at a comma/clause once it's this long
MAX_CHUNK_CHARS      = 100    # force a cut at a space if no punctuation shows up (shorter = faster to synthesise)

# ── TTS robustness / speed ─────────────────────────────────
TTS_WORKERS          = 3      # chunks are synthesised in PARALLEL, played in order
TTS_SYNTH_TIMEOUT    = 6.0    # a stuck Edge-TTS request is abandoned after this many seconds…
TTS_SYNTH_RETRIES    = 1      # …and retried once (instead of leaving silence / skipping the chunk)

# ── Wake words (lowercase, substring match) ────────────────
WAKE_WORDS = [
    "hello", "hey",
    "helo", "hallo", "hullo",                   # common Whisper mis-hearings
    
    "हेलो", "हैलो", "हलो", "हॅलो", "रोबोबॉट",      # Devanagari spellings
    "ہیلو", "ہلو", "ہیلو",                  # Urdu-script spellings (Whisper sometimes does this)
]

# ── Fixed phrases (pre-synthesised at startup → instant playback) ──
WAKE_ACK_TEXT = "Haan, mein sun rahi hoon. Aap apna sawaal poochhiye."
IDLE_MSG_TEXT = "Mein abhi idle mode mein ja rahi hoon. Jab zaroorat ho, 'Hello' kahiye."

SYSTEM_EN = (
    "Your name is Aayro. You are the helpful AI assistant . "
    "Keep responses concise and conversational. "
    "Answer in 1 or 2 short sentences, maximum 30 words, like a quick spoken reply. "
    "Only give more detail if the user explicitly asks for it. "
    "Do not add follow-up offers, introductions or summaries. "
    "No bullet points or markdown. "
    "you are created by Robotwala"
)

SYSTEM_HI = (
    "Aapka naam Aayro hai. Aap helpful AI assistant hain. "
    "Hamesha Devanagari lipi (हिंदी) mein jawab dein. "
    "Kabhi bhi Urdu ya Arabic lipi ka istemal na karein, chahe user ka sawaal kisi bhi lipi mein ho. "
    "Apne uttar chhote aur batcheet ke andaz mein rakhein. "
    "Jawab sirf 1 ya 2 chhote vaakya mein dein, adhiktam 30 shabd. "
    "Zyada vistaar tabhi dein jab user khud maange. "
    "Ant mein koi extra sawaal ya offer na jodein. "
    "Koi bullet points ya markdown nahi. "
    "tumhein robotwala ne banaya hein"
)

# ──────────────────────────────────────────────
#  STATE
# ──────────────────────────────────────────────

class State(Enum):
    IDLE      = "idle"
    LISTENING = "listening"
    SPEAKING  = "speaking"

# ──────────────────────────────────────────────
#  SETUP
# ──────────────────────────────────────────────

client = Groq(api_key=GROQ_API_KEY)

# Separate history per language so the model never sees
# cross-language context and stays in the right language naturally.
history: dict = {
    "en": [],
    "hi": [],
}

pygame.mixer.init()


def warm_connection():
    """
    Fire-and-forget request that keeps the HTTPS connection to Groq alive,
    so the next transcription doesn't pay a fresh TLS handshake.
    Runs in the background while the user is still speaking.
    """
    def _ping():
        try:
            client.models.list()
        except Exception:
            pass
    threading.Thread(target=_ping, daemon=True).start()


# ──────────────────────────────────────────────
#  VAD RECORDING
# ──────────────────────────────────────────────

def capture_speech(
    timeout: float,
    silence_secs: float = SILENCE_AFTER_SPEECH,
    max_secs: Optional[float] = None,
) -> Optional[np.ndarray]:
    """
    Listens via microphone using Voice Activity Detection.

    Returns audio ndarray when:
      • Speech is detected AND then silence >= silence_secs seconds
        (or the recording reaches max_secs, if given).

    Returns None when:
      • No speech detected for `timeout` seconds (caller decides what to do).

    How it works:
      1. Continuously reads 100ms audio chunks into a queue.
      2. Computes RMS energy per chunk.
      3. Above ENERGY_THRESHOLD  → "speech": start/continue recording.
      4. Below threshold after speech began → silence timer starts.
      5. Silence timer expires              → user finished speaking, return audio.
      6. No speech at all for `timeout`     → return None.

    The long trailing silence is trimmed before returning so less audio
    is uploaded to Whisper.
    """
    audio_q   = queue.Queue()
    blocksize = int(SAMPLE_RATE * CHUNK_SECS)

    def callback(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=blocksize,
        callback=callback,
    )
    stream.start()

    speech_buffer: list            = []
    pre_buffer:    list            = []   # rolling window before speech onset
    recording                      = False
    silence_start: Optional[float] = None
    idle_clock                     = time.time()
    record_start:  float           = 0.0
    trailing_silent                = 0    # silent chunks at the end of speech_buffer

    try:
        while True:
            try:
                chunk = audio_q.get(timeout=0.5)
            except queue.Empty:
                # Check idle timeout even when mic is completely silent
                if not recording and time.time() - idle_clock >= timeout:
                    return None
                continue

            # Hard cap on clip length (used for the short wake-word clip)
            if recording and max_secs and time.time() - record_start >= max_secs:
                break

            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if rms >= ENERGY_THRESHOLD:
                # ── Speech detected ──────────────────────────────────
                idle_clock      = time.time()   # reset the no-speech clock
                silence_start   = None
                trailing_silent = 0

                if not recording:
                    recording    = True
                    record_start = time.time()
                    # Prepend pre-roll so the first syllable isn't clipped
                    speech_buffer = list(pre_buffer)

                speech_buffer.append(chunk)

            elif recording:
                # ── Silence after speech has started ─────────────────
                speech_buffer.append(chunk)   # keep trailing silence for natural cut
                trailing_silent += 1
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= silence_secs:
                    break                     # end of user's turn → exit loop

            else:
                # ── Still waiting for speech ─────────────────────────
                pre_buffer.append(chunk)
                if len(pre_buffer) > PRE_ROLL_CHUNKS:
                    pre_buffer.pop(0)

                if time.time() - idle_clock >= timeout:
                    return None               # timed out with no speech

    finally:
        stream.stop()
        stream.close()

    if not speech_buffer:
        return None

    # Length check uses the full clip (same behaviour as before)
    if sum(len(c) for c in speech_buffer) < SAMPLE_RATE * MIN_SPEECH_SECS:
        return None

    # Trim most of the trailing silence → smaller upload, faster STT
    trim = trailing_silent - TRAILING_KEEP_CHUNKS
    if trim > 0:
        speech_buffer = speech_buffer[:-trim]

    return np.concatenate(speech_buffer, axis=0)


# ──────────────────────────────────────────────
#  TRANSCRIBE
# ──────────────────────────────────────────────

def transcribe(
    audio: np.ndarray, model: str = STT_MODEL, fix_script: bool = True
) -> Tuple[str, str]:
    """
    Returns (text, lang_code) where lang_code is 'hi' or 'en'.

    Language detection uses 3 layers in order:
      1. Whisper's detected language tag  (quick but sometimes wrong)
      2. Script scan of the transcribed text  ← THE KEY FIX
         Whisper always transcribes the correct script even when
         its language tag is wrong. Devanagari/Arabic in the text
         means Hindi, period.
      3. Default → 'en'

    Speed-ups: audio is encoded in-memory as 16-bit FLAC (no temp file,
    ~4-6x smaller upload than float32 WAV) and decoding is greedy (temp 0).
    """
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    audio_bytes = buf.getvalue()

    def _call(language: Optional[str] = None):
        kwargs = dict(
            model=model,
            file=("speech.flac", audio_bytes),
            response_format="verbose_json",
            temperature=0.0,
        )
        if language:
            kwargs["language"] = language
        return client.with_options(
            timeout=STT_TIMEOUT, max_retries=STT_MAX_RETRIES
        ).audio.transcriptions.create(**kwargs)

    try:
        result = _call()
    except Exception as e:
        print(f"⚠️  STT failed/timed out: {e}")
        return "", "en"

    text = (result.text or "").strip()
    lang = (result.language or "en").strip().lower()

    # Hindi heard as Urdu → Arabic-script transcript. Redo it as Hindi (Devanagari).
    if (
        fix_script
        and FORCE_DEVANAGARI_HINDI
        and (lang in ("ur", "urdu") or any(0x0600 <= ord(c) <= 0x06FF for c in text))
    ):
        try:
            retry      = _call(language="hi")
            retry_text = (retry.text or "").strip()
            if retry_text:
                text, lang = retry_text, "hi"
        except Exception as e:
            print(f"⚠️  Hindi re-transcribe failed (using first result): {e}")

    # Layer 1: normalise Whisper tag
    if lang in ("ur", "urdu"):
        lang = "hi"
    if lang not in ("hi", "en"):
        lang = "en"

    # Layer 2: script scan — overrides the tag if script is Hindi
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:   # Devanagari script
            lang = "hi"
            break
        if 0x0600 <= cp <= 0x06FF:   # Arabic / Urdu script
            lang = "hi"
            break

    return text, lang


# ──────────────────────────────────────────────
#  WAKE WORD
# ──────────────────────────────────────────────

def is_wake_word(text: str) -> bool:
    lower = text.lower().strip()
    return any(w in lower for w in WAKE_WORDS)


# ──────────────────────────────────────────────
#  TEXT CLEAN-UP FOR TTS
# ──────────────────────────────────────────────

# Every kind of hyphen / dash (incl. the non-breaking hyphen U+2011 that the
# LLM loves to emit, e.g. "self‑contained") → space.  The Hindi neural voice
# does not know these characters and spells the whole word letter by letter.
_DASH_CHARS = "-\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uFE58\uFE63\uFF0D"

_TTS_TRANSLATE = {ord(c): " " for c in _DASH_CHARS}
_TTS_TRANSLATE.update({
    0x00AD: None,   # soft hyphen
    0x200B: None,   # zero-width space
    0x200C: None,   # zero-width non-joiner
    0x200D: None,   # zero-width joiner
    0xFEFF: None,   # BOM / zero-width no-break space
    0x00A0: " ",    # non-breaking space
    0x202F: " ",    # narrow no-break space
    0x2018: "'", 0x2019: "'",    # curly single quotes
    0x201C: '"', 0x201D: '"',    # curly double quotes
    0x2026: ".",                 # ellipsis
})

_MARKDOWN_RE = re.compile(r"[*_#`~>|\[\]{}<>^\\]")
_EMOJI_RE    = re.compile("[\U0001F000-\U0001FFFF\u2600-\u27BF\uFE0F]")


def clean_for_tts(text: str) -> str:
    text = text.translate(_TTS_TRANSLATE)
    text = _MARKDOWN_RE.sub(" ", text)
    text = _EMOJI_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


# ──────────────────────────────────────────────
#  SENTENCE CHUNKER (for streaming TTS)
# ──────────────────────────────────────────────

class SentenceChunker:
    """
    Collects streamed LLM tokens and releases speakable chunks:
      • the FIRST chunk may end at a comma/clause so speech starts ASAP
      • later chunks end at . ! ? । or a newline
      • very long run-ons are cut at a space
    Decimal numbers like 3.5 are never split (terminator must be followed
    by whitespace).
    """

    def __init__(self):
        self.buf   = ""
        self.first = True

    def _cut_index(self) -> Optional[int]:
        buf = self.buf
        n   = len(buf)
        for i, ch in enumerate(buf):
            if i + 1 >= n:
                break                         # need one char of look-ahead
            nxt = buf[i + 1]
            if ch == "\n":
                if i + 1 >= MIN_CHUNK_CHARS:
                    return i + 1
            elif ch in ".!?।۔؟" and nxt.isspace():
                if i + 1 >= MIN_CHUNK_CHARS:
                    return i + 1
            elif self.first and ch in ",;:" and nxt.isspace():
                if i + 1 >= FIRST_CLAUSE_MIN:
                    return i + 1
        if n >= MAX_CHUNK_CHARS:
            sp = buf.rfind(" ", 0, MAX_CHUNK_CHARS)
            if sp > 0:
                return sp
        return None

    def feed(self, token: str) -> List[str]:
        self.buf += token
        out: List[str] = []
        while True:
            idx = self._cut_index()
            if idx is None:
                break
            chunk    = self.buf[:idx].strip()
            self.buf = self.buf[idx:]
            if chunk:
                out.append(chunk)
                self.first = False
        return out

    def flush(self) -> str:
        rest, self.buf = self.buf.strip(), ""
        return rest


# ──────────────────────────────────────────────
#  AI REPLY  (streamed)
# ──────────────────────────────────────────────

def get_ai_reply(user_text: str, lang: str, on_sentence=None) -> str:
    """
    Streams the reply from the LLM. Every time a speakable chunk is ready,
    `on_sentence(chunk)` is called immediately (while the LLM keeps
    generating). Returns the full reply text.
    """
    system       = SYSTEM_HI if lang == "hi" else SYSTEM_EN
    lang_history = history[lang]

    lang_history.append({"role": "user", "content": user_text})

    kwargs = dict(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            *lang_history,
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.7,
        stream=True,
    )
    if REASONING_EFFORT:
        kwargs["extra_body"] = {"reasoning_effort": REASONING_EFFORT}

    chunker = SentenceChunker()
    parts: List[str] = []
    spoken_chunks: List[str] = []
    capped   = False
    t0       = time.time()
    got_first = False

    try:
        stream = client.chat.completions.create(**kwargs)
        for event in stream:
            if not event.choices:
                continue
            token = event.choices[0].delta.content
            if not token:
                continue

            if not got_first:
                got_first = True
                print(f"   ⚡ first token in {time.time() - t0:.2f}s")

            parts.append(token)
            for sentence in chunker.feed(token):
                spoken_chunks.append(sentence)
                if on_sentence:
                    on_sentence(sentence)
                if len(spoken_chunks) >= MAX_REPLY_CHUNKS:
                    capped = True
                    break
            if capped:
                break
        if capped:
            try:
                stream.close()          # stop generating tokens we won't use
            except Exception:
                pass
    except Exception as e:
        print(f"⚠️  LLM stream error: {e}")

    if not capped:
        tail = chunker.flush()
        if tail:
            spoken_chunks.append(tail)
            if on_sentence:
                on_sentence(tail)

    # What the user actually heard is what goes into history
    reply = " ".join(spoken_chunks).strip() if capped else "".join(parts).strip()
    if reply:
        lang_history.append({"role": "assistant", "content": reply})
    else:
        lang_history.pop()      # nothing came back — drop the unanswered user turn
    return reply


# ──────────────────────────────────────────────
#  VOICE SELECTION
# ──────────────────────────────────────────────

def pick_voice(text: str, lang: str) -> str:
    # Arabic/Urdu script can't be read by the Hindi voice ("No audio was received")
    if any(0x0600 <= ord(c) <= 0x06FF for c in text):
        return TTS_VOICE_UR

    if lang == "hi":
        return TTS_VOICE_HI

    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            return TTS_VOICE_HI
        if 0x0600 <= cp <= 0x06FF:
            return TTS_VOICE_HI

    return TTS_VOICE_EN


# ──────────────────────────────────────────────
#  SPEAK
# ──────────────────────────────────────────────

async def _tts(text: str, path: str, voice: str):
    await asyncio.wait_for(
        edge_tts.Communicate(text, voice=voice).save(path),
        timeout=TTS_SYNTH_TIMEOUT,
    )


def _synth_to_file(spoken: str, voice: str) -> str:
    """
    Synthesises already-cleaned text to a temp mp3 and returns its path.
    A slow / failed / empty attempt is retried instead of stalling or being lost.
    """
    last_err: Exception = RuntimeError("TTS failed")
    for _ in range(TTS_SYNTH_RETRIES + 1):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            asyncio.run(_tts(spoken, tmp_path, voice))
            if os.path.getsize(tmp_path) > 0:
                return tmp_path
            last_err = RuntimeError("empty audio file")
        except Exception as e:
            last_err = e
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    raise last_err


def _play_file(path: str):
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(30)
    pygame.mixer.music.unload()


# Fixed phrases are synthesised once and re-used (never deleted until exit)
_phrase_cache: dict = {}


def speak(text: str, lang: str = "en", cache: bool = False):
    spoken = clean_for_tts(text)
    if not any(c.isalnum() for c in spoken):
        return
    voice = pick_voice(spoken, lang)
    print(f"   🔊 Voice → {voice}")

    key  = (spoken, voice)
    path = _phrase_cache.get(key) if cache else None
    if path is None or not os.path.exists(path):
        path = _synth_to_file(spoken, voice)
        if cache:
            _phrase_cache[key] = path

    _play_file(path)

    if not cache:
        os.unlink(path)


def prewarm_phrases():
    """Background job: pre-synthesise the fixed phrases so they play instantly."""
    for phrase in (WAKE_ACK_TEXT, IDLE_MSG_TEXT):
        try:
            spoken = clean_for_tts(phrase)
            voice  = pick_voice(spoken, "hi")
            _phrase_cache[(spoken, voice)] = _synth_to_file(spoken, voice)
        except Exception as e:
            print(f"⚠️  Prewarm failed for a phrase: {e}")


# ── Streaming speech pipeline ──────────────────────────────────
#   every chunk is sent to a thread pool the moment the LLM finishes it,
#   so chunk 2, 3… are synthesised WHILE chunk 1 is still playing.
#   The player takes the finished results strictly in order.

_STOP = object()


def _synth_job(sentence: str, lang: str) -> Optional[str]:
    spoken = clean_for_tts(sentence)
    if not any(c.isalnum() for c in spoken):
        return None
    return _synth_to_file(spoken, pick_voice(spoken, lang))


def _play_worker(order_q: "queue.Queue"):
    """Plays chunks back to back, in order, as soon as each one is ready."""
    while True:
        fut = order_q.get()
        if fut is _STOP:
            return

        t0 = time.time()
        try:
            path = fut.result()
        except Exception as e:
            print(f"⚠️  TTS error (chunk skipped): {e}")
            continue
        if path is None:
            continue

        waited = time.time() - t0
        if waited > 0.3:
            print(f"   ⏳ waited {waited:.1f}s for audio")

        try:
            _play_file(path)
        except Exception as e:
            print(f"⚠️  Playback error: {e}")
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


def respond_streaming(user_text: str, lang: str) -> str:
    """
    LLM streaming + TTS + playback running concurrently.
    Speech begins as soon as the first meaningful chunk exists, and later
    chunks are already synthesised by the time the previous one finishes.
    """
    order_q: queue.Queue = queue.Queue()
    pool   = ThreadPoolExecutor(max_workers=TTS_WORKERS)
    player = threading.Thread(target=_play_worker, args=(order_q,), daemon=True)
    player.start()

    def on_sentence(sentence: str):
        print(f"   AI  [{lang.upper()}] › {sentence}")
        order_q.put(pool.submit(_synth_job, sentence, lang))

    try:
        reply = get_ai_reply(user_text, lang, on_sentence=on_sentence)
    finally:
        order_q.put(_STOP)

    player.join()
    pool.shutdown(wait=False)
    return reply


# ──────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────

def print_banner():
    print("\n" + "=" * 56)
    print("=" * 56)
    print("  States:")
    print("    👂 LISTENING  — auto-detects your voice")
    print(f"    😴 IDLE       — {int(IDLE_TIMEOUT)}s silence → idle")
    print("                   say 'Hello' to wake up")
    print("    🔊 SPEAKING   — playing response; loops back after")
    print("  Ctrl+C to quit")
    print("=" * 56 + "\n")


def state_label(state: State) -> str:
    return {
        State.IDLE:      "😴 IDLE",
        State.LISTENING: "👂 LISTENING",
        State.SPEAKING:  "🔊 SPEAKING",
    }[state]


# ──────────────────────────────────────────────
#  MAIN LOOP
# ──────────────────────────────────────────────

def main():
    print_banner()

    # Pre-generate fixed phrases in the background while the greeting plays
    threading.Thread(target=prewarm_phrases, daemon=True).start()

    state     = State.LISTENING   # start directly in LISTENING after the greeting
    user_text = ""
    lang      = "hi"

    # ── Opening greeting ──────────────────────────────────────
    speak(
        "Hello, mein Aayro hoon. "
        "Mein aapki kese madad kar sakti hoon? "
        "Krapya apna sawaal poochhiye.",
        lang="hi",
    )

    try:
        while True:

            # ════════════════════════════════════════════════════
            #  IDLE — wait silently for wake word
            # ════════════════════════════════════════════════════
            if state == State.IDLE:
                print(f"\n{state_label(state)}  — say 'Hello' to activate...")

                # Short end-silence + hard cap → the wake clip is small and
                # reaches Whisper quickly.
                warm_connection()
                audio = capture_speech(
                    timeout=IDLE_POLL_TIMEOUT,
                    silence_secs=WAKE_SILENCE_SECS,
                    max_secs=WAKE_MAX_SECS,
                )

                if audio is None:
                    # Nobody spoke for IDLE_POLL_TIMEOUT — keep idling silently
                    continue

                print("🔍 Checking for wake word...")
                t0 = time.time()
                wake_text, _ = transcribe(audio, model=STT_MODEL_WAKE, fix_script=False)
                print(f"   Heard: {wake_text!r}  ({time.time() - t0:.2f}s)")

                if is_wake_word(wake_text):
                    state = State.LISTENING
                    print("\n✅ Wake word detected!")
                    speak(WAKE_ACK_TEXT, lang="hi", cache=True)
                else:
                    print("   Not a wake word — staying idle.")

                continue

            # ════════════════════════════════════════════════════
            #  LISTENING — auto VAD; 10 s of silence → IDLE
            # ════════════════════════════════════════════════════
            if state == State.LISTENING:
                print(f"\n{state_label(state)}  "
                      f"— silence for {int(IDLE_TIMEOUT)}s → idle")

                warm_connection()
                audio = capture_speech(timeout=IDLE_TIMEOUT, max_secs=MAX_UTTERANCE_SECS)

                if audio is None:
                    # 10 seconds of nothing — go idle
                    state = State.IDLE
                    print(f"\n⏱️  No speech for {int(IDLE_TIMEOUT)}s — going idle.")
                    speak(IDLE_MSG_TEXT, lang="hi", cache=True)
                    continue

                # ── Got speech — transcribe ────────────────────
                print("🔍 Transcribing...")
                t0 = time.time()
                user_text, lang = transcribe(audio)
                print(f"   (STT: {len(audio) / SAMPLE_RATE:.1f}s audio → {time.time() - t0:.2f}s)")

                if not user_text:
                    print("⚠️  Could not understand — listening again.")
                    continue

                print(f"   You [{lang.upper()}] › {user_text}")

                state = State.SPEAKING
                continue

            # ════════════════════════════════════════════════════
            #  SPEAKING — stream reply while speaking it,
            #             then return to LISTENING
            # ════════════════════════════════════════════════════
            if state == State.SPEAKING:
                print(f"\n{state_label(state)}")
                print("🤔 Thinking...")
                respond_streaming(user_text, lang)
                state = State.LISTENING
                continue

    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        


if __name__ == "__main__":
    main()