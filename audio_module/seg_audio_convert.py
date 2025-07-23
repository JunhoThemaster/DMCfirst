import os
import numpy as np
import librosa
import parselmouth
import whisper
import json
from collections import Counter

# ========== CONFIG ==========
audio_path = "ckmk_a_bm_f_e_47109.wav"  # 분석 대상 wav
model_size = "medium"
THRESHOLD_SILENCE = 1.0   # 침묵 판단 기준 (초)
TOP_DB = 30               # 음성 감지 기준 dB
MIN_FREQUENT_WORD = 3
MIN_WORD_LENGTH = 2

# ========== LOAD ==========
y, sr = librosa.load(audio_path, sr=None)
duration = librosa.get_duration(y=y, sr=sr)

# ========== TRANSCRIBE ==========
model = whisper.load_model(model_size)
result = model.transcribe(audio_path, language='ko', word_timestamps=True)
transcript = result["text"].strip()

# ========== HABIT WORD DETECTION ==========
words = transcript.replace(".", "").replace(",", "").split()
filtered_words = [w for w in words if len(w) >= MIN_WORD_LENGTH]
word_counts = Counter(filtered_words)
habit_words = {w: c for w, c in word_counts.items() if c >= MIN_FREQUENT_WORD}

# ========== SILENCE GAP DETECTION ==========
non_silence = librosa.effects.split(y, top_db=TOP_DB)
pause_times = []
for i in range(1, len(non_silence)):
    prev_end = non_silence[i - 1][1] / sr
    curr_start = non_silence[i][0] / sr
    if curr_start - prev_end >= THRESHOLD_SILENCE:
        pause_times.append((prev_end, curr_start))

# ========== PITCH PREPARATION ==========
sound = parselmouth.Sound(audio_path)
pitch_obj = sound.to_pitch()
pitch_values_all = pitch_obj.selected_array['frequency']
pitch_times = pitch_obj.xs()

# ========== HELPER ==========
def count_syllables_korean(text):
    return len(text.replace(" ", "").replace(".", "").replace(",", ""))

# ========== SEGMENT ANALYSIS ==========
segments_output = []
for i, seg in enumerate(result.get("segments", [])):
    start = seg["start"]
    end = seg["end"]
    text = seg["text"].strip()

    # pitch 범위 필터링
    pitch_segment = [p for t, p in zip(pitch_times, pitch_values_all) if start <= t <= end and p > 50]
    mean_pitch = float(np.mean(pitch_segment)) if pitch_segment else 0.0
    std_pitch = float(np.std(pitch_segment)) if pitch_segment else 0.0

    # 속도 계산 (음절 수 / 초)
    syllables = count_syllables_korean(text)
    seg_duration = end - start
    speed = round(syllables / seg_duration, 2) if seg_duration > 0 else 0.0

    # 이전 멈춤 여부 판단
    has_pause = any(p[1] <= start <= p[1] + 0.3 for p in pause_times)

    seg_entry = {
        "id": i,
        "seek": seg.get("seek", 0),
        "start": round(start, 2),
        "end": round(end, 2),
        "text": text,
        "tokens": seg.get("tokens", []),
        "temperature": seg.get("temperature", 0.0),
        "avg_logprob": seg.get("avg_logprob", 0.0),
        "compression_ratio": seg.get("compression_ratio", 0.0),
        "no_speech_prob": seg.get("no_speech_prob", 0.0),
        "mean_pitch": round(mean_pitch, 1),
        "std_pitch": round(std_pitch, 1),
        "speed": speed,
        "has_pause_before": has_pause
    }

    segments_output.append(seg_entry)

# ========== SAVE ==========
output_filename = os.path.splitext(os.path.basename(audio_path))[0] + "_segments_pitch.json"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump({"segments": segments_output}, f, ensure_ascii=False, indent=2)

print(f"✅ Whisper segment + pitch 정보가 저장되었습니다:\n{output_filename}")
