import os
import numpy as np
import librosa
import parselmouth
import whisper
import json

# ========== CONFIG ==========
audio_path = "ckmk_a_bm_f_e_47109.wav"  # 분석 대상 wav
model_size = "medium"     # whisper 모델 크기 (모델 을 낮출지 고려대상)
THRESHOLD_SILENCE = 1.0   # 침묵 판단 기준 (초)
TOP_DB = 30               # 음성 감지 기준 dB 30dB아래 소리는 침묵/잡음으로 판단

# ========== LOAD ==========
y, sr = librosa.load(audio_path, sr=None)   # y에 오디오파일의 전체시간 동안의 진폭데이터를 저장 (1차원 배열로 이루어져잇음 )  ,sr은 samplelingrate 즉 1초동안 몇번 측정햇냐 (Hz단위)
duration = librosa.get_duration(y=y, sr=sr)  #오디오 파일의 전체시간 (실수형)  len(y) / sr 로 계산 

# ========== TRANSCRIBE ==========
model = whisper.load_model(model_size)   #whisper 모델 
result = model.transcribe(audio_path, language='ko', word_timestamps=True)

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
        "mean_pitch": round(mean_pitch, 1),
        "std_pitch": round(std_pitch, 1),
        "speed": speed,
        "has_pause_before": has_pause
    }

    segments_output.append(seg_entry)

# ========== SAVE ==========
output_filename = os.path.splitext(os.path.basename(audio_path))[0] + "_sound_only_segments.json"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump({"segments": segments_output}, f, ensure_ascii=False, indent=2)

print(f"✅ 음성 기반 피처만 추출 완료:\n{output_filename}")
