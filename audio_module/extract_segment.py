import os
import numpy as np
import librosa
import pandas as pd

# 피처 추출 함수
def extract_features(filepath):
    y, sr = librosa.load(filepath, sr=None)  # 원래 sampling rate 유지
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_series = [np.mean(p[p > 0]) if np.any(p > 0) else 0 for p in pitches.T]
    energy_series = librosa.feature.rms(y=y)[0]

    return {
        "filename": os.path.basename(filepath),
        "avg_pitch": float(np.mean(pitch_series)),
        "pitch_std": float(np.std(pitch_series)),
        "pitch_slope": float(np.polyfit(range(len(pitch_series)), pitch_series, 1)[0]),

        "avg_energy": float(np.mean(energy_series)),
        "energy_std": float(np.std(energy_series)),
        "energy_slope": float(np.polyfit(range(len(energy_series)), energy_series, 1)[0]),
    }

# 폴더 내 모든 wav 파일 처리 - "ckmk_a"로 시작하는 파일만
def process_audio_folder(folder_path):
    data = []

    for fname in os.listdir(folder_path):
        if fname.endswith(".wav") and fname.startswith("ckmk_a"):
            fpath = os.path.join(folder_path, fname)
            try:
                features = extract_features(fpath)
                data.append(features)
            except Exception as e:
                print(f"⚠️ {fname} 처리 중 오류: {e}")

    return pd.DataFrame(data)

# 예시 실행
folder = r"C:\Users\main\Documents\emotionCLF\Mock\01.Management\Female\Experienced"
df = process_audio_folder(folder)

df.to_csv("data/calculated_wav_feature.csv",index=False)
