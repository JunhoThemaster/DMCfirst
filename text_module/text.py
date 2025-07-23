import pandas as pd

import json

import os
import json
import pandas as pd

base_dir = r"TL_01.Management/"
all_records = []


for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".json"):
            json_path = os.path.join(root, file)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                info = data["dataSet"]["info"]
                question = data["dataSet"]["question"]
                answer = data["dataSet"]["answer"]
                raw_info = data["rawDataInfo"]

                # Emotion/Intent 안전하게 꺼내기
                emotion_list = answer.get("emotion", [])
                intent_list = answer.get("intent", [])

                first_emotion = emotion_list[0] if len(emotion_list) > 0 else {}
                first_intent = intent_list[0] if len(intent_list) > 0 else {}

                record = {
                    "date": info["date"],
                    "occupation": info["occupation"],
                    "channel": info["channel"],
                    "place": info["place"],
                    "gender": info["gender"],
                    "ageRange": info["ageRange"],
                    "experience": info["experience"],

                    "question_text": question["raw"]["text"],
                    "question_word_count": question["raw"]["wordCount"],

                    "answer_text": answer["raw"]["text"],
                    "answer_word_count": answer["raw"]["wordCount"],

                    "summary": answer["summary"]["text"],

                    "emotion_expression": first_emotion.get("expression", None),
                    "emotion_category": first_emotion.get("category", None),

                    "intent_expression": first_intent.get("expression", None),
                    "intent_category": first_intent.get("category", None),

                    "audio_path": raw_info["answer"]["audioPath"]
                }

                all_records.append(record)

            except Exception as e:
                print(f"⚠️ {file} 처리 중 오류 발생: {e}")

df = pd.DataFrame(all_records)
print(df.head())
