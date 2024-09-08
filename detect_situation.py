import os

import openai

# OpenAI APIキーを設定
openai.api_key = os.environ['OPENAI_API_KEY']


# OpenAI APIを使って状況を説明する関数
def detect_situation_with_openai(detected_classes):
    prompt = f"私は以下の物体を検出しました: {', '.join(detected_classes)}。これらの物体を基に状況を説明してください。"

    # 正しいAPI呼び出し形式
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "あなたは物体検出に基づいて状況を説明するアシスタントです。",
            },
            {"role": "user", "content": prompt},
        ],
    )

    return response["choices"][0]["message"]["content"]
