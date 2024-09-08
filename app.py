import streamlit as st
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import openai

# OpenAI APIキーを設定
openai.api_key = 'sk-hhtpwX8LDcTsg-4k3Z8KnbEvYm6uUFlIsYyiUe4FZqT3BlbkFJu7fNd_EG1pfa1CAsyxMQI1yyPTrwNFqUU4L_ygPt0A'

# 画像をBase64に変換する関数
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Base64エンコードされた画像データを画像ファイルに変換する関数
def base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image

def detect_objects(image):
    # YOLOの設定ファイルと重みファイルをロード
    net = cv2.dnn.readNet("models/yolo/yolov3.weights", "models/yolo/yolov3.cfg")
    layer_names = net.getLayerNames()

    # クラスラベルのロード（COCO.names）
    classes = []
    with open("models/yolo/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    # OpenCVのバージョンに応じて適切にレイヤーを取得
    try:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except IndexError:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # 画像の前処理と物体検出
    image_np = np.array(image)
    height, width, channels = image_np.shape

    blob = cv2.dnn.blobFromImage(image_np, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detected_classes = []
    for i in range(len(boxes)):
        if i in indexes:
            label = str(classes[class_ids[i]])
            detected_classes.append(label)

    return detected_classes

# OpenAI APIを使って状況を説明する関数
def detect_situation_with_openai(detected_classes):
    prompt = f"私は以下の物体を検出しました: {', '.join(detected_classes)}。これらの物体を基に状況を説明してください。"

    # 正しいAPI呼び出し形式
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # または "gpt-4"
        messages=[
            {"role": "system", "content": "あなたは物体検出に基づいて状況を説明するアシスタントです。"},
            {"role": "user", "content": prompt}
        ]
    )

    return response['choices'][0]['message']['content']

# StreamlitアプリのUI
st.title("画像物体検出アプリ")

# 画像のアップロード（JPEG、PNGファイルを許可）
uploaded_file = st.file_uploader("画像ファイルをアップロードしてください", type=["jpeg", "png", "jpg"])

if uploaded_file is not None:
    # 画像を読み込み、PIL形式に変換
    image = Image.open(uploaded_file)
    
    # 画像を表示
    st.image(image, caption="アップロードされた画像", use_column_width=True)

    # 画像をBase64エンコードして処理に使用
    base64_string = image_to_base64(image)
    image = base64_to_image(base64_string)

    # 物体検出を実行
    detected_classes = detect_objects(image)
    st.write(f"検出された物体: {', '.join(detected_classes)}")

    # OpenAI APIで状況を説明
    situation = detect_situation_with_openai(detected_classes)
    st.write(f"状況説明: {situation}")


