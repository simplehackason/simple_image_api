from utils import base64_to_image, pil_to_base64
from detect_object import detect_objects
from detect_situation import detect_situation_with_openai


def run(base64_string) -> tuple[str, str]:
    image = base64_to_image(base64_string)
    # yolo を用いて物体検出
    detect_image, detected_classes = detect_objects(image)
    # 物体検出の内容をもとに LLM による状況説明
    situation = detect_situation_with_openai(detected_classes)
    # 検出した画像を base64 に変換
    output_base64 = pil_to_base64(detect_image)
    return situation, output_base64
