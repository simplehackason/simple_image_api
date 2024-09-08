from utils import base64_to_image
from detect_object import detect_objects
from detect_situation import detect_situation_with_openai


def run(base64_string) -> tuple[str, str]:
    image = base64_to_image(base64_string)
    detected_classes = detect_objects(image)
    situation = detect_situation_with_openai(detected_classes)
    # TODO: image の変換処理を追加する
    return situation, base64_string
