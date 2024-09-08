import cv2
import numpy as np


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
    height, width, _ = image_np.shape

    blob = cv2.dnn.blobFromImage(
        image_np, 0.00392, (416, 416), (0, 0, 0), True, crop=False
    )
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
