#!/usr/bin/env python3
from flask import Flask, request
from PIL import Image, ImageDraw
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch

class COCOObjectDetection:

    processor = DetrImageProcessor.from_pretrained("detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("detr-resnet-50")

    def predict_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        return self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]


app = Flask(__name__)
coco = COCOObjectDetection()

@app.route('/get_predictions', methods=['POST'])
def get_predictions():
    file = request.files['file']
    image = Image.open(file.stream)
    results = coco.predict_image(image)

    response = []

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        response.append({
            "score": round(score.item(), 3),
            "label": coco.model.config.id2label[label.item()],
            "box": [round(i, 2) for i in box.tolist()]
        })

    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
