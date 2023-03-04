#!/usr/bin/env python3
import requests
import glob
import os
from PIL import Image, ImageDraw

# Draw bounding boxes on image using PIL
def draw_bounding_boxes(image, response):
    draw = ImageDraw.Draw(image)
    for result in response:
        box = result["box"]
        label = result["label"]
        score = result["score"]
        draw.rectangle(box, outline="red")
        draw.text((box[0], box[1]), f"{label}: {score}", fill="red")
    return image

if __name__ == "__main__":

    images_file_paths = [i for i in glob.glob(os.path.join(os.path.dirname(__file__), "images/*"))]

    for image_file_path in images_file_paths:
        print(image_file_path)
        
        files = {'file': open(image_file_path, 'rb')}
        response = requests.post('http://localhost:8080/get_predictions', files=files)

        if not response.status_code == 200:
            print("Error: ", response.status_code)
            exit(1)

        # Print response
        print(response.json())

        image = Image.open(image_file_path)
        image.show(draw_bounding_boxes(image, response.json()))
        