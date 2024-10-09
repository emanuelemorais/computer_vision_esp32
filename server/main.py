import os
import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

UPLOAD_FOLDER = './uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.data  
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    img_name = os.path.join(UPLOAD_FOLDER, 'processed_image.jpg')
    cv2.imwrite(img_name, img) 

    return jsonify({'message': 'Image uploaded and saved locally', 'file_path': img_name}), 200

if __name__ == '__main__':
    print("Starting server...")
    app.run(host='0.0.0.0', port=5001)
