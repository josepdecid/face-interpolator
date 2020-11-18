import base64
import cv2
import io
import numpy as np
import torch
from PIL import Image
from flask import Flask, jsonify
from flask import request
from flask_cors import CORS, cross_origin
from torchvision import transforms

from face_interpolator.constants import MEAN, STD, CELEBA_SIZE
from face_interpolator.models import ConvVAE
from face_interpolator.utils.unormalize import UnNormalize

import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'

CORS(app, resources={
    r'/parametrize': {'origins': 'http://localhost:3000'},
    r'/interpolate': {'origins': 'http://localhost:3000'}
})

CKPT_PATH = 'C:\\Users\\jdeci\\OneDrive\\Escritorio\\plotsnigga\\run01-epoch=23-val_loss=0.33.ckpt'


def load_checkpoint():
    model = ConvVAE.load_from_checkpoint(CKPT_PATH, bottleneck_size=256)
    model.eval()
    return model


model = load_checkpoint()


@app.route('/parametrize', methods=['POST'])
@cross_origin(origin='localhost', headers=['Content- Type'])
def extract_parameters():
    file_str = request.files['imageData'].read()
    img_np = np.fromstring(file_str, np.uint8)
    img_array = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    # TODO: BW images
    img = cv2.resize(img_array, (CELEBA_SIZE[1], CELEBA_SIZE[0]), interpolation=cv2.INTER_AREA)
    img = torch.tensor(img / 255, dtype=torch.float)

    img = img.permute(2, 0, 1)
    img = transforms.Normalize((0.5063, 0.4258, 0.3832), (0.2660, 0.2452, 0.2414))(img)
    img = img.unsqueeze(0)

    plt.imshow(img[0].permute(1, 2, 0).numpy())
    plt.show()

    with torch.no_grad():
        mu, logvar = model.encode(img)
        parameters = model.reparametrize(mu, logvar)

    # TODO: Parameters only 1 dimension, return all
    return jsonify({'parameters': parameters[0].tolist()})


@app.route('/interpolate', methods=['POST'])
@cross_origin(origin='localhost', headers=['Content- Type'])
def interpolate():
    parameters = request.get_json().get('parameters')

    with torch.no_grad():
        parameters = torch.tensor(parameters, dtype=torch.float).unsqueeze(0)
        interpolated_image = model.decode(parameters)

    unorm = UnNormalize(mean=MEAN, std=STD)
    img = unorm(interpolated_image[0])
    img = img.permute(1, 2, 0).numpy()
    img = Image.fromarray((img * 255).astype('uint8'))
    rawBytes = io.BytesIO()
    img.save(rawBytes, 'JPEG')
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    return jsonify({'image': img_base64.decode()})


if __name__ == '__main__':
    app.run(debug=True)
