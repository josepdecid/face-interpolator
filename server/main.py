import base64
import cv2
import io
import jwt
import numpy as np
import os
import torch
from PIL import Image
from flask import Flask, jsonify
from flask import request
from flask_cors import CORS, cross_origin
from torchvision import transforms

from face_interpolator.data.celeba_dataset import CelebaDataset
from face_interpolator.utils.constants import MEAN, STD, CELEBA_SIZE
from face_interpolator.utils.unormalize import UnNormalize
from models.conditional_predictive_vae.conditional_vae import ConditionalConvVAE

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'

CORS(app, resources={
    r'/parametrize': {'origins': 'http://localhost:3000'},
    r'/interpolate': {'origins': 'http://localhost:3000'}
})

CKPT_PATH = '../output/run09_cvae/checkpoints/run09_cvae-epoch=286-val_loss=2108884.25.ckpt'


def load_checkpoint():
    attributes_size = CelebaDataset.image_attributes_size
    model = ConditionalConvVAE.load_from_checkpoint(CKPT_PATH, bottleneck_size=256, attribute_size=attributes_size)
    model.eval()
    return model


model = load_checkpoint()


@app.route('/parametrize', methods=['POST'])
@cross_origin(origin='localhost', headers=['Content- Type'])
def extract_parameters():
    if not is_valid_user(request.headers.get('Authorization')):
        return {'status_code': 401}

    file_str = request.files['imageData'].read()
    img_np = np.frombuffer(file_str, np.uint8)
    img_array = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    img = cv2.resize(img_array, (CELEBA_SIZE[1], CELEBA_SIZE[0]), interpolation=cv2.INTER_AREA)
    img = torch.tensor(img / 255, dtype=torch.float)

    img = img.permute(2, 0, 1)
    img = transforms.Normalize(mean=MEAN, std=STD)(img)
    img = img.unsqueeze(0)

    with torch.no_grad():
        mu, logvar = model.encode(img)
        parameters = model.reparametrize(mu, logvar)
        attributes = model.predict_attributes(parameters)

        sorted_idx_by_variance = np.argsort(logvar[0].numpy())[::-1]

    return jsonify({
        'attributeNames': CelebaDataset.attribute_names,
        'parameters': attributes[0].tolist() + parameters[0].tolist(),
        'maxVarianceIdx': sorted_idx_by_variance.tolist()
    })


@app.route('/interpolate', methods=['POST'])
@cross_origin(origin='localhost', headers=['Content- Type'])
def interpolate():
    if not is_valid_user(request.headers.get('Authorization')):
        return {'status_code': 401}

    parameters = request.get_json().get('parameters')

    with torch.no_grad():
        parameters = torch.tensor(parameters, dtype=torch.float).unsqueeze(0)
        interpolated_image = model.decode(parameters[:, CelebaDataset.image_attributes_size:],
                                          parameters[:, :CelebaDataset.image_attributes_size])

    unorm = UnNormalize(mean=MEAN, std=STD)
    img = unorm(interpolated_image[0])
    img = img.permute(1, 2, 0).numpy()
    img = Image.fromarray((img * 255).astype('uint8'))
    raw_bytes = io.BytesIO()
    img.save(raw_bytes, 'JPEG')
    raw_bytes.seek(0)
    img_base64 = base64.b64encode(raw_bytes.read())
    return jsonify({'image': img_base64.decode()})


def is_valid_user(authorization_header):
    user_data = jwt.decode(authorization_header.split(' ')[-1], os.environ.get('JWT_SECRET', ''))
    return set({}) == set(user_data.keys())


if __name__ == '__main__':
    app.run(debug=True)
