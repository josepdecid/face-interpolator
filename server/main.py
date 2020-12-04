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

from face_interpolator.utils.constants import MEAN, STD, CELEBA_SIZE
from face_interpolator.data.celeba_dataset import CelebaDataset
from face_interpolator.models.conditional_vae import ConditionalConvVAE
from face_interpolator.utils.unormalize import UnNormalize

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'

CORS(app, resources={
    r'/parametrize': {'origins': 'http://localhost:3000'},
    r'/interpolate': {'origins': 'http://localhost:3000'}
})

CKPT_PATH = '../output/run01_cvae/checkpoints/run01-epoch=138-val_loss=2180395.50.ckpt'


def load_checkpoint():
    attributes_size = CelebaDataset.image_attributes_size
    model = ConditionalConvVAE.load_from_checkpoint(CKPT_PATH, bottleneck_size=256, attribute_size=attributes_size)
    model.eval()
    return model


model = load_checkpoint()


@app.route('/parametrize', methods=['POST'])
@cross_origin(origin='localhost', headers=['Content- Type'])
def extract_parameters():
    file_str = request.files['imageData'].read()
    img_np = np.frombuffer(file_str, np.uint8)
    img_array = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    # TODO: BW images
    img = cv2.resize(img_array, (CELEBA_SIZE[1], CELEBA_SIZE[0]), interpolation=cv2.INTER_AREA)
    img = torch.tensor(img / 255, dtype=torch.float)

    img = img.permute(2, 0, 1)
    img = transforms.Normalize(mean=MEAN, std=STD)(img)
    img = img.unsqueeze(0)

    # plt.imshow(img[0].permute(1, 2, 0).numpy())
    # plt.show()
    attributes = torch.zeros(1, CelebaDataset.image_attributes_size)

    with torch.no_grad():
        mu, logvar = model.encode(img, attributes)
        parameters = model.reparametrize(mu, logvar)

    # TODO: Parameters only 1 dimension, return all
    return jsonify({
        'attributeNames': CelebaDataset.attribute_names,
        'parameters': attributes[0].tolist() + parameters[0].tolist()
    })


@app.route('/interpolate', methods=['POST'])
@cross_origin(origin='localhost', headers=['Content- Type'])
def interpolate():
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


if __name__ == '__main__':
    app.run(debug=True)
