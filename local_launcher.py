import json
import subprocess
import sys
from argparse import ArgumentParser

import os

from face_interpolator.constants import CONFIGS_DIR

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()

    config_file_name = os.path.join(CONFIGS_DIR, f'{args.config}.json')
    with open(config_file_name) as config_file:
        params = json.load(config_file)

    subprocess.run(f'{sys.executable} {params["launcher"]} {params["args"]}')
