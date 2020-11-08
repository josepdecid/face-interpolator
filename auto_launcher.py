import json
import subprocess
from argparse import ArgumentParser
from time import strftime

import os

import sys

from face_interpolator.constants import CONFIGS_DIR

P9_MODULES = ['ibm', 'openmpi/4.0.1', 'gcc/8.3.0', 'cuda/10.2', 'cudnn/7.6.4', 'nccl/2.4.8',
              'tensorrt/6.0.1', 'fftw/3.3.8', 'ffmpeg/4.2.1', 'opencv/4.1.1', 'atlas/3.10.3',
              'scalapack/2.0.2', 'szip/2.1.1', 'python/3.7.4_ML']

DATA_TO_UPLOAD = [
    'launcher.sh',
    'face_interpolator'
]

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--user', type=str)
    args = parser.parse_args()

    config_file_name = os.path.join(CONFIGS_DIR, f'{args.config}.json')
    with open(config_file_name) as config_file:
        params = json.load(config_file)

    logs_path = os.path.join('face-interpolator', strftime('%Y%m%d%H%M%S') + '_' + params['job_name'])
    output_logs_path = logs_path

    bash_data = f"""#!/bin/bash
    
#SBATCH --job-name={params['job_name']}
#SBATCH --qos=bsc_cs
#SBATCH --workdir=face_interpolator
#SBATCH --output={logs_path}.out
#SBATCH --error={logs_path}.err
#SBATCH --cpus-per-task={params['cpu_per_task']}
#SBATCH --gres={params['gres']}
#SBATCH --time={params['time']}

module purge
module load {' '.join(P9_MODULES)}

export PYTHONPATH=face_interpolator
python {params['launcher']} {params['args']}
"""

    with open('launcher.sh', mode='w') as f:
        f.write(bash_data)

    upload_command = f'scp -i .ssh/id_rsa -r {" ".join(DATA_TO_UPLOAD)} {args.user}@dt01.bsc.es:~/face-interpolator/.'
    execute_command = f'ssh -i .ssh/id_rsa -t {args.user}@plogin1.bsc.es "cd face-interpolator;sed -i.bak \'s/\r$//\' launcher.sh; sbatch launcher.sh"'

    print('[UPLOADING CODEBASE]')
    print(f'> {upload_command}')

    subprocess.run(upload_command)

    print('[EXECUTING JOB]')
    print(f'> {execute_command}')

    subprocess.run(execute_command)

    os.remove('launcher.sh')
