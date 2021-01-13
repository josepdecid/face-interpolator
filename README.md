## Python Environment

Create the Python virtual environment using the following command:

```shell script
python3 -m venv <venv_path>
```

Then, activate it with

```shell script
source <venv_path>/bin/activate
```

or set it as the Python interpreter in PyCharm or any other IDE. Then, install the requirements from the `requirements.txt` file, using

```shell script
pip install -r requirements.txt
```

## Dataset

We use the data from the [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to train our model.

Download the dataset from their website and extract it to `datasets` folder.

The following folder structure must be followed:

```
datasets

  ┕ CelebA

      ┕ Anno: Contains all the annotation txt files.
  
      ┕ Eval: Contains the file with the data partitions for train, validation and test.
  
      ┕ Img: Contains all the jpg images from the dataset.
```

## Train Models

To train the models in `face_interpolator/models` we can use the `local_launcher.py` and `auto_launcher.py` scripts.
Both files are job launcher managers that handle local runs and cluster runs in Slurm environments.

To use them we need to call the file with an argument that corresponds to the configuration file name in folder `configs`.
These files have the following scheme:

```json
{
  "launcher": "train.py",
  "args": "--gpus 1 --job_name run01 --bottleneck 256",
  "job_name": "run01",
  "nodes": 1,
  "ntasks": 1,
  "cpu_per_task": 160,
  "gres": "gpu:4",
  "time": "48:00:00"
}
```

In these, we specify training parameters such as the model to train with its launcher, number of CPUs and GPUs or model arguments.
Then, the launcher managers call the specific launcher file for the model that we want to train.

It is possible to create new configurations with the name `configs/{NAME}.json`, containing the same keys as in the example.

## Run Server

To run the server, simply run the command:

```bash
python server/main.py
```

This will run the server in the default port 3000.