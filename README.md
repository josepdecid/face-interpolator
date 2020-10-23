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

