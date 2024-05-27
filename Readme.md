# Kognitive Robotik CNN Kaggle competition

## Description

Join the Kaggle competition using this link:
https://www.kaggle.com/t/5aed720896bc4a018b626d73fae7d535

There you will find all the information you need!

## Ideas you can try out to improve the classifier

If you want to improve the performance of the classifier, check out some of the
following ideas. It is a good idea to add one change at a time and evaluate how your
results change ;)

- Data Augmentation (Rotation, Adding noise, changing brightness, etc.)
- Changing the model architecture (Add Convolutional Layers, play around with kernel
  sizes and number of filters, add another Dense Layer)
- Try adding BatchNorm Layers (Research what they do and where to place those)
- Try adding Regularization (Research which parameters you can regularize and why you
  would want to)
- Use a Learning Rate Reducer and Early Stopping Callback (Keras Documentation has you
  covered there)

Documentation:
https://keras.io/api/ (We are using tensorflow2.16.1 which comes with keras3.3)

## Setting Up Python and Virtual Environments

The dependencies in this repository require a Python version between Python3.9 and
Python3.11. Below you will find instructions on how to set up Python and create a
virtual environment in Windows Native, Linux & Windows WSL and MacOS.

https://docs.python.org/3.11/library/venv.html

**NOTE:** if after setting up your virtual environemnt you **don't** see the environment
name in your prompt and `python --version` returns a different python version than what
you expect, first make sure you actuall ran `source env_name/bin/activate` or `.\env_name\Scripts\activate.bat` on Windows Native, and if that doesn't sort you out,
you can run the `pip` and `python` commands directly from the `env_name/bin` or
`env_name\Scripts` directory.
For example:

```sh
env_name/bin/pip install -r requirements.txt
env_name/bin/python train.py
```

### Native Windows

#### 1. Install a Specific Python Version

1. **Download the Installer:**

   - Go to the [Python Downloads page](https://www.python.org/downloads/).
   - Select the desired version and download the Windows installer
     (usually an executable file).

2. **Run the Installer:**
   - Run the downloaded installer.
   - Make sure to check the box that says "Add Python to PATH".
   - Choose "Customize installation" to specify the installation directory if needed.

#### 2. Set Up a Virtual Environment

1. **Open Command Prompt or PowerShell:**

   - Press `Win + R`, type `cmd`, and press Enter to open Command Prompt.
   - Alternatively, you can search for PowerShell in the start menu.

2. **Create a Virtual Environment:**

   - Navigate to the directory where you want to create your virtual environment.
   - Run the following command, replacing `env_name` with your desired environment name:
     ```sh
     python -m venv env_name
     ```

3. **Activate the Virtual Environment:**
   - To activate the virtual environment, run:
     ```sh
     .\env_name\Scripts\activate.bat
     # or for powershell
     .\env_name\Scripts\activate.ps1
     ```
   - Your prompt should now include `(env_name)` indicating the virtual environment
     is active.

### Linux and Windows Subsystem for Linux (WSL)

#### 1. Install a Specific Python Version

1. **Install Prerequisites:**
   ```sh
   sudo apt update
   sudo apt install software-properties-common
   ```
2. **Add PPA for multiple python versions**

```sh
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
```

3. **Install the Desired Python Version:**

- Replace 3.x with the desired Python version (e.g., 3.10):

```sh
sudo apt install python3.x python3.x-venv python3.x-dev
```

#### 2. Set Up a Virtual Environment

1. **Create a Virtual Environment:**

- Navigate to the directory where you want to create your virtual environment.
- Run the following command, replacing 3.x with your installed Python version and
  env_name with your desired environment name:
  ```sh
  python3.x -m venv env_name
  ```

2. **Activate the Virtual Environment:**
   - To activate the virtual environment, run:
     ```sh
     source .\env_name\Scripts\activate
     ```
   - Your prompt should now include `(env_name)` indicating the virtual environment is
     active.

### MacOS

#### 1. Install a Specific Python Version

1. **Install Homebrew (if not already installed):**

   - Homebrew is a package manager for macOS.
   - Open Terminal and run:
     ```sh
     /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
     ```

2. **Install the Desired Python Version:**
   - Replace `3.x` with the desired Python version (e.g., `3.10`):
     ```sh
     brew install python@3.x
     ```

#### 2. Set Up a Virtual Environment

1. **Create a Virtual Environment:**

   - Navigate to the directory where you want to create your virtual environment.
   - Run the following command, replacing `env_name` with your desired environment name:
     ```sh
     python3 -m venv env_name
     ```

2. **Activate the Virtual Environment:**
   - To activate the virtual environment, run:
     ```sh
     source env_name/bin/activate
     ```
   - Your prompt should now include `(env_name)` indicating the virtual environment
     is active.

## Installing requirements

Set up Python (if needed) and virtualenv, now you can install the package requirements
as follows:

```sh
# Make sure to replace env_name with your actual environment name
source env_name/bin/activate # env_name\Scripts\activate on Windows Native
pip install -r requirements.txt
```

or, if the activate scripts should for some reason not work for you:

```sh
# Make sure to replace env_name with your actual environment name
env_name/bin/pip install -r requirements.txt
```

### Jupyter Notebook (Optional)

If you want to use the jupyter notebook example, either in your IDE or with the
`jupyter notebook` command, you need to register your virtual environment as a
usable kernel:

```sh
# Make sure to replace env_name with your actual environment name
python -m ipykernel install --user --name=env_name
```

or, if the venv activate script should for some reason not work for you:

```sh
# Make sure to replace env_name with your actual environment name
env_name/bin/python -m ipykernel install --user --name=env_name
```

## Run the CNN Example training script

To run the CNN example `train.py` model training, make sure you set up and activated
your virtual environment. Then you can run

```sh
python train.py
```

or, if the venv activate script should for some reason not work for you:

```sh
# Make sure to replace env_name with your actual environment name
source env_name/bin/python train.py
```

### Create a Jupyter notebook

You can also run Jupyter notebooks if you like, just make sure you followed the jupyter
install instructions mentioned above!

```sh
python -m jupyter notebook
```

or, if the venv activate script should for some reason not work for you:

```sh
# Make sure to replace env_name with your actual environment name
env_name/bin/python -m jupyter notebook
```
