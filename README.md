# This fork is for installing on Jetson Xavier - arm64 platform
## Useful links
1. https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
2. https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch
3. https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html --> Installing PyTorch for Jetson Platform
4. https://forums.developer.nvidia.com/t/manually-installing-cuda-11-0-2-on-jetson-xavier-nx-help/191909/4 --> installing cuda toolkit manually
5. https://repo.download.nvidia.com/jetson/
6. https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#upgradable-package-for-jetson --> cuda compatible version
7. https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions --> to verify if device is cuda compatible
8. https://developer.nvidia.com/cuda-12-0-0-download-archive --> cuda toolkit download


## Notes
- Install the wheel from nvidia jetson page for the appropriate Jetpack version using pip install
- For these wheels we need Python 3.8 (you can activate a virtual environment to downgrade)
- Kernel version 5.10.104-tegra
- CUDA 12.0 is compatible with t186 (xavier agx) and jetpack 5.0.2

## Setup instructions (Jetpack 5.0.2)
1. sudo apt-get update && sudo apt-get check
2. If software updater prompts, update that as well
3. install curl
    1. sudo apt-get install libcurl4=7.68.0-1ubuntu2
    2. sudo apt-get install curl
5. Continue to clone the repo, lfs has some error, but all the files seem to be there.
6. install dependencies:

### Downgrading Python to 3.8 to install PyTorch for Jetpack 5.0.5
```
sudo apt-get install python3.8-dev python3.8-venv
```
```
cd
mkdir virtual_env
/usr/bin/python3.8 -m venv ~/virtual_env/venv_with_python3.8
source ~/virtual_env/venv_with_python3.8/bin/activate
python --version
```

### Installing Dependencies
1. Conda (latest) get the aarch64(arm) version [py312_24.3.0-0] https://docs.anaconda.com/free/miniconda/
2. Install pytorch
    1. wget https://developer.download.nvidia.com/compute/redist/jp/v502/pytorch/torch-1.13.0a0+410ce96a.nv22.12-cp38-cp38-linux_aarch64.whl
    2. Activate python 3.8
```
export TORCH_INSTALL=./torch-1.13.0a0+410ce96a.nv22.12-cp38-cp38-linux_aarch64.whl
python3 -m pip install --upgrade pip
pip install numpy==1.21.6
pip install --no-cache $TORCH_INSTALL
```
3. Install CUDA Toolkit
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/arm64/cuda-ubuntu2004.pinsudo
mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-tegra-repo-ubuntu2004-12-0-local_12.0.0-1_arm64.deb
sudo dpkg -i cuda-tegra-repo-ubuntu2004-12-0-local_12.0.0-1_arm64.deb
sudo cp /var/cuda-tegra-repo-ubuntu2004-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/sudo apt-get update
sudo apt-get -y install cuda
```
4. pip install torchaudio



# Fastspeech2 Model using Hybrid Segmentation (HS)

This repository contains a Fastspeech2 Model for 16 Indian languages (male and female both) implemented using the Hybrid Segmentation (HS) for speech synthesis. The model is capable of generating mel-spectrograms from text inputs and can be used to synthesize speech..

The Repo is large in size: We have used [Git LFS](https://git-lfs.com/) due to Github's size constraint (please install latest git LFS from the link, we have provided the current one below).
```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.python.sh | bash
sudo apt-get install git-lfs
git lfs install
```

Language model files are uploaded using git LFS. so please use:

```
git lfs fetch --all
git lfs pull
```
to get the original files in your directory. 

## Model Files

The model for each language includes the following files:

- `config.yaml`: Configuration file for the Fastspeech2 Model.
- `energy_stats.npz`: Energy statistics for normalization during synthesis.
- `feats_stats.npz`: Features statistics for normalization during synthesis.
- `feats_type`: Features type information.
- `pitch_stats.npz`: Pitch statistics for normalization during synthesis.
- `model.pth`: Pre-trained Fastspeech2 model weights.

## Installation

1. Install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) first. Create a conda environment using the provided `environment.yml` file:

```shell
conda env create -f environment.yml
```

2.Activate the conda environment (check inside environment.yaml file):
```shell
conda activate tts-hs-hifigan
```

3.  Install PyTorch separately (you can install the specific version based on your requirements):
```shell
conda install pytorch cudatoolkit
pip install torchaudio
pip install numpy==1.23.0
```
## Vocoder
For generating WAV files from mel-spectrograms, you can use a vocoder of your choice. One popular option is the [HIFIGAN](https://github.com/jik876/hifi-gan) vocoder (Clone this repo and put it in the current working directory). Please refer to the documentation of the vocoder you choose for installation and usage instructions. 

(**We have used the HIFIGAN vocoder and have provided Vocoder tuned on Aryan and Dravidian languages**)

## Usage

The directory paths are Relative. ( But if needed, Make changes to **text_preprocess_for_inference.py** and **inference.py** file, Update folder/file paths wherever required.)

**Please give language/gender in small cases and sample text between quotes. Adjust output speed using the alpha parameter (higher for slow voiced output and vice versa). Output argument is optional; the provide name will be used for the output file.** 

Use the inference file to synthesize speech from text inputs:
```shell
python inference.py --sample_text "Your input text here" --language <language> --gender <gender> --alpha <alpha> --output_file <file_name.wav OR path/to/file_name.wav>
```

**Example:**

```
python inference.py --sample_text "श्रीलंका और पाकिस्तान में खेला जा रहा एशिया कप अब तक का सबसे विवादित टूर्नामेंट होता जा रहा है।" --language hindi --gender male --alpha 1 --output_file male_hindi_output.wav
```
The file will be stored as `male_hindi_output.wav` and will be inside current working directory. If **--output_file** argument is not given it will be stored as `<language>_<gender>_output.wav` in the current working directory.


### Citation
If you use this Fastspeech2 Model in your research or work, please consider citing:

“
COPYRIGHT
2023, Speech Technology Consortium,

Bhashini, MeiTY and by Hema A Murthy & S Umesh,


DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING
and
ELECTRICAL ENGINEERING,
IIT MADRAS. ALL RIGHTS RESERVED "



Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
