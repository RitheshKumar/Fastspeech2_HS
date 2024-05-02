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
9. https://forums.developer.nvidia.com/t/having-problems-updating-cmake-on-xavier-nx/169265 --> install cmake
10. https://onnxruntime.ai/docs/build/eps.html#nvidia-jetson-tx1tx2nanoxavier --> Install onnxruntime for Jetson Devices
11. https://elinux.org/Jetson_Zoo#ONNX_Runtime --> install this for onnxruntime-gpu for python 3.8 --> this is the pytorch we have
12. https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html --> CUDA EP install/usage
13. https://developer.nvidia.com/cudnn-downloads --> CuDNN library install
14. https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html --> to find the correct versions for onnxruntime + cuda + cudnn
15. https://elenacliu-pytorch-cuda-driver.streamlit.app/ --> Checking version compatiblities
16. https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-890/install-guide/index.html --> CuDNN installation

## Version Map [These are interdependent on version]
1. OS: Ubuntu 20.04.6 LTS (Focal Fossa)
2. Kernel: 5.10.104-tegra
3. Architecture: aarch64 (arm)
4. Jetpack 5.0.2 (rev2)
5. Conda 24.3.0 aarch64
6. Curl 7.68.0
7. Python 3.8.10
8. numpy 1.21.6
9. Pytorch 2.2.2, Supposed to be 1.13 [torch-1.13.0a0+410ce96a.nv22.12-cp38-cp38-linux_aarch64.whl]
10. torchAudio 2.2.2
11. CUDA 12.0.0




## Notes
- Install the wheel from nvidia jetson page for the appropriate Jetpack version using pip install
- For these wheels we need Python 3.8 (you can activate a virtual environment to downgrade)
- Kernel version 5.10.104-tegra
- CUDA 12.0 is compatible with t186 (xavier agx) and jetpack 5.0.2
- $LD_LIBRARY_PATH is where CUDA/python related library paths are found [need to validate this statement]

## Setup instructions (Jetpack 5.0.2)
1. sudo apt-get update && sudo apt-get check
2. If software updater prompts, update that as well
3. install curl
    1. sudo apt-get install libcurl4=7.68.0-1ubuntu2
    2. sudo apt-get install curl
5. Continue to clone the repo, lfs has some error, but all the files seem to be there.
6. install dependencies:

### Activating Python 3.8 to install PyTorch for Jetpack 5.0.5
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
You can verify installation with `nvcc --version`
4. pip install torchaudio
5. Also install libhdf5-dev [make sure the dev version is installed]
6. For having conda in custom location:
```
conda create --prefix /work/mydir/mypath

Package cache directory: $HOME/.conda/pkgs [default]
You can add pkgs_dir $HOME/.condarc
OR set CONDA_PKGS_DIRS environment variable
```
7. conda list --> gives the list of libraries and their versions
8. [MAYBE] typeguard==2.13.3


## Onnx Model Route
1. cuda toolkit is installed [12.0]
2. pytorch is installed
3. install cmake 3.26 or greater [3.26.6]
```
tar -zxvf cmake-3.26.6-linux-aarch64.tar.gz
cd cmake-3.26.6-linux-aarch64/
sudo cp -rf bin/ doc/ share/ /usr/local/
sudo cp -rf man/* /usr/local/man
sync
cmake --version 
```
4. Install onnxruntime for Jetson
```
git clone --recursive -b rel-1.12.0 https://github.com/microsoft/onnxruntime
export PATH="/usr/local/cuda/bin:${PATH}"
export CUDACXX="/usr/local/cuda/bin/nvcc"
sudo apt install -y --no-install-recommends build-essential software-properties-common libopenblas-dev libpython3.8-dev python3-pip python3-dev python3-setuptools python3-wheel
./build.sh --config Release --update --build --parallel 2 --build_wheel --use_tensorrt --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu --tensorrt_home /usr/lib/aarch64-linux-gnu
sudo pip install build/Linux/Release/dist/onnxruntime_gpu-1.12.0-cp38-cp38-linux_aarch64.whl
```
OR
4. Install the onnxruntime wheel found in: https://elinux.org/Jetson_Zoo#ONNX_Runtime [python 3.8, Jetpack 5.0. onnxruntime 1.12.1]
`pip install <downloaded-wheel>.whl`

5. Install CuDNN library [version 8.5.0]
Download 8.5.0, ubuntu 20.04, arm64sbsa package from NVidia CuDNN archive: cudnn-local-repo-ubuntu2004-8.5.0.96_1.0-1_arm64.deb
(You may need to login to download)
Direct link: https://developer.nvidia.com/compute/cudnn/secure/8.5.0/local_installers/11.7/cudnn-local-repo-ubuntu2004-8.5.0.96_1.0-1_arm64.deb
```
sudo dpkg -i cudnn-local-repo-ubuntu2004-8.5.0.96_1.0-1_arm64.deb 
sudo cp /var/cudnn-local-repo-ubuntu2004-8.5.0.96/cudnn-local-0CCB36B3-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install libcudnn8
sudo apt-get install libcudnn8-dev
sudo apt-get install libcudnn8-samples
sudo apt-get install zlib1g
```
Verify installation by searching for "libcudnn" in `/usr/lib/aarch64-linux-gnu`

ALSO verify installation by:
a. Install freeimage library
```
sudo apt-get update
sudo apt-get install -y libfreeimage-dev
```
b. Now run their sample code
```
cp -r /usr/src/cudnn_samples_v8/ $HOME
cd  $HOME/cudnn_samples_v8/mnistCUDNN
make clean && make
./mnistCUDNN
Test passed!
```

## Take 3
1. Install Jetpack 5.0.2 using `sdkmanager --archived-versions`
2. `sudo apt-get update && sudo apt-get check` on the device
3. Ensure you install SDK components as well
4. Validate Cuda using sample code at /usr/local/cuda/samples/1_Utilities
5. Validate CuDNN using 
```
cp -r /usr/src/cudnn_samples_v8/ $HOME
cd  $HOME/cudnn_samples_v8/mnistCUDNN
make clean && make
./mnistCUDNN
Test passed!
```
6. install onnxruntime-gpu from  https://elinux.org/Jetson_Zoo#ONNX_Runtime
```
pip install <onnxruntime-gpu-file>.whl
```
7. Mount sd card on to /usr/local path. This is so we have enough space for everything.
```
lsblk (identify the sdcard partition path: let's assume /dev/mmcblk0p1 )
sudo mkdir /usr/local/sd
sudo mount /dev/mmcblk1p1 /usr/local/sd
df -H -T /usr/local/sd --> to verify the mount
Edit /etc/fstab UUID=<find this out from lsblk> /usr/local/sdcard ext4 noauto,nofail,rw,suid,dev,exec,nouser,async 1 2
```

## Commands for Debugging Dependencies
1. set include-system-site-packages key to true in pyvenv.cfg
2. pip list --local
3. PYTHONPATH variable lists where this environment's look up dirs
4. ldd $(which python) —> lists the shared libraries used by python interpreter
5. To include the share libs Set include-system-site-packages key to true in pyvenv.cfg
6. Check sys.prefix != sys.base_prefix to ensure that paths are correct.
7. Open venv/bin/activate: export PATH=$PATH:/my/custom/path; source venv/bin/activate; echo $PATH
8. pip show <packagename> --> shows the install location
9. python -m site --user-site --> shows where site-packages are looked up for the current environment
10. Open python interpreter:
```	
  import site
  print(site.getsitepackages())
```
11. dpkg -L <package-name> --> lists the already installed path
12. dpkg-deb -c <package.deb> --> lists the paths about to be installed
13. strace -o log.txt head --> gives a detailed trace of all the c calls in the kernel
14. printenv
15. tar -ztf <filename> --> to find the contents of a tarball
16. ldd <appname> --> gives the list of shared libaries used by the app
17. /etc/ld.so.cache --> is where an app looks up for available shared objects, to load from
18. /etc/ld.so.conf points to /etc/ld.so.conf.d which has all the configurations for corresponding shared libraries.
19. ldconfig -p | grep <library_name>
20. apt search <library_name>
21. readelf -d /bin/curl or readelf -d libshared.so
22. readelf -d /path/to/executable | grep -E 'RPATH|RUNPATH'

### Misc
1. udevadm info /dev/mmcblk1p1 --> is a kernel level tool that polls for newly connected devices
2. /etc/udev/rules.d/ is where UFS(sdcard) connection rules are specified. This takes precedence.


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
