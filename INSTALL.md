## 1. Runtimes

### a) use WinGet

```cmd
winget install --id=Microsoft.VisualStudio.2022.BuildTools --force --override "--wait --passive --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22000"
winget install --id Git.Git -e
winget install --id Python.Python.3.10 -e
```

### b) manual installation

- Visual Studio Build Tools 2022

> https://aka.ms/vs/17/release/vs_BuildTools.exe

- Visual C++ 2022 Redistributable

> https://aka.ms/vs/17/release/vc_redist.x64.exe

> NOTE: You must install workload "Desktop development with C++" and component "Windows 10 SDK" / "Windows 11 SDK".

- Git for Windows

> https://github.com/git-for-windows/git/releases/

- Python 3.9 / 3.10

> https://www.python.org/downloads/release/python-3913/

> https://www.python.org/downloads/release/python-31011/

## 2. Clone repository

```bash
git clone https://github.com/TundraWork/RVC-Tundra.git
cd RVC-Tundra
```

## 3. PyTorch

- PyTorch 2.0.1 (CUDA 11.8)
- TorchVision 0.15.2
- TorchAudio 2.0.2

### a) poetry using pyproject.toml

Install poetry:

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

Then skip to step 4.

### b) poetry from scratch

```bash
poetry env use "C:\Program Files\Python310\python.exe"
poetry source add -p explicit pytorch-cu118 https://download.pytorch.org/whl/cu118
poetry add --source pytorch-cu118 torch@2.0.1 torchvision@0.15.2 torchaudio@2.0.2
```

### c) pip

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

## 4. Other dependencies

### a) poetry using pyproject.toml

```bash
poetry env use "C:\Program Files\Python310\python.exe"
poetry install --no-root
```

### b) poetry from scratch

```bash
poetry add tornadohttp setuptools pydantic fairseq wheel google-auth-oauthlib pedalboard pydub httpx faiss_cpu ffmpeg_python ffmpy websockets@11.0.3 gradio@3.50.2 librosa llvmlite elevenlabs gTTS wget psutil matplotlib mega.py git+https://github.com/wkentaro/gdown.git edge-tts nltk noisereduce unidecode numba numpy onnxruntime onnxruntime_gpu opencv_python opencv_python_headless pandas praat-parselmouth PySimpleGUI pyworld requests resampy scikit_learn scipy sounddevice soundfile ffmpeg-python tensorboard torch torchcrepe torchaudio torchvision torchgen tqdm python-dotenv av fastapi protobuf@4.23.4 tensorboardX gin gin_config flask_cors flask
```

### c) pip

```bash
pip install -r requirements.txt
``````

## 5. Pretrained model weights

All required pretrained model weights will be downloaded automatically from huggingface on first run.

See `infer-web.py` for details.
