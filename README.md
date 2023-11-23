# RVC-Tundra

An enhanced RVC (Retrieval based Voice Conversion) GUI fork mostly for self-use.

This fork is equal to [Applio-RVC-Fork](https://github.com/IAHispano/Applio-RVC-Fork) and [Mangio-RVC-Fork](https://github.com/Mangio621/Mangio-RVC-Fork) on the code level.

This repository has a nature to be outdated. Please refer to the above repositories for the latest updates.

## Installation

See [INSTALL.md](INSTALL.md).

## Usage

### Trainning and advanced inference

#### Launch Gradio server

```bash
python infer-web.py

# or use your favorite environment manager

poetry run python infer-web.py
```

Server will listen on `http://localhost:7865/`. The script will launch a WebUI in your default browser.

#### Use TensorBoard

```bash
python lib/fixes/tensor-launch.py

# or use your favorite environment manager

poetry run python lib/fixes/tensor-launch.py
```

Server will listen on `http://localhost:6006/`. The script will give you a link to TensorBoard with preferred graph configurations.

Refer to [Applio-RVC-Fork/issues](https://github.com/IAHispano/Applio-RVC-Fork/issues) if you encounter any problems.

### Real-time inference

```bash
python infer-realtime.py

# or use your favorite environment manager

poetry run python infer-realtime.py
```

It will start a GUI program using PySimpleGUI. You will be able to select the input and output devices for real-time voice conversion.

Refer to [Mangio-RVC-Fork/issues](https://github.com/Mangio621/Mangio-RVC-Fork/issues) if you encounter any problems.

## License

[MIT License](https://opensource.org/license/mit/)

See [LICENSE](LICENSE) for licenses of other libraries.
