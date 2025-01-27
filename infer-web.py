import os, sys

from tensorboard import program

now_dir = os.getcwd()
sys.path.append(now_dir)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
import logging
import shutil
import threading
from assets.configs.config import Config

import lib.tools.model_fetcher as model_fetcher
import math as math
import ffmpeg as ffmpeg
import traceback
import warnings
from random import shuffle
from subprocess import Popen
from time import sleep
import json
import pathlib
import fairseq
import socket
import requests
import subprocess

logging.getLogger("faiss").setLevel(logging.WARNING)
import faiss
import gradio as gr
import numpy as np
import torch as torch
import regex as re
import soundfile as SF

SFWrite = SF.write
from dotenv import load_dotenv
from sklearn.cluster import MiniBatchKMeans
import datetime


from glob import glob1
import signal
from signal import SIGTERM
from assets.i18n.i18n import I18nAuto
from lib.infer.infer_libs.train.process_ckpt import (
    change_info,
    extract_small_model,
    merge,
    show_info,
)
from lib.infer.modules.uvr5.mdxnet import MDXNetDereverb
from lib.infer.modules.uvr5.preprocess import AudioPre, AudioPreDeEcho
from lib.infer.modules.vc.modules import VC
from lib.infer.modules.vc.utils import *
import lib.globals.globals as rvc_globals
import nltk

nltk.download("punkt", quiet=True)

import tabs.resources as resources
import tabs.tts as tts
import tabs.merge as mergeaudios
import tabs.processing as processing

from lib.infer.infer_libs.csvutil import CSVutil
import time
import csv
from shlex import quote as SQuote

logger = logging.getLogger(__name__)

RQuote = lambda val: SQuote(str(val))

tmp = os.path.join(now_dir, "temp")

# directories = ["logs", "datasets", "weights", "audio-others", "audio-outputs"]

shutil.rmtree(tmp, ignore_errors=True)

os.makedirs(tmp, exist_ok=True)

# Start the download server
if True == True:
    host = "localhost"
    port = 8000

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)  # Timeout in seconds

    try:
        sock.connect((host, port))
        logger.info("Starting the Flask server")
        logger.warn(
            f"Something is listening on port {port}; check open connection and restart Applio."
        )
        logger.warn("Trying to start it anyway")
        sock.close()
        requests.post("http://localhost:8000/shutdown")
        time.sleep(3)
        script_path = os.path.join(now_dir, "lib", "tools", "server.py")
        try:
            subprocess.Popen(f"python {script_path}", shell=True)
            logger.info("Flask server started!")
        except Exception as e:
            logger.error(f"Failed to start the Flask server")
            logger.error(e)
    except Exception as e:
        logger.info("Starting the Flask server")
        sock.close()
        script_path = os.path.join(now_dir, "lib", "tools", "server.py")
        try:
            subprocess.Popen(f"python {script_path}", shell=True)
            logger.info("Flask server started!")
        except Exception as e:
            logger.error("Failed to start the Flask server")
            logger.error(e)

os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs/weights"), exist_ok=True)
os.environ["temp"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)
logging.getLogger("numba").setLevel(logging.WARNING)

if not os.path.isdir("lib/csvdb/"):
    os.makedirs("lib/csvdb")
    frmnt, stp = open("lib/csvdb/formanting.csv", "w"), open("lib/csvdb/stop.csv", "w")
    frmnt.close()
    stp.close()

global DoFormant, Quefrency, Timbre

try:
    DoFormant, Quefrency, Timbre = CSVutil(
        "lib/csvdb/formanting.csv", "r", "formanting"
    )
    DoFormant = (
        lambda DoFormant: True
        if DoFormant.lower() == "true"
        else (False if DoFormant.lower() == "false" else DoFormant)
    )(DoFormant)
except (ValueError, TypeError, IndexError):
    DoFormant, Quefrency, Timbre = False, 1.0, 1.0
    CSVutil(
        "lib/csvdb/formanting.csv", "w+", "formanting", DoFormant, Quefrency, Timbre
    )

load_dotenv()
config = Config()
vc = VC(config)

if config.dml == True:

    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res

    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml

i18n = I18nAuto()
i18n.print()
# 判断是否有能用来训练和加速推理的N卡
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

isinterrupted = 0


if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_infos.append("%s\t%s" % (i, gpu_name))
        mem.append(
            int(
                torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024
                + 0.4
            )
        )
if len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = (
        "Unfortunately, there is no compatible GPU available to support your training."
    )
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


import lib.infer.infer_libs.uvr5_pack.mdx as mdx
from lib.infer.modules.uvr5.mdxprocess import (
    get_model_list,
    get_demucs_model_list,
    id_to_ptm,
    prepare_mdx,
    run_mdx,
)

hubert_model = None
weight_root = os.getenv("weight_root")
weight_uvr5_root = os.getenv("weight_uvr5_root")
index_root = os.getenv("index_root")
datasets_root = "datasets"
fshift_root = "lib/infer/infer_libs/formantshiftcfg"
audio_root = "assets/audios"
audio_others_root = "assets/audios/audio-others"
sup_audioext = {
    "wav",
    "mp3",
    "flac",
    "ogg",
    "opus",
    "m4a",
    "mp4",
    "aac",
    "alac",
    "wma",
    "aiff",
    "webm",
    "ac3",
}

names = [
    os.path.join(root, file)
    for root, _, files in os.walk(weight_root)
    for file in files
    if file.endswith((".pth", ".onnx"))
]

indexes_list = [
    os.path.join(root, name)
    for root, _, files in os.walk(index_root, topdown=False)
    for name in files
    if name.endswith(".index") and "trained" not in name
]

audio_paths = [
    os.path.join(root, name)
    for root, _, files in os.walk(audio_root, topdown=False)
    for name in files
    if name.endswith(tuple(sup_audioext)) and root == audio_root
]

audio_others_paths = [
    os.path.join(root, name)
    for root, _, files in os.walk(audio_others_root, topdown=False)
    for name in files
    if name.endswith(tuple(sup_audioext)) and root == audio_others_root
]

check_for_name = lambda: sorted(names)[0] if names else ""

datasets = []
for foldername in os.listdir(os.path.join(now_dir, datasets_root)):
    if os.path.isdir(os.path.join(now_dir, "datasets", foldername)):
        datasets.append(foldername)

def get_dataset():
    if len(datasets) > 0:
        return sorted(datasets)[0]
    else:
        return ""

def change_dataset(
        trainset_dir4
):
    return gr.Textbox.update(value=trainset_dir4)

uvr5_names = ["HP2_all_vocals.pth", "HP3_all_vocals.pth", "HP5_only_main_vocal.pth",
             "VR-DeEchoAggressive.pth", "VR-DeEchoDeReverb.pth", "VR-DeEchoNormal.pth"]

__s = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/"

def id_(mkey):
    if mkey in uvr5_names:
        model_name, ext = os.path.splitext(mkey)
        mpath = f"{now_dir}/assets/uvr5_weights/{mkey}"
        if not os.path.exists(f'{now_dir}/assets/uvr5_weights/{mkey}'):
            print('Downloading model...',end=' ')
            subprocess.run(
                ["python", "-m", "wget", "-o", mpath, __s+mkey]
            )
            print(f'saved to {mpath}')
            return model_name
        else:
            return model_name
    else:
        return None

def update_model_choices(select_value):
    model_ids = get_model_list()
    model_ids_list = list(model_ids)
    demucs_model_ids = get_demucs_model_list()
    demucs_model_ids_list = list(demucs_model_ids)
    if select_value == "VR":
        return {"choices": uvr5_names, "__type__": "update"}
    elif select_value == "MDX":
        return {"choices": model_ids_list, "__type__": "update"}
    elif select_value == "Demucs (Beta)":
        return {"choices": demucs_model_ids_list, "__type__": "update"}


def update_dataset_list(name):
    new_datasets = []
    for foldername in os.listdir(os.path.join(now_dir, datasets_root)):
        if os.path.isdir(os.path.join(now_dir, "datasets", foldername)):
            new_datasets.append(
                os.path.join(
                    now_dir,
                    "datasets",
                    foldername,
                )
            )
    return gr.Dropdown.update(choices=new_datasets)


def get_indexes():
    indexes_list = [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(index_root)
        for filename in filenames
        if filename.endswith(".index") and "trained" not in filename
    ]

    return indexes_list if indexes_list else ""


def get_fshift_presets():
    fshift_presets_list = [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(fshift_root)
        for filename in filenames
        if filename.endswith(".txt")
    ]

    return fshift_presets_list if fshift_presets_list else ""

def uvr(
    model_name,
    inp_root,
    save_root_vocal,
    paths,
    save_root_ins,
    agg,
    format0,
    architecture,
):
    infos = []
    if architecture == "VR":
        try:
            
            inp_root = inp_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            save_root_vocal = (
                save_root_vocal.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            )
            save_root_ins = (
                save_root_ins.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            )

            model_name = id_(model_name)
            if model_name == None:
                return ""
            else:
                pass
            
            infos.append(
                i18n("Starting audio conversion... (This might take a moment)")
            )

            if model_name == "onnx_dereverb_By_FoxJoy":
                pre_fun = MDXNetDereverb(15, config.device)
            else:
                func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
                pre_fun = func(
                    agg=int(agg),
                    model_path=os.path.join(
                        os.getenv("weight_uvr5_root"), model_name + ".pth"
                    ),
                    device=config.device,
                    is_half=config.is_half,
                )
            if inp_root != "":
                paths = [
                    os.path.join(inp_root, name)
                    for root, _, files in os.walk(inp_root, topdown=False)
                    for name in files
                    if name.endswith(tuple(sup_audioext)) and root == inp_root
                ]
            else:
                paths = [path.name for path in paths]
            for path in paths:
                inp_path = os.path.join(inp_root, path)
                need_reformat = 1
                done = 0
                try:
                    info = ffmpeg.probe(inp_path, cmd="ffprobe")
                    if (
                        info["streams"][0]["channels"] == 2
                        and info["streams"][0]["sample_rate"] == "44100"
                    ):
                        need_reformat = 0
                        pre_fun._path_audio_(
                            inp_path, save_root_ins, save_root_vocal, format0
                        )
                        done = 1
                except:
                    need_reformat = 1
                    traceback.print_exc()
                if need_reformat == 1:
                    tmp_path = "%s/%s.reformatted.wav" % (
                        os.path.join(os.environ["tmp"]),
                        os.path.basename(inp_path),
                    )
                    os.system(
                        "ffmpeg -i %s -vn -acodec pcm_s16le -ac 2 -ar 44100 %s -y"
                        % (inp_path, tmp_path)
                    )
                    inp_path = tmp_path
                try:
                    if done == 0:
                        pre_fun.path_audio(
                            inp_path, save_root_ins, save_root_vocal, format0
                        )
                    infos.append("%s->Success" % (os.path.basename(inp_path)))
                    yield "\n".join(infos)
                except:
                    try:
                        if done == 0:
                            pre_fun._path_audio_(
                                inp_path, save_root_ins, save_root_vocal, format0
                            )
                        infos.append("%s->Success" % (os.path.basename(inp_path)))
                        yield "\n".join(infos)
                    except:
                        infos.append(
                            "%s->%s"
                            % (os.path.basename(inp_path), traceback.format_exc())
                        )
                        yield "\n".join(infos)
        except:
            infos.append(traceback.format_exc())
            yield "\n".join(infos)
        finally:
            try:
                if model_name == "onnx_dereverb_By_FoxJoy":
                    del pre_fun.pred.model
                    del pre_fun.pred.model_
                else:
                    del pre_fun.model
                    del pre_fun
            except:
                traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Executed torch.cuda.empty_cache()")
        yield "\n".join(infos)
    elif architecture == "MDX":
        try:
            infos.append(
                i18n("Starting audio conversion... (This might take a moment)")
            )
            yield "\n".join(infos)
            inp_root, save_root_vocal, save_root_ins = [
                x.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
                for x in [inp_root, save_root_vocal, save_root_ins]
            ]

            if inp_root != "":
                paths = [
                    os.path.join(inp_root, name)
                    for root, _, files in os.walk(inp_root, topdown=False)
                    for name in files
                    if name.endswith(tuple(sup_audioext)) and root == inp_root
                ]
            else:
                paths = [path.name for path in paths]
            print(paths)
            invert = True
            denoise = True
            use_custom_parameter = True
            dim_f = 3072
            dim_t = 256
            n_fft = 7680
            use_custom_compensation = True
            compensation = 1.025
            suffix = "Vocals_custom"  # @param ["Vocals", "Drums", "Bass", "Other"]{allow-input: true}
            suffix_invert = "Instrumental_custom"  # @param ["Instrumental", "Drumless", "Bassless", "Instruments"]{allow-input: true}
            print_settings = True  # @param{type:"boolean"}
            onnx = id_to_ptm(model_name)
            compensation = (
                compensation
                if use_custom_compensation or use_custom_parameter
                else None
            )
            mdx_model = prepare_mdx(
                onnx,
                use_custom_parameter,
                dim_f,
                dim_t,
                n_fft,
                compensation=compensation,
            )

            for path in paths:
                # inp_path = os.path.join(inp_root, path)
                suffix_naming = suffix if use_custom_parameter else None
                diff_suffix_naming = suffix_invert if use_custom_parameter else None
                run_mdx(
                    onnx,
                    mdx_model,
                    path,
                    format0,
                    diff=invert,
                    suffix=suffix_naming,
                    diff_suffix=diff_suffix_naming,
                    denoise=denoise,
                )

            if print_settings:
                print()
                print("[MDX-Net_Colab settings used]")
                print(f"Model used: {onnx}")
                print(f"Model MD5: {mdx.MDX.get_hash(onnx)}")
                print(f"Model parameters:")
                print(f"    -dim_f: {mdx_model.dim_f}")
                print(f"    -dim_t: {mdx_model.dim_t}")
                print(f"    -n_fft: {mdx_model.n_fft}")
                print(f"    -compensation: {mdx_model.compensation}")
                print()
                print("[Input file]")
                print("filename(s): ")
                for filename in paths:
                    print(f"    -{filename}")
                    infos.append(f"{os.path.basename(filename)}->Success")
                    yield "\n".join(infos)
        except:
            infos.append(traceback.format_exc())
            yield "\n".join(infos)
        finally:
            try:
                del mdx_model
            except:
                traceback.print_exc()

            print("clean_empty_cache")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    elif architecture == "Demucs (Beta)":
        try:
            infos.append(
                i18n("Starting audio conversion... (This might take a moment)")
            )
            yield "\n".join(infos)
            inp_root, save_root_vocal, save_root_ins = [
                x.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
                for x in [inp_root, save_root_vocal, save_root_ins]
            ]

            if inp_root != "":
                paths = [
                    os.path.join(inp_root, name)
                    for root, _, files in os.walk(inp_root, topdown=False)
                    for name in files
                    if name.endswith(tuple(sup_audioext)) and root == inp_root
                ]
            else:
                paths = [path.name for path in paths]

            # Loop through the audio files and separate sources
            for path in paths:
                input_audio_path = os.path.join(inp_root, path)
                filename_without_extension = os.path.splitext(
                    os.path.basename(input_audio_path)
                )[0]
                _output_dir = os.path.join(tmp, model_name, filename_without_extension)
                vocals = os.path.join(_output_dir, "vocals.wav")
                no_vocals = os.path.join(_output_dir, "no_vocals.wav")

                os.makedirs(tmp, exist_ok=True)

                if torch.cuda.is_available():
                    cpu_insted = ""
                else:
                    cpu_insted = "-d cpu"
                print(cpu_insted)

                # Use with os.system  to separate audio sources becuase at invoking from the command line it is faster than invoking from python
                os.system(
                    f"python -m .separate --two-stems=vocals -n {model_name} {cpu_insted} {input_audio_path} -o {tmp}"
                )

                # Move vocals and no_vocals to the output directory assets/audios for the vocal and assets/audios/audio-others for the instrumental
                shutil.move(vocals, save_root_vocal)
                shutil.move(no_vocals, save_root_ins)

                # And now rename the vocals and no vocals with the name of the input audio file and the suffix vocals or instrumental
                os.rename(
                    os.path.join(save_root_vocal, "vocals.wav"),
                    os.path.join(
                        save_root_vocal, f"{filename_without_extension}_vocals.wav"
                    ),
                )
                os.rename(
                    os.path.join(save_root_ins, "no_vocals.wav"),
                    os.path.join(
                        save_root_ins, f"{filename_without_extension}_instrumental.wav"
                    ),
                )

                # Remove the temporary directory
                os.rmdir(tmp, model_name)

                infos.append(f"{os.path.basename(input_audio_path)}->Success")
                yield "\n".join(infos)

        except:
            infos.append(traceback.format_exc())
            yield "\n".join(infos)


def change_choices():
    names = [
        os.path.join(root, file)
        for root, _, files in os.walk(weight_root)
        for file in files
        if file.endswith((".pth", ".onnx"))
    ]
    indexes_list = [
        os.path.join(root, name)
        for root, _, files in os.walk(index_root, topdown=False)
        for name in files
        if name.endswith(".index") and "trained" not in name
    ]
    audio_paths = [
        os.path.join(root, name)
        for root, _, files in os.walk(audio_root, topdown=False)
        for name in files
        if name.endswith(tuple(sup_audioext)) and root == audio_root
    ]

    return (
        {"choices": sorted(names), "__type__": "update"},
        {"choices": sorted(indexes_list), "__type__": "update"},
        {"choices": sorted(audio_paths), "__type__": "update"},
    )


def change_choices2():
    names = [
        os.path.join(root, file)
        for root, _, files in os.walk(weight_root)
        for file in files
        if file.endswith((".pth", ".onnx"))
    ]
    indexes_list = [
        os.path.join(root, name)
        for root, _, files in os.walk(index_root, topdown=False)
        for name in files
        if name.endswith(".index") and "trained" not in name
    ]

    return (
        {"choices": sorted(names), "__type__": "update"},
        {"choices": sorted(indexes_list), "__type__": "update"},
    )


def clean():
    return {"value": "", "__type__": "update"}


def export_onnx():
    from lib.infer.modules.onnx.export import export_onnx as eo

    eo()


sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


def if_done(done, p):
    while 1:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True


def if_done_multi(done, ps):
    while 1:
        # poll==None代表进程未结束
        # 只要有一个进程未结束都不停
        flag = 1
        for p in ps:
            if p.poll() is None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True


def formant_enabled(
    cbox, qfrency, tmbre, frmntapply, formantpreset, formant_refresh_button
):
    if cbox:
        DoFormant = True
        CSVutil(
            "lib/csvdb/formanting.csv", "w+", "formanting", DoFormant, qfrency, tmbre
        )

        # print(f"is checked? - {cbox}\ngot {DoFormant}")

        return (
            {"value": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
        )

    else:
        DoFormant = False
        CSVutil(
            "lib/csvdb/formanting.csv", "w+", "formanting", DoFormant, qfrency, tmbre
        )

        # print(f"is checked? - {cbox}\ngot {DoFormant}")
        return (
            {"value": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
        )


def formant_apply(qfrency, tmbre):
    Quefrency = qfrency
    Timbre = tmbre
    DoFormant = True
    CSVutil("lib/csvdb/formanting.csv", "w+", "formanting", DoFormant, qfrency, tmbre)

    return (
        {"value": Quefrency, "__type__": "update"},
        {"value": Timbre, "__type__": "update"},
    )


def update_fshift_presets(preset, qfrency, tmbre):
    if preset:
        with open(preset, "r") as p:
            content = p.readlines()
            qfrency, tmbre = content[0].strip(), content[1]

        formant_apply(qfrency, tmbre)
    else:
        qfrency, tmbre = preset_apply(preset, qfrency, tmbre)

    return (
        {"choices": get_fshift_presets(), "__type__": "update"},
        {"value": qfrency, "__type__": "update"},
        {"value": tmbre, "__type__": "update"},
    )


def preprocess_dataset(trainset_dir, exp_dir, sr, dataset_path):
    if re.search(r"[^0-9a-zA-Z !@#$%^&\(\)_+=\-`~\[\]\{\};',.]", exp_dir):
        raise gr.Error("Model name contains non-ASCII characters!")
    if not dataset_path.strip() == "":
        trainset_dir = dataset_path
    else:
        trainset_dir = os.path.join(now_dir, "datasets", trainset_dir)
    sr = sr_dict[sr]
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "w")
    f.close()
    per = 3.0 if config.is_half else 3.7
    cmd = (
        '"%s" lib/infer/modules/train/preprocess.py "%s" %s %s "%s/logs/%s" %s %.1f'
        % (
            config.python_cmd,
            trainset_dir,
            sr,
            rvc_globals.CpuCores,
            now_dir,
            exp_dir,
            config.noparallel,
            per,
        )
    )
    logger.info(cmd)
    p = Popen(cmd, shell=True)  # , stdin=PIPE, stdout=PIPE,stderr=PIPE,cwd=now_dir
    ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    logger.info(log)
    yield log


def extract_f0_feature(f0method, if_f0, exp_dir, version19, echl):
    if re.search(r"[^0-9a-zA-Z !@#$%^&\(\)_+=\-`~\[\]\{\};',.]", exp_dir):
        raise gr.Error("Model name contains non-ASCII characters!")
    gpus_raw = rvc_globals.GpuIds
    gpus = gpus_raw.split(",")
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "w")
    f.close()
    if if_f0:
        if f0method != "rmvpe-gpu":
            cmd = (
                '"%s" lib/infer/modules/train/extract/extract_f0_print.py "%s/logs/%s" %s %s %s'
                % (config.python_cmd, now_dir, exp_dir, rvc_globals.CpuCores, f0method, RQuote(echl))
            )
            logger.info(cmd)
            p = Popen(
                cmd, shell=True, cwd=now_dir
            )  # , stdin=PIPE, stdout=PIPE,stderr=PIPE
            ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
            done = [False]
            threading.Thread(
                target=if_done,
                args=(
                    done,
                    p,
                ),
            ).start()
        else:
            if gpus_raw != "-":
                gpus_raw = gpus_raw.split(",")
                leng = len(gpus_raw)
                ps = []
                for idx, n_g in enumerate(gpus_raw):
                    cmd = (
                        '"%s" lib/infer/modules/train/extract/extract_f0_rmvpe.py %s %s %s "%s/logs/%s" %s '
                        % (
                            config.python_cmd,
                            leng,
                            idx,
                            n_g,
                            now_dir,
                            exp_dir,
                            config.is_half,
                        )
                    )
                    logger.info(cmd)
                    p = Popen(
                        cmd, shell=True, cwd=now_dir
                    )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
                    ps.append(p)
                ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
                done = [False]
                threading.Thread(
                    target=if_done_multi,  #
                    args=(
                        done,
                        ps,
                    ),
                ).start()
            else:
                cmd = (
                    config.python_cmd
                    + ' lib/infer/modules/train/extract/extract_f0_rmvpe_dml.py "%s/logs/%s" '
                    % (
                        now_dir,
                        exp_dir,
                    )
                )
                logger.info(cmd)
                p = Popen(
                    cmd, shell=True, cwd=now_dir
                )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
                p.wait()
                done = [True]
        while 1:
            with open(
                "%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r"
            ) as f:
                yield (f.read())
            sleep(1)
            if done[0]:
                break
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            log = f.read()
        logger.info(log)
        yield log
    ####对不同part分别开多进程
    """
    n_part=int(sys.argv[1])
    i_part=int(sys.argv[2])
    i_gpu=sys.argv[3]
    exp_dir=sys.argv[4]
    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)
    """
    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = (
            '"%s" lib/infer/modules/train/extract_feature_print.py %s %s %s %s "%s/logs/%s" %s %s'
            % (
                config.python_cmd,
                config.device,
                leng,
                idx,
                n_g,
                now_dir,
                exp_dir,
                version19,
                config.is_half,
            )
        )
        logger.info(cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
        ps.append(p)
    ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done_multi,
        args=(
            done,
            ps,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    logger.info(log)
    yield log


def get_pretrained_models(path_str, f0_str, sr2):
    if_pretrained_generator_exist = os.access(
        "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if_pretrained_discriminator_exist = os.access(
        "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if not if_pretrained_generator_exist:
        logger.warn(
            "assets/pretrained%s/%sG%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    if not if_pretrained_discriminator_exist:
        logger.warn(
            "assets/pretrained%s/%sD%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    return (
        "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2)
        if if_pretrained_generator_exist
        else "",
        "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2)
        if if_pretrained_discriminator_exist
        else "",
    )


def change_sr2(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    return get_pretrained_models(path_str, f0_str, sr2)


def change_version19(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    if sr2 == "32k" and version19 == "v1":
        sr2 = "40k"
    to_return_sr2 = (
        {"choices": ["40k", "48k"], "__type__": "update", "value": sr2}
        if version19 == "v1"
        else {"choices": ["40k", "48k", "32k"], "__type__": "update", "value": sr2}
    )
    f0_str = "f0" if if_f0_3 else ""
    return (
        *get_pretrained_models(path_str, f0_str, sr2),
        to_return_sr2,
    )


def change_f0(if_f0_3, sr2, version19):  # f0method8,pretrained_G14,pretrained_D15
    path_str = "" if version19 == "v1" else "_v2"
    return (
        {"visible": if_f0_3, "__type__": "update"},
        *get_pretrained_models(path_str, "f0", sr2),
    )


global log_interval


def set_log_interval(exp_dir, batch_size12):
    log_interval = 1
    folder_path = os.path.join(exp_dir, "1_16k_wavs")

    if os.path.isdir(folder_path):
        wav_files_num = len(glob1(folder_path, "*.wav"))

        if wav_files_num > 0:
            log_interval = math.ceil(wav_files_num / batch_size12)
            if log_interval > 1:
                log_interval += 1

    return log_interval


global PID, PROCESS, TB


def click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
    if_retrain_collapse20,
    if_stop_on_fit21,
    stop_on_fit_grace22,
    smoothness23,
    collapse_threshold24
):
    CSVutil("lib/csvdb/stop.csv", "w+", "formanting", False)
    # 生成filelist
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    log_interval = set_log_interval(exp_dir, batch_size12)

    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    logger.debug("Write filelist done")
    logger.info("Use gpus: %s", rvc_globals.GpuIds)
    if pretrained_G14 == "":
        logger.info("No pretrained Generator")
    if pretrained_D15 == "":
        logger.info("No pretrained Discriminator")
    if version19 == "v1" or sr2 == "40k":
        config_path = "v1/%s.json" % sr2
    else:
        config_path = "v2/%s.json" % sr2
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(
                config.json_config[config_path],
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")
    cmd = (
        '"%s" lib/infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s %s %s'
        % (
            config.python_cmd,
            exp_dir1,
            sr2,
            1 if if_f0_3 else 0,
            batch_size12,
            ("-g %s" % rvc_globals.GpuIds) if rvc_globals.GpuIds else "",
            total_epoch11,
            save_epoch10,
            "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
            "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
            1 if if_save_latest13 == True else 0,
            1 if if_cache_gpu17 == True else 0,
            1 if if_save_every_weights18 == True else 0,
            version19,
            ("-sof %s -sofg %s -sm %s" % (1 if if_stop_on_fit21 == True else 0, stop_on_fit_grace22, smoothness23)) if if_stop_on_fit21 else "",
            ("-rc %s -ct %s" % (1 if if_retrain_collapse20 == True else 0, collapse_threshold24)) if if_retrain_collapse20 else "",
        )
    )
    logger.info(cmd)
    global p, PID
    p = Popen(cmd, shell=True, cwd=now_dir)
    PID = p.pid

    p.wait()
    batchSize = batch_size12
    colEpoch = 0
    while if_retrain_collapse20:
        if not os.path.exists(f"logs/{exp_dir1}/col"):
            break
        with open(f"logs/{exp_dir1}/col") as f:
            col = f.read().split(',')
            if colEpoch < int(col[1]):
                colEpoch = int(col[1])
                logger.info(f"Epoch to beat {col[1]}")
                if batchSize != batch_size12:
                    batchSize = batch_size12 + 1
            batchSize -= 1
        if batchSize < 1:
            break
        p = Popen(cmd.replace(f"-bs {batch_size12}", f"-bs {batchSize}"), shell=True, cwd=now_dir)
        PID = p.pid
        p.wait()
        
    return (
        i18n("Training is done, check train.log"),
        {"visible": False, "__type__": "update"},
        {"visible": True, "__type__": "update"},
    )


def train_index(exp_dir1, version19):
    exp_dir = os.path.join(now_dir, "logs", exp_dir1)
    # exp_dir = "logs/%s" % (exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if not os.path.exists(feature_dir):
        return "Please do the feature extraction first"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "Please perform the feature extraction first"
    infos = []
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 2e5:
        infos.append("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
        yield "\n".join(infos)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            logger.info(info)
            infos.append(info)
            yield "\n".join(infos)

    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    # infos.append("%s,%s" % (big_npy.shape, n_ivf))
    yield "\n".join(infos)
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    # index = faiss.index_factory(256if version19=="v1"else 768, "IVF%s,PQ128x4fs,RFlat"%n_ivf)
    infos.append("Generating training file...")
    print("Generating training file...")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )

    infos.append("Generating adding file...")
    print("Generating adding file...")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    infos.append("Files generated successfully!")
    print("Files generated successfully!")


def change_info_(ckpt_path):
    if not os.path.exists(ckpt_path.replace(os.path.basename(ckpt_path), "train.log")):
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
    try:
        with open(
            ckpt_path.replace(os.path.basename(ckpt_path), "train.log"), "r"
        ) as f:
            info = eval(f.read().strip("\n").split("\n")[0].split("\t")[-1])
            sr, f0 = info["sample_rate"], info["if_f0"]
            version = "v2" if ("version" in info and info["version"] == "v2") else "v1"
            return sr, str(f0), version
    except:
        traceback.print_exc()
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}


F0GPUVisible = config.dml == False


def execute_generator_function(genObject):
    for _ in genObject:
        pass


def preset_apply(preset, qfer, tmbr):
    if str(preset) != "":
        with open(str(preset), "r") as p:
            content = p.readlines()
            qfer, tmbr = content[0].split("\n")[0], content[1]
            formant_apply(qfer, tmbr)
    else:
        pass
    return (
        {"value": qfer, "__type__": "update"},
        {"value": tmbr, "__type__": "update"},
    )


def switch_pitch_controls(f0method0):
    is_visible = f0method0 != "rmvpe"

    if rvc_globals.NotesOrHertz:
        return (
            {"visible": False, "__type__": "update"},
            {"visible": is_visible, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": is_visible, "__type__": "update"},
        )
    else:
        return (
            {"visible": is_visible, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": is_visible, "__type__": "update"},
            {"visible": False, "__type__": "update"},
        )


def match_index(sid0: str) -> str:
    if not sid0:
        return ""

    sid0strip = re.sub(r"\.pth|\.onnx$", "", sid0)
    sid0name = os.path.split(sid0strip)[-1]  # Extract only the name, not the directory

    # Check if the sid0strip has the specific ending format _eXXX_sXXX
    if re.match(r".+_e\d+_s\d+$", sid0name):
        base_model_name = sid0name.rsplit("_", 2)[0]
    else:
        base_model_name = sid0name

    sid_directory = os.path.join(index_root, base_model_name)
    directories_to_search = [sid_directory] if os.path.exists(sid_directory) else []
    directories_to_search.append(index_root)

    matching_index_files = []

    for directory in directories_to_search:
        for filename in os.listdir(directory):
            if filename.endswith(".index") and "trained" not in filename:
                # Condition to match the name
                name_match = any(
                    name.lower() in filename.lower()
                    for name in [sid0name, base_model_name]
                )

                # If in the specific directory, it's automatically a match
                folder_match = directory == sid_directory

                if name_match or folder_match:
                    index_path = os.path.join(directory, filename)
                    if index_path in indexes_list:
                        matching_index_files.append(
                            (
                                index_path,
                                os.path.getsize(index_path),
                                " " not in filename,
                            )
                        )

    if matching_index_files:
        # Sort by favoring files without spaces and by size (largest size first)
        matching_index_files.sort(key=lambda x: (-x[2], -x[1]))
        best_match_index_path = matching_index_files[0][0]
        return best_match_index_path

    return ""


def stoptraining(mim):
    if int(mim) == 1:
        CSVutil("lib/csvdb/stop.csv", "w+", "stop", "True")
        # p.terminate()
        # p.kill()
        try:
            os.kill(PID, signal.SIGTERM)
        except Exception as e:
            print(f"Couldn't click due to {e}")
            pass
    else:
        pass

    return (
        {"visible": False, "__type__": "update"},
        {"visible": True, "__type__": "update"},
    )


weights_dir = "weights/"


def note_to_hz(note_name):
    SEMITONES = {
        "C": -9,
        "C#": -8,
        "D": -7,
        "D#": -6,
        "E": -5,
        "F": -4,
        "F#": -3,
        "G": -2,
        "G#": -1,
        "A": 0,
        "A#": 1,
        "B": 2,
    }
    pitch_class, octave = note_name[:-1], int(note_name[-1])
    semitone = SEMITONES[pitch_class]
    note_number = 12 * (octave - 4) + semitone
    frequency = 440.0 * (2.0 ** (1.0 / 12)) ** note_number
    return frequency


def save_to_wav(record_button):
    if record_button is None:
        pass
    else:
        path_to_file = record_button
        new_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".wav"
        target_path = os.path.join("assets", "audios", os.path.basename(new_name))

        shutil.move(path_to_file, target_path)
        return target_path


def save_to_wav2_edited(dropbox):
    if dropbox is None:
        pass
    else:
        file_path = dropbox.name
        target_path = os.path.join("assets", "audios", os.path.basename(file_path))

        if os.path.exists(target_path):
            os.remove(target_path)
            print("Replacing old dropdown file...")

        shutil.move(file_path, target_path)
    return


def save_to_wav2(dropbox):
    file_path = dropbox.name
    target_path = os.path.join("assets", "audios", os.path.basename(file_path))

    if os.path.exists(target_path):
        os.remove(target_path)
        print("Replacing old dropdown file...")

    shutil.move(file_path, target_path)
    return target_path


import lib.tools.loader_themes as loader_themes

my_applio = loader_themes.load_json()
if my_applio:
    pass
else:
    my_applio = "JohnSmith9982/small_and_pretty"


def GradioSetup():
    default_weight = ""

    with gr.Blocks(title="RVC-Tundra") as app:
        gr.HTML("<h2>RVC-Tundra</h2>")
        with gr.Tabs():
            with gr.TabItem(i18n("Inference")):
                with gr.Row():
                    sid0 = gr.Dropdown(
                        label=i18n("Voice weight"),
                        choices=sorted(names),
                        value=i18n("Choose a voice weight file..."),
                    )
                    file_index2 = gr.Dropdown(
                        label=i18n("Voice feature index"),
                        choices=get_indexes(),
                        interactive=True,
                        allow_custom_value=True,
                    )
                    sid0.select(
                        fn=match_index,
                        inputs=[sid0],
                        outputs=[file_index2],
                    )
                    with gr.Column():
                        refresh_button = gr.Button(i18n("Refresh list"), variant="primary")
                        clean_button = gr.Button(
                            i18n("Unload voices from GPU"), variant="primary"
                        )
                    clean_button.click(
                        fn=lambda: ({"value": "", "__type__": "update"}),
                        inputs=[],
                        outputs=[sid0],
                        api_name="infer_clean",
                    )

                with gr.TabItem(i18n("Single")):
                    with gr.Row():
                        with gr.Column():
                            with gr.Group():
                                dropbox = gr.File(label=i18n("Select an audio file"))
                                record_button = gr.Audio(
                                    source="microphone",
                                    label=i18n("Or record an audio"),
                                    type="filepath",
                                    interactive=True,
                                )
                                input_audio1 = gr.Dropdown(
                                    label=i18n(
                                        "History audio files"
                                    ),
                                    choices=sorted(audio_paths),
                                    value="",
                                    interactive=True,
                                )
                                dropbox.upload(
                                    fn=save_to_wav2,
                                    inputs=[dropbox],
                                    outputs=[input_audio1],
                                )
                                record_button.change(
                                    fn=save_to_wav,
                                    inputs=[record_button],
                                    outputs=[input_audio1],
                                )
                                refresh_button.click(
                                    fn=change_choices,
                                    inputs=[],
                                    outputs=[sid0, file_index2, input_audio1],
                                    api_name="infer_refresh",
                                )
                        with gr.Column():
                            spk_item = gr.Number(
                                minimum=0,
                                maximum=127,
                                step=1,
                                label=i18n("Speaker ID"),
                                value=0,
                                precision=0,
                                interactive=True,
                            )
                            vc_transform0 = gr.Number(
                                label=i18n(
                                    "Transpose (number of semitones, raise by an octave: 12, lower by an octave: -12)"
                                ),
                                value=0,
                            )
                            f0_autotune = gr.Checkbox(
                                label=i18n("Use autotune"),
                                info=i18n("Round pitch to the nearest note, suitable for singing"),
                                interactive=True,
                                value=False,
                            )
                            split_audio = gr.Checkbox(
                                label=i18n("Use auto segmentation"),
                                info=i18n("Split audio into segments based on silence, suppresses model noise"),
                                interactive=True,
                            )
                            format1_ = gr.Radio(
                                label=i18n("Export file format"),
                                choices=["wav", "flac", "mp3", "m4a"],
                                value="wav",
                                interactive=True,
                            )
                            advanced_settings_checkbox = gr.Checkbox(
                                value=False,
                                label=i18n("Show advanced settings"),
                                interactive=True,
                            )
                    # Advanced settings container
                    with gr.Column(
                        visible=False
                    ) as advanced_settings:  # Initially hidden
                        with gr.Row(label=i18n("Advanced settings"), open=False):
                            with gr.Column():
                                with gr.Group():
                                    f0method0 = gr.Radio(
                                        label=i18n(
                                            "Pitch extraction algorithm"
                                        ),
                                        choices=[
                                            "pm",
                                            "harvest",
                                            "dio",
                                            "crepe",
                                            "crepe-tiny",
                                            "mangio-crepe",
                                            "mangio-crepe-tiny",
                                            "rmvpe",
                                            "rmvpe+",
                                        ]
                                        if config.dml == False
                                        else [
                                            "pm",
                                            "harvest",
                                            "dio",
                                            "rmvpe",
                                            "rmvpe+",
                                        ],
                                        value="rmvpe+",
                                        interactive=True,
                                    )
                                    crepe_hop_length = gr.Slider(
                                        minimum=1,
                                        maximum=512,
                                        step=1,
                                        label=i18n("Hop length"),
                                        info=i18n("Lower hop length provides higher accuracy in pitch while takes more time"),
                                        value=120,
                                        interactive=True,
                                        visible=False,
                                    )
                                    minpitch_slider = gr.Slider(
                                        label=i18n("Min pitch (frequency)"),
                                        info=i18n(
                                            "Specify minimal pitch for inference [Hz]"
                                        ),
                                        step=0.1,
                                        minimum=1,
                                        scale=0,
                                        value=50,
                                        maximum=16000,
                                        interactive=True,
                                        visible=(not rvc_globals.NotesOrHertz)
                                        and (f0method0.value != "rmvpe"),
                                    )
                                    minpitch_txtbox = gr.Textbox(
                                        label=i18n("Min pitch (note)"),
                                        info=i18n(
                                            "Specify minimal pitch for inference [NOTE]"
                                        ),
                                        placeholder="C5",
                                        visible=(rvc_globals.NotesOrHertz)
                                        and (f0method0.value != "rmvpe"),
                                        interactive=True,
                                    )
                                    maxpitch_slider = gr.Slider(
                                        label=i18n("Max pitch (frequency)"),
                                        info=i18n("Specify max pitch for inference [Hz]"),
                                        step=0.1,
                                        minimum=1,
                                        scale=0,
                                        value=1100,
                                        maximum=16000,
                                        interactive=True,
                                        visible=(not rvc_globals.NotesOrHertz)
                                        and (f0method0.value != "rmvpe"),
                                    )
                                    maxpitch_txtbox = gr.Textbox(
                                        label=i18n("Max pitch (note)"),
                                        info=i18n("Specify max pitch for inference [NOTE]"),
                                        placeholder="C6",
                                        visible=(rvc_globals.NotesOrHertz)
                                        and (f0method0.value != "rmvpe"),
                                        interactive=True,
                                    )
                                    f0_file = gr.File(
                                        label=i18n(
                                            "Pitch guidance file (optional). One pitch per line. Overwrites all F0 and pitch modulation."
                                        )
                                    )
                            f0method0.change(
                                fn=lambda radio: (
                                    {
                                        "visible": radio
                                        in ["mangio-crepe", "mangio-crepe-tiny"],
                                        "__type__": "update",
                                    }
                                ),
                                inputs=[f0method0],
                                outputs=[crepe_hop_length],
                            )
                            f0method0.change(
                                fn=switch_pitch_controls,
                                inputs=[f0method0],
                                outputs=[
                                    minpitch_slider,
                                    minpitch_txtbox,
                                    maxpitch_slider,
                                    maxpitch_txtbox,
                                ],
                            )
                            with gr.Column():
                                resample_sr0 = gr.Slider(
                                    minimum=0,
                                    maximum=48000,
                                    label=i18n("Output sample rate"),
                                    info=i18n(
                                        "Resample the output audio before output. 0 = no resampling"
                                    ),
                                    value=0,
                                    step=1,
                                    interactive=True,
                                )
                                rms_mix_rate0 = gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    label=i18n(
                                        "Output volume envelope ratio"
                                    ),
                                    info=i18n(
                                        "Use input volume to envelope output volume. 0.0 = no envelope, 1.0 = full envelope"
                                    ),
                                    value=0.25,
                                    interactive=True,
                                )
                                protect0 = gr.Slider(
                                    minimum=0,
                                    maximum=0.5,
                                    label=i18n(
                                        "Voiceless consonant protection ratio"
                                    ),
                                    info=i18n(
                                        "Protect voiceless consonants and breath sounds to prevent artifacts like tearing in electronic music. 0.5 = disable, lower = more protection strength (can reduce indexing accuracy)"
                                    ),
                                    value=0.33,
                                    step=0.01,
                                    interactive=True,
                                )
                                filter_radius0 = gr.Slider(
                                    minimum=0,
                                    maximum=7,
                                    label=i18n(
                                        "Pitch harvest smoothing radius"
                                    ),
                                    info=i18n(
                                        "Apply median filtering to the harvested pitch results if value >= 3. higher = reduce breathiness, lower = more pitch accuracy"
                                    ),
                                    value=3,
                                    step=1,
                                    interactive=True,
                                )
                                index_rate1 = gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    label=i18n("Feature index search ratio"),
                                    info=i18n(
                                        "The ratio of feature from Faiss index search. 0.0 = do not use index search, 1.0 = all"
                                    ),
                                    value=0.75,
                                    interactive=True,
                                )
                                with gr.Group():
                                    formanting = gr.Checkbox(
                                        value=bool(DoFormant),
                                        label=i18n("Audio formant shifting"),
                                        info=i18n(
                                            "Use it for cross-gender (male to female and vice-versa) voice conversions"
                                        ),
                                        interactive=True,
                                        visible=True,
                                    )
                                    formant_preset = gr.Dropdown(
                                        value="",
                                        choices=get_fshift_presets(),
                                        label=i18n("Presets"),
                                        info=i18n(
                                            "f2m: female to male, m2f: male to female, random: noise"
                                        ),
                                        visible=bool(DoFormant),
                                    )
                                    formant_refresh_button = gr.Button(
                                        value="Refresh list",
                                        visible=bool(DoFormant),
                                        variant="primary",
                                    )
                                    qfrency = gr.Slider(
                                        value=Quefrency,
                                        label=i18n("Quefrency (ms)"),
                                        info=i18n("Used for formant preservation. See https://github.com/jurihock/stftPitchShift"),
                                        minimum=0.0,
                                        maximum=16.0,
                                        step=0.1,
                                        visible=bool(DoFormant),
                                        interactive=True,
                                    )
                                    tmbre = gr.Slider(
                                        value=Timbre,
                                        label=i18n("Timbre shifting"),
                                        info=i18n("Fractional timbre shifting factor related to quefrency. default = 1.0"),
                                        minimum=0.0,
                                        maximum=16.0,
                                        step=0.1,
                                        visible=bool(DoFormant),
                                        interactive=True,
                                    )
                                    frmntbut = gr.Button(
                                        "Apply", variant="primary", visible=bool(DoFormant)
                                    )

                            formant_preset.change(
                                fn=preset_apply,
                                inputs=[formant_preset, qfrency, tmbre],
                                outputs=[qfrency, tmbre],
                            )
                            formanting.change(
                                fn=formant_enabled,
                                inputs=[
                                    formanting,
                                    qfrency,
                                    tmbre,
                                    frmntbut,
                                    formant_preset,
                                    formant_refresh_button,
                                ],
                                outputs=[
                                    formanting,
                                    qfrency,
                                    tmbre,
                                    frmntbut,
                                    formant_preset,
                                    formant_refresh_button,
                                ],
                            )
                            frmntbut.click(
                                fn=formant_apply,
                                inputs=[qfrency, tmbre],
                                outputs=[qfrency, tmbre],
                            )
                            formant_refresh_button.click(
                                fn=update_fshift_presets,
                                inputs=[formant_preset, qfrency, tmbre],
                                outputs=[formant_preset, qfrency, tmbre],
                            )
                    def toggle_advanced_settings(checkbox):
                        return {"visible": checkbox, "__type__": "update"}
                    advanced_settings_checkbox.change(
                        fn=toggle_advanced_settings,
                        inputs=[advanced_settings_checkbox],
                        outputs=[advanced_settings],
                    )
                    but0 = gr.Button(i18n("Inference"), variant="primary").style(
                        full_width=True
                    )
                    with gr.Row():
                        vc_output1 = gr.Textbox(label=i18n("Log messages"))
                        vc_output2 = gr.Audio(
                            label=i18n(
                                "Result"
                            ),
                            show_download_button=True,
                            show_edit_button=True,
                        )
                    but0.click(
                        vc.vc_single,
                        [
                            spk_item,
                            input_audio1,
                            vc_transform0,
                            f0_file,
                            f0method0,
                            file_index2,
                            index_rate1,
                            filter_radius0,
                            resample_sr0,
                            rms_mix_rate0,
                            protect0,
                            format1_,
                            split_audio,
                            crepe_hop_length,
                            minpitch_slider,
                            minpitch_txtbox,
                            maxpitch_slider,
                            maxpitch_txtbox,
                            f0_autotune,
                        ],
                        [vc_output1, vc_output2],
                        api_name="infer_convert",
                    )

                with gr.TabItem(i18n("Batch")):
                    with gr.Row():
                        with gr.Column():
                            with gr.Group():
                                inputs = gr.File(
                                    file_count="multiple",
                                    label=i18n(
                                        "Select input audio files"
                                    ),
                                )
                                dir_input = gr.Textbox(
                                    label=i18n(
                                        "Or enter path to the folder of audio files to be processed"
                                    ),
                                    value=os.path.join(now_dir, "assets", "audios"),
                                )
                                opt_input = gr.Textbox(
                                    label=i18n("Audio output folder"), value="assets/audios/audio-outputs"
                                )
                        with gr.Column():
                            vc_transform1 = gr.Number(
                                label=i18n(
                                    "Transpose (number of semitones, raise by an octave: 12, lower by an octave: -12)"
                                ),
                                value=0,
                            )
                            f0_autotune = gr.Checkbox(
                                label="Use autotune",
                                info="Round pitch to the nearest note, suitable for singing",
                                interactive=True,
                                value=False,
                            )
                            format1 = gr.Radio(
                                label=i18n("Export file format"),
                                choices=["wav", "flac", "mp3", "m4a"],
                                value="wav",
                                interactive=True,
                            )
                            advanced_settings_batch_checkbox = gr.Checkbox(
                                value=False,
                                label=i18n("Show advanced settings"),
                                interactive=True,
                            )
                    with gr.Row():
                        with gr.Column():
                            with gr.Row(
                                visible=False
                            ) as advanced_settings_batch:
                                with gr.Row(
                                    label=i18n("Advanced settings"), open=False
                                ):
                                    with gr.Column():
                                        f0method1 = gr.Radio(
                                            label=i18n(
                                                "Pitch extraction algorithm"
                                            ),
                                            choices=[
                                                "pm",
                                                "harvest",
                                                "dio",
                                                "crepe",
                                                "crepe-tiny",
                                                "mangio-crepe",
                                                "mangio-crepe-tiny",
                                                "rmvpe",
                                            ]
                                            if config.dml == False
                                            else [
                                                "pm",
                                                "harvest",
                                                "dio",
                                                "rmvpe",
                                            ],
                                            value="rmvpe",
                                            interactive=True,
                                        )
                                with gr.Column():
                                    resample_sr1 = gr.Slider(
                                        minimum=0,
                                        maximum=48000,
                                        label=i18n("Output sample rate"),
                                        info=i18n("Resample the output audio before output. 0 = no resampling"),
                                        value=0,
                                        step=1,
                                        interactive=True,
                                    )
                                    rms_mix_rate1 = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        label=i18n("Output volume envelope ratio"),
                                        info=i18n("Use input volume to envelope output volume. 0.0 = no envelope, 1.0 = full envelope"),
                                        value=1,
                                        interactive=True,
                                    )
                                    protect1 = gr.Slider(
                                        minimum=0,
                                        maximum=0.5,
                                        label=i18n("Voiceless consonant protection ratio"),
                                        info=i18n("Protect voiceless consonants and breath sounds to prevent artifacts like tearing in electronic music. 0.5 = disable, lower = more protection strength (can reduce indexing accuracy)"),
                                        value=0.33,
                                        step=0.01,
                                        interactive=True,
                                    )
                                    filter_radius1 = gr.Slider(
                                        minimum=0,
                                        maximum=7,
                                        label=i18n("Pitch harvest smoothing radius"),
                                        info=i18n("Apply median filtering to the harvested pitch results if value >= 3. higher = reduce breathiness, lower = more pitch accuracy"),
                                        value=3,
                                        step=1,
                                        interactive=True,
                                    )
                                    index_rate2 = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        label=i18n("Feature extraction ratio"),
                                        info=i18n("The ratio of feature to be extracted from the model. 0.0 = ?, 1.0 = all"),
                                        value=0.75,
                                        interactive=True,
                                    )
                                    hop_length = gr.Slider(
                                        minimum=1,
                                        maximum=512,
                                        step=1,
                                        label=i18n("Hop length"),
                                        info=i18n("Lower hop length provides higher accuracy in pitch while takes more time"),
                                        value=120,
                                        interactive=True,
                                        visible=False,
                                    )
                            but1 = gr.Button(i18n("Inference"), variant="primary")
                            vc_output3 = gr.Textbox(label=i18n("Log messages"))
                            but1.click(
                                vc.vc_multi,
                                [
                                    spk_item,
                                    dir_input,
                                    opt_input,
                                    inputs,
                                    vc_transform1,
                                    f0method1,
                                    file_index2,
                                    index_rate2,
                                    filter_radius1,
                                    resample_sr1,
                                    rms_mix_rate1,
                                    protect1,
                                    format1,
                                    hop_length,
                                    minpitch_slider,
                                    minpitch_txtbox,
                                    maxpitch_slider,
                                    maxpitch_txtbox,
                                    f0_autotune,
                                ],
                                [vc_output3],
                                api_name="infer_convert_batch",
                            )
                    sid0.select(
                        fn=vc.get_vc,
                        inputs=[sid0, protect0, protect1],
                        outputs=[spk_item, protect0, protect1],
                        api_name="infer_change_voice",
                    )

                    # Function to toggle advanced settings
                    def toggle_advanced_settings_batch(checkbox):
                        return {"visible": checkbox, "__type__": "update"}

                    # Attach the change event
                    advanced_settings_batch_checkbox.change(
                        fn=toggle_advanced_settings_batch,
                        inputs=[advanced_settings_batch_checkbox],
                        outputs=[advanced_settings_batch],
                    )

            with gr.TabItem(i18n("Training")):
                with gr.Accordion(label=i18n("Configurations")):
                    with gr.Row():
                        with gr.Column():
                            exp_dir1 = gr.Textbox(
                                label=i18n("Model name"),
                            )
                            spk_id5 = gr.Number(
                                minimum=0,
                                maximum=127,
                                step=1,
                                label=i18n("Speaker ID"),
                                value=0,
                                precision=0,
                                interactive=True,
                            )
                        with gr.Column():
                            version19 = gr.Radio(
                                label=i18n("Model version"),
                                choices=["v1", "v2"],
                                value="v2",
                                interactive=True,
                                visible=True,
                            )
                            sr2 = gr.Radio(
                                label=i18n("Target sample rate"),
                                choices=["40k", "48k", "32k"],
                                value="40k",
                                interactive=True,
                            )
                            if_f0_3 = gr.Checkbox(
                                label=i18n("Use pitch guidance"),
                                value=True,
                                interactive=True,
                                info=i18n("Pitch guidance model can follow the pitch of the input audio"),
                            )
                        with gr.Column():
                            with gr.Group():
                                trainset_dir4 = gr.Dropdown(
                                    choices=sorted(datasets),
                                    label=i18n("Dataset"),
                                    value="",
                                )
                                trainset_dir4.change(
                                    change_dataset,
                                    [trainset_dir4],
                                    [exp_dir1]
                                )
                                dataset_path = gr.Textbox(
                                    label=i18n("OR input dataset path"),
                                    interactive=True,
                                )
                                btn_update_dataset_list = gr.Button(
                                    i18n("Refresh list"), variant="primary"
                                )

                with gr.Accordion(label=i18n("Data preprocessing")):
                    with gr.Row():
                        with gr.Group():
                            info1 = gr.Textbox(
                                label=i18n("Log messages"),
                                lines=5,
                                max_lines=5,
                                value="",
                            )
                            but1 = gr.Button(i18n("Preprocess data"), variant="primary")
                            but1.click(
                                preprocess_dataset,
                                [trainset_dir4, exp_dir1, sr2, dataset_path],
                                [info1],
                                api_name="train_preprocess",
                            )

                with gr.Accordion(label=i18n("Feature extraction")):
                    with gr.Row():
                        with gr.Column():
                            f0method8 = gr.Radio(
                                label=i18n("Pitch extraction algorithm"),
                                choices=[
                                    "pm",
                                    "harvest",
                                    "dio",
                                    "crepe",
                                    "mangio-crepe",
                                    "rmvpe",
                                    "rmvpe-gpu",
                                ]
                                if config.dml == False
                                    else [
                                        "pm",
                                        "harvest",
                                        "dio",
                                        "rmvpe",
                                        "rmvpe-gpu",
                                    ],
                                value="rmvpe",
                                interactive=True,
                            )
                            hop_length = gr.Slider(
                                minimum=1,
                                maximum=512,
                                step=1,
                                label=i18n(
                                    "Hop length"
                                ),
                                info="Lower hop length results more accuracy in pitch while takes more time",
                                value=64,
                                interactive=True,
                            )
                        with gr.Column():
                            with gr.Group():
                                info2 = gr.Textbox(
                                    label=i18n("Log messages"),
                                    value="",
                                    lines=5,
                                    max_lines=5,
                                    interactive=False,
                                )
                                but2 = gr.Button(i18n("Extract feature"), variant="primary")
                            but2.click(
                                extract_f0_feature,
                                [
                                    f0method8,
                                    if_f0_3,
                                    exp_dir1,
                                    version19,
                                    hop_length,
                                ],
                                [info2],
                                api_name="train_extract_f0_feature",
                            )

                with gr.Row():
                    with gr.Accordion(label=i18n("Model training")):
                        with gr.Row():
                            with gr.Column():
                                if_save_latest13 = gr.Checkbox(
                                    label=i18n(
                                        "Save only the latest epoch to snapshot file"
                                    ),
                                    info=i18n("The latest epoch snapshot will be saved as \"D_9999999.pth\" and \"G_9999999.pth\""),
                                    value=True,
                                    interactive=True,
                                )
                                if_cache_gpu17 = gr.Checkbox(
                                    label=i18n(
                                        "Cache all training sets to vRAM"
                                    ),
                                    info=i18n("Useful to speedup training on small (< 10min) datasets, use with caution on larger datasets"),
                                    value=False,
                                    interactive=True,
                                )
                                if_save_every_weights18 = gr.Checkbox(
                                    label=i18n(
                                        "Save final model for snapshots"
                                    ),
                                    info=i18n("Save a small final model to the \"weights\" directory at each snapshot"),
                                    value=True,
                                    interactive=True,
                                )
                                if_stop_on_fit21 = gr.Checkbox(
                                    label="Over-train detection",
                                    info=i18n(
                                        "Stop training if no improvement seen in the last N epochs"
                                    ),
                                    value=False,
                                    interactive=True,
                                )
                                if_retrain_collapse20 = gr.Checkbox(
                                    label="Mode collapse detection",
                                    info=i18n(
                                        "Detect mode collapse and restart training from the latest snapshot before it"
                                    ),
                                    value=False,
                                    interactive=True,
                                )
                            with gr.Column():
                                total_epoch11 = gr.Slider(
                                    minimum=10,
                                    maximum=4000,
                                    step=5,
                                    label=i18n("Max training epochs"),
                                    value=100,
                                    interactive=True,
                                )
                                batch_size12 = gr.Slider(
                                    minimum=1,
                                    maximum=50,
                                    step=1,
                                    label=i18n("Batch size per GPU"),
                                    value=default_batch_size,
                                    # value=20,
                                    interactive=True,
                                )
                                save_epoch10 = gr.Slider(
                                    minimum=0,
                                    maximum=100,
                                    step=1,
                                    label=i18n("Snapshot frequency (epochs)"),
                                    value=10,
                                    interactive=True,
                                    visible=True,
                                )
                                stop_on_fit_grace22 = gr.Slider(
                                    minimum=10,
                                    maximum=400,
                                    step=1,
                                    label=i18n("Grace period for over-train detection"),
                                    value=100,
                                    interactive=True,
                                    visible=False
                                )
                                smoothness23 = gr.Slider(
                                    minimum=0,
                                    maximum=0.99,
                                    step=0.005,
                                    label=i18n("Over-train improvement calculation smoothness"),
                                    value=0.975,
                                    interactive=True,
                                    visible=False
                                )
                                collapse_threshold24 = gr.Slider(
                                    minimum=1,
                                    maximum=50,
                                    step=1,
                                    label=i18n("Mode collapse detection threshold (%)"),
                                    value=25,
                                    interactive=True,
                                    visible=False
                                )
                            with gr.Column():
                                pretrained_G14 = gr.Textbox(
                                    label=i18n("Pre-trained foundation model G"),
                                    value="assets/pretrained_v2/f0G40k.pth",
                                    interactive=True,
                                )
                                pretrained_D15 = gr.Textbox(
                                    label=i18n("Pre-trained foundation model D"),
                                    value="assets/pretrained_v2/f0D40k.pth",
                                    interactive=True,
                                )
                                butstop = gr.Button(
                                    i18n("Stop training"),
                                    variant="primary",
                                    visible=False,
                                )
                                but3 = gr.Button(
                                    i18n("Train model"), variant="primary", visible=True
                                )
                                but3.click(
                                    fn=stoptraining,
                                    inputs=[gr.Number(value=0, visible=False)],
                                    outputs=[but3, butstop],
                                    api_name="train_stop",
                                )
                                butstop.click(
                                    fn=stoptraining,
                                    inputs=[gr.Number(value=1, visible=False)],
                                    outputs=[but3, butstop],
                                )
                                but4 = gr.Button(
                                    i18n("Generate feature index"), variant="primary"
                                )
                                with gr.Group():
                                    save_action = gr.Radio(
                                        show_label=False,
                                        choices=[
                                            i18n("Save final model"),
                                            i18n("Backup all"),
                                            i18n("Backup snapshot"),
                                        ],
                                        value=i18n("Save final model"),
                                        interactive=True,
                                    )
                                    but7 = gr.Button(i18n("Save model"), variant="primary")
                            # if_save_every_weights18.change(
                            #     fn=lambda if_save_every_weights: (
                            #         {
                            #             "visible": if_save_every_weights,
                            #             "__type__": "update",
                            #         }
                            #     ),
                            #     inputs=[if_save_every_weights18],
                            #     outputs=[save_epoch10],
                            # )
                            if_retrain_collapse20.change(
                                fn=lambda if_retrain_collapse20: (
                                    {
                                        "visible": if_retrain_collapse20,
                                        "__type__": "update",
                                    }
                                ),
                                inputs=[if_retrain_collapse20],
                                outputs=[collapse_threshold24],
                            )
                            if_stop_on_fit21.change(
                                fn=lambda if_stop_on_fit21: (
                                    {
                                        "visible": if_stop_on_fit21,
                                        "__type__": "update",
                                    }
                                ),
                                inputs=[if_stop_on_fit21],
                                outputs=[smoothness23],
                            )
                            if_stop_on_fit21.change(
                                fn=lambda if_stop_on_fit21: (
                                    {
                                        "visible": if_stop_on_fit21,
                                        "__type__": "update",
                                    }
                                ),
                                inputs=[if_stop_on_fit21],
                                outputs=[stop_on_fit_grace22],
                            )
                            sr2.change(
                                change_sr2,
                                [sr2, if_f0_3, version19],
                                [pretrained_G14, pretrained_D15],
                            )
                            version19.change(
                                change_version19,
                                [sr2, if_f0_3, version19],
                                [pretrained_G14, pretrained_D15, sr2],
                            )
                            if_f0_3.change(
                                fn=change_f0,
                                inputs=[if_f0_3, sr2, version19],
                                outputs=[f0method8, pretrained_G14, pretrained_D15],
                            )
                        with gr.Row():
                            info3 = gr.Textbox(
                                label=i18n("Log messages"),
                                lines=2,
                                max_lines=2,
                                value="",
                            )
                        but3.click(
                            click_train,
                            [
                                exp_dir1,
                                sr2,
                                if_f0_3,
                                spk_id5,
                                save_epoch10,
                                total_epoch11,
                                batch_size12,
                                if_save_latest13,
                                pretrained_G14,
                                pretrained_D15,
                                if_cache_gpu17,
                                if_save_every_weights18,
                                version19,
                                if_retrain_collapse20,
                                if_stop_on_fit21,
                                stop_on_fit_grace22,
                                smoothness23,
                                collapse_threshold24
                            ],
                            [info3, butstop, but3],
                            api_name="train_start",
                        )
                        but4.click(train_index, [exp_dir1, version19], info3)
                        but7.click(resources.save_model, [exp_dir1, save_action], info3)

            with gr.TabItem(i18n("UVR5")):  # UVR section
                with gr.Row():
                    with gr.Column():
                        model_select = gr.Radio(
                            label=i18n("Model architecture"),
                            choices=["VR", "MDX", "Demucs (Beta)"],
                            value="VR",
                            interactive=True,
                        )
                        dir_wav_input = gr.Textbox(
                            label=i18n("Path to input folder for audio files"),
                            value=os.path.join(now_dir, "assets", "audios", "uvr5", "input"),
                        )
                        wav_inputs = gr.File(
                            file_count="multiple",
                            label=i18n("Or select input audio files"),
                        )
                    with gr.Column():
                        model_choose = gr.Dropdown(
                            label=i18n("Model"), choices=uvr5_names
                        )
                        agg = gr.Slider(
                            minimum=0,
                            maximum=20,
                            step=1,
                            label="Extraction aggressiveness",
                            value=10,
                            interactive=True,
                        )
                        opt_vocal_root = gr.Textbox(
                            label=i18n("Path to output folder for vocals"),
                            value=os.path.join(now_dir, "assets", "audios", "uvr5", "output-vocals"),
                        )
                        opt_ins_root = gr.Textbox(
                            label=i18n("Path to output folder for accompaniment"),
                            value=os.path.join(now_dir, "assets", "audios", "uvr5", "output-others"),
                        )
                        format0 = gr.Radio(
                            label=i18n("Export file format"),
                            choices=["wav", "flac", "mp3", "m4a"],
                            value="flac",
                            interactive=True,
                        )
                        model_select.change(
                            fn=update_model_choices,
                            inputs=model_select,
                            outputs=model_choose,
                        )
                    with gr.Column():
                        with gr.Group():
                            vc_output4 = gr.Textbox(
                                label=i18n("Log messages"),
                                lines=16,
                                max_lines=16,
                            )
                            but2 = gr.Button(i18n("Convert"), variant="primary")
                            # wav_inputs.upload(fn=save_to_wav2_edited, inputs=[wav_inputs], outputs=[])
                        but2.click(
                            uvr,
                            [
                                model_choose,
                                dir_wav_input,
                                opt_vocal_root,
                                wav_inputs,
                                opt_ins_root,
                                agg,
                                format0,
                                model_select,
                            ],
                            [vc_output4],
                            api_name="uvr_convert",
                        )
            with gr.TabItem(i18n("TTS")):
                with gr.Column():
                    text_test = gr.Textbox(
                        label=i18n("Text"),
                        placeholder=i18n(
                            "Enter the text you want to convert to voice..."
                        ),
                        lines=6,
                    )

                with gr.Row():
                    with gr.Column():
                        tts_methods_voice = ["Edge-tts", "Google-tts"]
                        ttsmethod_test = gr.Dropdown(
                            tts_methods_voice,
                            value="Edge-tts",
                            label=i18n("TTS Engine"),
                            visible=True,
                        )
                        tts_test = gr.Dropdown(
                            tts.set_edge_voice,
                            label=i18n("TTS Voice"),
                            visible=True,
                        )
                        ttsmethod_test.change(
                            fn=tts.update_tts_methods_voice,
                            inputs=ttsmethod_test,
                            outputs=tts_test,
                        )

                    with gr.Column():
                        with gr.Group():
                            model_voice_path07 = gr.Dropdown(
                                label=i18n("Voice weight"),
                                choices=sorted(names),
                                value=i18n("Choose a voice weight file..."),
                            )
                            file_index2_07 = gr.Dropdown(
                                label=i18n("Voice feature index"),
                                choices=get_indexes(),
                                interactive=True,
                                allow_custom_value=True,
                            )
                            model_voice_path07.change(
                                fn=match_index,
                                inputs=[model_voice_path07],
                                outputs=[file_index2_07],
                            )
                            refresh_button_ = gr.Button(i18n("Refresh list"), variant="primary")
                            refresh_button_.click(
                                fn=change_choices2,
                                inputs=[],
                                outputs=[model_voice_path07, file_index2_07],
                            )
                with gr.Row():
                    original_ttsvoice = gr.Audio(label=i18n("TTS Output"))
                    ttsvoice = gr.Audio(label=i18n("RVC Output"))

                with gr.Row():
                    button_test = gr.Button(i18n("Inference"), variant="primary")

                button_test.click(
                    tts.use_tts,
                    inputs=[
                        text_test,
                        tts_test,
                        model_voice_path07,
                        file_index2_07,
                        # transpose_test,
                        vc_transform0,
                        f0method8,
                        index_rate1,
                        crepe_hop_length,
                        f0_autotune,
                        ttsmethod_test,
                    ],
                    outputs=[ttsvoice, original_ttsvoice],
                )

            with gr.TabItem(i18n("Resources")):
                resources.download_model()
                resources.download_backup()
                resources.download_dataset(trainset_dir4)
                resources.download_audio()
                resources.audio_downloader_separator()
            with gr.TabItem(i18n("Extra")):
                gr.Markdown(
                    value=i18n(
                        "This section contains some extra utilities that often may be in experimental phases"
                    )
                )
                with gr.TabItem(i18n("Merge Audios")):
                    mergeaudios.merge_audios()

                with gr.TabItem(i18n("Processing")):
                    processing.processing_()

            with gr.TabItem(i18n("Settings")):
                with gr.TabItem(i18n("Inference")):
                    noteshertz = gr.Checkbox(
                        label=i18n("Use note names for pitch input"),
                        info=i18n("Use note names instead of hertz value, eg. use [C5] instead of [523.25]Hz"),
                        value=rvc_globals.NotesOrHertz,
                        interactive=True,
                    )
                    noteshertz.change(
                        fn=lambda nhertz: rvc_globals.__setattr__("NotesOrHertz", nhertz),
                        inputs=[noteshertz],
                        outputs=[],
                    )
                    noteshertz.change(
                        fn=switch_pitch_controls,
                        inputs=[f0method0],
                        outputs=[
                            minpitch_slider,
                            minpitch_txtbox,
                            maxpitch_slider,
                            maxpitch_txtbox,
                        ],
                    )
                with gr.TabItem(i18n("Training")):
                    with gr.Row():
                        with gr.Column():
                            n_cpu = gr.Slider(
                                minimum=1,
                                maximum=config.n_cpu,
                                step=1,
                                label=i18n("CPU threads"),
                                value=config.n_cpu,
                                interactive=True,
                            )
                        with gr.Column():
                            with gr.Row():
                                gpu_ids = gr.Textbox(
                                    label=i18n(
                                        "GPU indexes (separated by \",\". eg. 0,1,2 for using GPU 0, 1, and 2)"
                                    ),
                                    value=gpus,
                                    interactive=True,
                                )
                                gr.Textbox(
                                    label=i18n("Available GPUs"),
                                    value=gpu_info,
                                    visible=F0GPUVisible,
                                )
                    n_cpu.change(
                        fn=lambda n_cpu: rvc_globals.__setattr__("CpuCores", n_cpu),
                        inputs=[n_cpu],
                        outputs=[],
                    )
                    gpu_ids.change(
                        fn=lambda gpu_ids: rvc_globals.__setattr__("GpuIds", gpu_ids),
                        inputs=[gpu_ids],
                        outputs=[],
                    )
                with gr.TabItem(i18n("About")):
                    gr.Markdown(
                        value=i18n("""
## RVC-Tundra

An enhanced RVC (Retrieval based Voice Conversion) GUI fork mostly for self-use.

This fork is equal to [Applio-RVC-Fork](https://github.com/IAHispano/Applio-RVC-Fork) and [Mangio-RVC-Fork](https://github.com/Mangio621/Mangio-RVC-Fork) on the code level.

This fork has a nature to be outdated. Please refer to the above repositories for the latest updates.

### License

MIT License (Non-Commercial)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to use,
copy, modify, merge, publish and/or distribute RVC-Tundra, subject to the following conditions:

1. The software and its derivatives may only be used for non-commercial
   purposes.

2. Any commercial use, sale, or distribution of the software or its derivatives
   is strictly prohibited.

3. The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS," WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The licenses for related libraries are as follows:

ContentVec
https://github.com/auspicious3000/contentvec/blob/main/LICENSE
MIT License

VITS
https://github.com/jaywalnut310/vits/blob/main/LICENSE
MIT License

HIFIGAN
https://github.com/jik876/hifi-gan/blob/master/LICENSE
MIT License

gradio
https://github.com/gradio-app/gradio/blob/main/LICENSE
Apache License 2.0

ffmpeg
https://github.com/FFmpeg/FFmpeg/blob/master/COPYING.LGPLv3
LGPLv3 License
MIT License

UVR5
https://github.com/Anjok07/ultimatevocalremovergui/blob/master/LICENSE
https://github.com/yang123qwe/vocal_separation_by_uvr5
MIT License

audio-slicer
https://github.com/openvpi/audio-slicer/blob/main/LICENSE
MIT License

PySimpleGUI
https://github.com/PySimpleGUI/PySimpleGUI/blob/master/license.txt
LGPLv3 License

Please note that under this license, the software and its derivatives can only be used for non-commercial purposes, and any commercial use, sale, or distribution is prohibited.
                        """)
                    )

        return app


def GradioRun(app):
    share_gradio_link = config.iscolab or config.paperspace
    concurrency_count = 511
    max_size = 1022

    if config.iscolab or config.paperspace:
        app.queue(concurrency_count=concurrency_count, max_size=max_size).launch(
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True,
            favicon_path="./assets/images/icon.png",
            share=share_gradio_link,
        )
    else:
        app.queue(concurrency_count=concurrency_count, max_size=max_size).launch(
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True,
            favicon_path="./assets/images/icon.png",
        )


if __name__ == "__main__":
    app = GradioSetup()
    GradioRun(app)
