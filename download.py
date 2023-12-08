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
import lib.globals.globals as rvc_globals

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

sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


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

