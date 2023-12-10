from lib.infer.modules.vc.pipeline import Pipeline
from lib.infer.modules.vc.utils import load_hubert
from assets.configs.config import Config
import torch
import av
import numpy as np
from io import BytesIO

from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)

def get_net_g(cfg, device, version, if_f0, is_half, cpt):
  synthesizer_classes = {
    ("v1", 1): SynthesizerTrnMs256NSFsid,
    ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
    ("v2", 1): SynthesizerTrnMs768NSFsid,
    ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
  }
  SynthesizerClass = synthesizer_classes.get((version, if_f0), SynthesizerTrnMs256NSFsid)
  
  net_g = SynthesizerClass(*cpt['config'], is_half=is_half)

  del net_g.enc_q # have no idea what it does
  net_g.load_state_dict(cpt['weight'], strict=False) # load model weight
  net_g.eval().to(device)
  if cfg.is_half:
    net_g = net_g.half()
  else:
    net_g = net_g.float()

  return net_g

def load_cpt(filename):
  cpt = torch.load(filename, map_location='cpu')
  cpt['config'][-3] = cpt['weight']['emb_g.weight'].shape[0] # speakers count
  return cpt

def load_voice_weight(cfg, device, filename):
  cpt = load_cpt(filename)
  tgt_sr = cpt['config'][-1] # target sample rate
  if_f0 = cpt.get('f0', 1)
  version = cpt.get('version', 'v1')
  net_g = get_net_g(cfg, device, version, if_f0, cfg.is_half, cpt)

  return {
    'cpt': cpt,
    'tgt_sr': tgt_sr,
    'if_f0': if_f0,
    'version': version,
    'net_g': net_g
  }

def import_audio_from_file(file):
  with BytesIO() as out_bytes:
    input = av.open(file, 'rb')
    output = av.open(out_bytes, 'wb', format='f32le')

    out_stream = output.add_stream('pcm_f32le', channels=1)
    out_stream.sample_rate = 16000

    for frame in input.decode(audio=0):
      for frame_out in out_stream.encode(frame):
        output.mux(frame_out)
      
    output.close()
    input.close()
    
    return np.frombuffer(out_bytes.getvalue(), np.float32).flatten()

class SimplePipeline:
  def __init__(self, voice_weight_file):
    cfg = Config()
    device = torch.device(
      "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available()
        else "cpu")
    )
    self.hubert_model = load_hubert(cfg)
    self.voice_weight = load_voice_weight(cfg, device, voice_weight_file)
    self.pipeline = Pipeline(self.voice_weight['tgt_sr'], cfg)

  def run(self, input):
    return self.pipeline.pipeline(
      self.hubert_model, # model
      self.voice_weight['net_g'],
      0, # sid
      input,
      '', # input_audio_path, don't make sense
      [0, 0, 0], # times
      12, # f0_up_key
      'rmvpe+', # f0_method
      ',', # file_index
      0.75, # index_rate,
      self.voice_weight['if_f0'],
      3, # filter_radius
      self.voice_weight['tgt_sr'],
      0, # resample_sr
      0.25, # rms_mix_rate
      self.voice_weight['version'],
      0.33, # protect
      120, # crepe_hop_length
      False, # f0_autotune
      None, # f0_file
      f0_min=50,
      f0_max=1100
    )

  def get_target_sample_rate(self):
    return self.voice_weight['tgt_sr']
