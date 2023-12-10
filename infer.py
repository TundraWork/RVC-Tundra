from dotenv import load_dotenv
load_dotenv()

from src.infer.simple_pipeline import SimplePipeline, import_audio_from_file
import soundfile

input = import_audio_from_file('assets/audios/pofsOMOh.mp3')
pipeline = SimplePipeline('logs/weights/XiaorouChannel-speech-pitch-v3-760.pth')
sample_rate = pipeline.get_target_sample_rate()
output = pipeline.run(input)

soundfile.write('assets/audios/audio-outputs/output.flac', output, sample_rate)
