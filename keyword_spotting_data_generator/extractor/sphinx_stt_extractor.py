import os
from .base_extractor import BaseAudioExtractor
from pocketsphinx import get_model_path, AudioFile, Pocketsphinx

import nemo.collections.asr as nemo_asr
from scipy.io import wavfile
import tempfile

# This line will download pre-trained QuartzNet15x5 model from NVIDIA's NGC cloud and instantiate it for you
quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")

def transcribe(path, start, end):
  if start >= end:
    return ""

  samplerate, data = wavfile.read(path)
  with tempfile.TemporaryDirectory() as tmp:
    tmp_path = os.path.join(tmp, 'audio.wav')
    # pad = 0.2 * 16000
    # start = int(start - pad)
    # end = int(end + pad)
    wavfile.write(tmp_path, samplerate, data[start:end])
    text = quartznet.transcribe(paths2audio_files=[tmp_path])[0]
    print(data.shape)
    print(start, end, samplerate)
    print('Transcribed as', text)
    return text

def good_start(text, keyword):
  return text.strip().lower().startswith(keyword)

def good_end(text, keyword):
  return text.strip().lower().endswith(keyword)

class SphinxSTTExtractor(BaseAudioExtractor):
    def __init__(self, keyword, threshold=1e-20):
        super().__init__(keyword, threshold)
        print(self.__class__.__name__, " is initialized with threshold ", threshold)

        self.kws_config = {
            'verbose': False,
            'keyphrase': self.keyword,
            'kws_threshold':threshold,
            'lm': False,
        }


    def extract_keywords(self, file_name, sample_rate=16000, window_ms=1000, hop_ms=500):

        kws_results = []

        files = [file_name]
        for fname, transcription in zip(files, quartznet.transcribe(paths2audio_files=files)):
          print(f"[NeMo] Audio in {fname} was recognized as: {transcription}")

        self.kws_config['audio_file'] = file_name

        audio = AudioFile(audio_file=file_name)
        print(f"Printing all audio segments in {file_name}")
        for phrase in audio:
          for s in phrase.seg():
            print(s.start_frame, s.end_frame, s.word)
            print(transcribe(file_name, s.start_frame * 160, s.end_frame * 160))
        print("Done printing segments")
        audio = AudioFile(**self.kws_config)

        for phrase in audio:
            result = phrase.segments(detailed=True)

            # TODO:: confirm that when multiple keywords are detected, every detection is valid
            if len(result) == 1:
                start_time = result[0][2] * 10
                end_time = result[0][3] * 10
                # print('%4sms ~ %4sms' % (start_time, end_time))
                text = transcribe(file_name, start_time * 16, end_time * 16)
                if self.keyword not in text.lower():
                  continue

                print("Pruning")
                while not good_start(text, self.keyword) and start_time < end_time:
                  start_time += 100
                  text = transcribe(file_name, start_time * 16, end_time * 16)

                while not good_end(text, self.keyword) and start_time < end_time:
                  end_time -= 100
                  text = transcribe(file_name, start_time * 16, end_time * 16)

                if text == self.keyword:
                  print("MATCH", file_name)
                  kws_results.append((start_time, end_time))

        return kws_results
