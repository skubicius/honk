import os
from .base_extractor import BaseAudioExtractor
from pocketsphinx import get_model_path, AudioFile, Pocketsphinx

import nemo.collections.asr as nemo_asr

# This line will download pre-trained QuartzNet15x5 model from NVIDIA's NGC cloud and instantiate it for you
quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")

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
        print("Done printing segments")
        audio = AudioFile(**self.kws_config)

        for phrase in audio:
            result = phrase.segments(detailed=True)

            # TODO:: confirm that when multiple keywords are detected, every detection is valid
            if len(result) == 1:
                start_time = result[0][2] * 10
                end_time = result[0][3] * 10
                # print('%4sms ~ %4sms' % (start_time, end_time))

                kws_results.append((start_time, end_time))

        return kws_results
