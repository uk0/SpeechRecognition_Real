import time

from huggingsound import SpeechRecognitionModel
# wbbbbb/wav2vec2-large-chinese-zh-cn 0.7
model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")
audio_paths = ["zh.wav"]
t1 = time.time()
transcriptions = model.transcribe(audio_paths)
model.processor()
t2 = time.time()
print("jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")
print(t2 - t1)
print(transcriptions[0]["transcription"])


# wbbbbb/wav2vec2-large-chinese-zh-cn 0.7
model = SpeechRecognitionModel("wbbbbb/wav2vec2-large-chinese-zh-cn")
audio_paths = ["zh.wav"]
t1 = time.time()
transcriptions = model.transcribe(audio_paths)
t2 = time.time()
print("wbbbbb/wav2vec2-large-chinese-zh-cn")
print(t2 - t1)
print(transcriptions[0]["transcription"])


# wbbbbb/wav2vec2-large-chinese-zh-cn 0.7
model = SpeechRecognitionModel("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")
audio_paths = ["zh.wav"]
t1 = time.time()
transcriptions = model.transcribe(audio_paths)
t2 = time.time()
print("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")
print(t2 - t1)
print(transcriptions[0]["transcription"])



# transcriptions format (a list of dicts, one for each audio file):
# [
#  {
#   "transcription": "extraordinary claims require extraordinary evidence",
#   "start_timestamps": [100, 120, 140, 180, ...],
#   "end_timestamps": [120, 140, 180, 200, ...],
#   "probabilities": [0.95, 0.88, 0.9, 0.97, ...]
# },
# ...]
#
# as you can see, not only the transcription is returned but also the timestamps (in milliseconds)
# and probabilities of each character of the transcription.
