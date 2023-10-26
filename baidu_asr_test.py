import time

from paddlespeech.cli.asr.infer import ASRExecutor
asr = ASRExecutor()
t1 = time.time()


result = asr(audio_file="zh.wav")
t2 = time.time()
print(t2 - t1)
print(result)