#!/usr/bin/env python3
import wave
import numpy as np

wavefile = wave.open("LJ037-0171.wav", "r")
length = wavefile.getnframes()
print(f"{length} frames")
wavedata = np.frombuffer(wavefile.readframes(length), np.int16)
print(len(wavedata))
print(wavedata[0:10])

