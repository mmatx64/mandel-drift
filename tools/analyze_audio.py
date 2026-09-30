"""Measure a decoded stereo 16-bit PCM WAV using NumPy."""
import json
import sys
import wave
import numpy as np

with wave.open(sys.argv[1], 'rb') as stream:
    rate = stream.getframerate()
    channels = stream.getnchannels()
    assert stream.getsampwidth() == 2
    assert channels == 2
    audio = np.frombuffer(stream.readframes(stream.getnframes()), '<i2').astype(float).reshape(-1, channels) / 32768
mono = audio.mean(axis=1)
size = 8192
hop = 2048
frames = np.lib.stride_tricks.sliding_window_view(audio, size, axis=0)[::hop]
power = abs(np.fft.rfft(frames * np.hanning(size), axis=-1)) ** 2
freq = np.fft.rfftfreq(size, 1 / rate)
spectrum = power.mean(axis=(0, 1))
bands = [(0, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 12000)]
peaks = np.flatnonzero((spectrum[1:-1] > spectrum[:-2]) & (spectrum[1:-1] > spectrum[2:])) + 1
peaks = sorted(peaks, key=lambda p: spectrum[p], reverse=True)[:24]
def note(hz):
    midi = 69 + 12 * np.log2(hz / 440)
    names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    rounded = int(round(midi))
    return f'{names[rounded % 12]}{rounded // 12 - 1} {100*(midi-rounded):+.0f}c'
window = int(rate * .1)
envelope = np.sqrt(np.mean(audio[:len(audio)//window*window].reshape(-1, window, channels)**2, axis=(1, 2)))
result = {
    'duration_seconds': len(audio) / rate,
    'rms_dbfs': float(20*np.log10(np.sqrt(np.mean(audio**2)))),
    'peak_dbfs': float(20*np.log10(abs(audio).max())),
    'stereo_correlation': float(np.corrcoef(audio.T)[0,1]),
    'side_to_mid_rms': float(np.sqrt(np.mean(((audio[:,0]-audio[:,1])/2)**2)) / np.sqrt(np.mean(mono**2))),
    'spectral_centroid_hz': float(np.sum(freq*spectrum)/spectrum.sum()),
    'energy_bands': {f'{lo}-{hi} Hz': float(spectrum[(freq>=lo)&(freq<hi)].sum()/spectrum.sum()) for lo,hi in bands},
    'dominant_peaks': [{'hz': round(float(freq[p]),2), 'note': note(freq[p])} for p in peaks if freq[p]>20],
    'rms_envelope_100ms': envelope.round(5).tolist(),
}
print(json.dumps(result, indent=2))
