import sounddevice as sd
import numpy as np

def record_audio(duration, fs, channels):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype='float64')
    sd.wait()
    print("Recording stopped.")
    return recording

def save_audio(filename, data, fs):
    sd.write(filename, data, fs)

if __name__ == "__main__":
    duration = 5  # Recording duration in seconds
    fs = 44100    # Sampling frequency
    channels = 1  # Number of channels (1 for mono, 2 for stereo)

    recording = record_audio(duration, fs, channels)
    save_audio("recorded_audio.wav", recording, fs)
