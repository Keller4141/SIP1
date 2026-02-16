# Assignment 2 – Signal and Image Processing
# Part 2.1, 2.2, 2.3

# ##2.1



import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt




soundbyte, samplerate = sf.read("laugh2.wav")

print("Shape of sound array:", soundbyte.shape)
print("Sample rate:", samplerate)




left = soundbyte[:, 0]
right = soundbyte[:, 1]

print("Left shape:", left.shape)
print("Right shape:", right.shape)





t = np.arange(soundbyte.shape[0]) / samplerate  

plt.figure(figsize=(14, 4))
plt.plot(t, left, label="Left channel")
plt.plot(t, right, label="Right channel", alpha=0.8)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("laugh2.wav: Left and Right channels")
plt.legend()
plt.tight_layout()
plt.show()


# ##2.2




sound_path = "xylo.wav"

clap_path = "Big-Living Room-Wooden-Floor--Hand-Clap-Sample--(Rode-NT2000-cardioid-Royer-R121-fig8).wav"

splash_path = "Splash 1.wav"




sound, sr_sound = sf.read(sound_path)
clap, sr_clap = sf.read(clap_path)
splash, sr_splash = sf.read(splash_path)

print("sound shape:", sound.shape, "sr:", sr_sound)
print("clap shape:", clap.shape, "sr:", sr_clap)
print("splash shape:", splash.shape, "sr:", sr_splash)



from scipy.signal import resample_poly
if sr_clap != sr_sound:
    clap = resample_poly(clap, up=sr_sound, down=sr_clap, axis=0)





from scipy.signal import convolve

def convolve_stereo(sound_stereo, impulse_stereo):
    left = convolve(sound_stereo[:, 0], impulse_stereo[:, 0], mode="full")
    right = convolve(sound_stereo[:, 1], impulse_stereo[:, 1], mode="full")
    return np.stack([left, right], axis=1)

y_clap = convolve_stereo(sound, clap)
y_splash = convolve_stereo(sound, splash)      

print("y_clap shape:", y_clap.shape)
print("y_splash shape:", y_splash.shape)




from IPython.display import Audio, display

display(Audio(sound.T, rate=sr_sound))
display(Audio(y_clap.T, rate=sr_sound))
display(Audio(y_splash.T, rate=sr_sound))




def plot_left(x, sr, title):
    left = x[:, 0] if x.ndim == 2 else x
    t = np.arange(len(left)) / sr
    plt.figure(figsize=(14, 4))
    plt.plot(t, left)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_left(sound, sr_sound, "Original: xylo.wav (left channel)")
plot_left(y_clap, sr_sound, "Response: xylo convolved with clap impulse")
plot_left(y_splash, sr_sound, "Response: xylo convolved with splash impulse")


# ##2.3



print("Original length:", sound.shape[0])
print("Clap length (after resample):", clap.shape[0])
print("Splash length:", splash.shape[0])

print("y_clap length:", y_clap.shape[0])
print("y_splash length:", y_splash.shape[0])

