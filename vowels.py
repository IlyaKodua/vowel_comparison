import numpy as np
import librosa
import matplotlib.pyplot as plt

def ploting(signals,sr):
    keys = list(signals.keys())

    for key in keys:
        N = len(signals[key])
        plt.figure(key)
        plt.plot(np.arange(N)/N*sr, np.abs(signals[key]), 'r')
    
    plt.show()


def corelation(x,y):
    return np.abs(np.sum(x*np.conj(y))) / np.sqrt(np.sum(np.abs(x)**2) * np.sum(np.abs(y)**2))

def get_vowel_dict(x):

    vowels = ['i', 'u', 'a', 'o', 'e']

    begin = [51600, 164500, 26510, 344600, 413800]
    end = [118200, 232300, 312800, 386700, 464000]

    signals = dict()

    min_len = end[0] - begin[0]
    for i in range(1,len(vowels)):
        min_len  = np.min([min_len, end[i] - begin[i]])
        print(min_len)
    
    for i in range(len(end)):
        signals[vowels[i]] = np.fft.fft(x[end[i] - min_len : end[i] + 1])[0:min_len//2]
        signals[vowels[i]][1:len(signals[vowels[i]])] *= 2


    return signals
    


x, sr = librosa.load('vowels.wav')




signals = get_vowel_dict(x)



keys = list(signals.keys())

for key_i in keys:
    for key_j in keys:
        cor = corelation(signals[key_i], signals[key_j])
        print('Correlation between ', key_i, ' and ', key_j, ': ', "{:10.4f}".format(cor))

ploting(signals,sr)
