import os
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio

#import data
SCREAMS_FILE = os.path.join('data','positive',"damm_0.wav")
NON_SCREAMS_FILE = os.path.join('data','negative','clnsp1.wav')

#Data loading function
def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav,axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    #Goes from 44100Hz to 16000Hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

wave = load_wav_16k_mono(SCREAMS_FILE)
nwave = load_wav_16k_mono(NON_SCREAMS_FILE)

plt.plot(wave)
plt.plot(nwave)
plt.show()

#Paths to positive and negative data
POS = os.path.join('data', 'positive')
NEG = os.path.join('data', 'negative')

#Tenseflow dataset
pos = tf.data.Dataset.list_files(POS+'/*.wav')
neg = tf.data.Dataset.list_files(NEG+'/*.wav')

#Labels and combine
positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
data = positives.concatenate(negatives)

#Calculate the average length of a scream/shout
lengths = []
for file in os.listdir(os.path.join('data','positive')):
    tensor_wave = load_wav_16k_mono(os.path.join('data','positive',file))
    lengths.append(len(tensor_wave))
    
#Preprocessing function
def preprocess (file_path, label):
    wav = load_wav_16k_mono(file_path)
    wav = wav[:120000]
    zero_padding = tf.zeros([120000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding,wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label

#Tenserflow Data Pipeline
data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(1)
data = data.prefetch(8)

#Split into Training and Testing Partitions
train = data.take(7)
test = data.skip(7).take(2)

#Build Sequential Model, Compile and View Summary
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(3741,257,1)))
model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

