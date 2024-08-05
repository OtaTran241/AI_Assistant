from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from keras.models import load_model
import tensorflow as tf
from os.path import join, dirname, abspath


model = Sequential()
model.add(Conv2D(1, (3,3), activation='relu', input_shape=(1491, 257,1)))
model.add(Conv2D(1, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(12, activation='softmax'))

model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

current_dir = dirname(abspath(__file__))

model = load_model(join(current_dir, 'Models/VoiceClassification.h5'))

def get_age(wav):
    wav = preprocess(wav)
    wav = tf.expand_dims(wav, axis=0)
    pre = model.predict([wav])
    age = 0
    accuracy = 0
    for i, v1 in enumerate(pre):
        for j, v in enumerate(v1):
            if v > 0.5:
                p = round(float(v), 2)*100
                age = j
                accuracy = p
                return transfer(age), accuracy
            
def preprocess(wav):
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

def transfer(num):
    if num <= 11:
        output_number = num + 16
    if num == 12:
        output_number = 28
    return output_number
