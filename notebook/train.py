import tensorflow.lite as tflite
import tensorflow.keras as keras
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import zlib
import os

def save_model(model, model_dir, model_lite_dir, spec = [None, 49, 10, 1]):
	run_model = tf.function(lambda x: model(x))
	concrete_func = run_model.get_concrete_function(tf.TensorSpec(spec, tf.float32))

	model.save(model_dir, signatures=concrete_func)
	converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)
	tflite_model = converter.convert()

	with open(model_lite_dir, 'wb') as fp:
		fp.write(tflite_model)

	print(f"Lite dimension: {os.path.getsize(model_lite_dir)/1024}")
	return 
  
def read_file(path):
	files = []
	for file in open(path).readlines():
		files.append(file.replace("\n", "").encode("UTF-8"))
	files = tf.convert_to_tensor(files, dtype=tf.string)
	return files

def bootstrap(train_files):
	indexes = np.random.randint(low=0, high=len(train_files), size=int(1.5*len(train_files)) )
	new_train_files = np.array(train_files)[indexes]
	return  tf.convert_to_tensor(new_train_files, dtype=tf.string)
		
class SignalGenerator:
	def __init__(self, labels, sampling_rate, num_coef):

		self.preprocess = self.preprocess_with_mfcc
		self.sampling_rate = sampling_rate
		self.num_coef = num_coef
		self.labels = labels
		NUM_SPECTROGRAM_BINS = 321 #spectrogram.shape[-1]
		NUM_MEL_BINS = 40
		LOWER_FREQUENCY=20
		UPPER_FREQUENCY=4000
		self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(NUM_MEL_BINS,
								NUM_SPECTROGRAM_BINS,
								sampling_rate,
								LOWER_FREQUENCY,
								UPPER_FREQUENCY)	
	def read(self, file_path):
		parts = tf.strings.split(file_path, os.path.sep)
		labels = parts[-2]
		label_id = tf.argmax(labels == self.labels)
		audio_binary = tf.io.read_file(file_path)
		audio, _ = tf.audio.decode_wav(audio_binary)
		audio = tf.squeeze(audio, axis = 1)
		return audio, label_id
    
	def pad(self, audio):
		zero_padding = tf.zeros([self.sampling_rate]-tf.shape(audio), dtype = tf.float32)
		audio = tf.concat([audio, zero_padding], 0)
		audio.set_shape([self.sampling_rate])
		return audio
    
	def get_spectrogram(self, audio):
		frame_length = 16 * 40
		frame_step = 16 * 20
		stft = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step,fft_length=frame_length)
		spectrogram = tf.abs(stft)
		return spectrogram
    
	def get_mfcc(self, spectrogram):
		mel_spectrogram = tf.tensordot(spectrogram, self.linear_to_mel_weight_matrix,1)
		log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
		mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :self.num_coef]
		return mfccs

	def preprocess_with_mfcc(self, file_path):
		audio, labels = self.read(file_path)
		audio = self.pad(audio)
		spectrogram = self.get_spectrogram(audio)
		mfccs = self.get_mfcc(spectrogram)
		mfccs = tf.expand_dims(mfccs, -1)
		return mfccs, labels
    
	def make_dataset(self, files, train):
		ds = tf.data.Dataset.from_tensor_slices(files)
		ds = ds.map(self.preprocess, num_parallel_calls = 4)
		ds = ds.batch(32)
		ds = ds.cache()
		if train is True:
			ds = ds.shuffle(200, reshuffle_each_iteration = True)
		ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
		return ds


parser = argparse.ArgumentParser()
parser.add_argument('--version', type = str, help='Tha available versions are: 1...4', required=True)
args = parser.parse_args()
version = args.version
version = int(version)

tf.random.set_seed(version-1)
np.random.seed(version-1)
  
zip_path = tf.keras.utils.get_file(origin = "http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip", 
                                   fname = "mini_speech_commands.zip", 
                                   extract = True, 
                                   cache_dir = ".",
                                   cache_subdir  = 'data')
data_dir = os.path.join(".", "data", "mini_speech_commands")
filenames = tf.io.gfile.glob(str(data_dir) + "/*/*")
filenames = tf.random.shuffle(filenames)
n = len(filenames)
train_files = read_file("kws_train_split.txt")
val_files = read_file("kws_val_split.txt")

test_files = read_file("kws_test_split.txt")

LABELS =  ['down', 'stop', 'right', 'left', 'up', 'yes', 'no', 'go']


generator = SignalGenerator(LABELS, 16000, 10)
new_train_files = bootstrap(train_files)  
train_ds = generator.make_dataset(new_train_files, True)
val_ds = generator.make_dataset(val_files, False)
test_ds = generator.make_dataset(test_files, False)

strides = [2, 1]
input_shape = [49, 10, 1]

model = keras.Sequential([
    keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=strides, use_bias=False, input_shape=(input_shape)), 
    keras.layers.BatchNormalization(momentum=0.1),
    keras.layers.ReLU(),
    keras.layers.GaussianNoise(0.1),
    keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], use_bias=False),
    keras.layers.BatchNormalization(momentum=0.1),
    keras.layers.ReLU(),
    keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], use_bias=False),
    keras.layers.BatchNormalization(momentum=0.1),
    keras.layers.ReLU(),
    keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
    keras.layers.BatchNormalization(momentum=0.1),
    keras.layers.ReLU(),
    keras.layers.Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
    keras.layers.BatchNormalization(momentum=0.1),
    keras.layers.ReLU(),
    keras.layers.Conv2D(filters=64, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
    keras.layers.BatchNormalization(momentum=0.1),
    keras.layers.ReLU(),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(units=8)
])

model_dir = str(version)
model_lite_dir = f'{version}.tflite'

model.compile(optimizer=tf.optimizers.Adam(), 
              loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics = keras.metrics.SparseCategoricalAccuracy() )

save_mod = tf.keras.callbacks.ModelCheckpoint('./best_model', monitor='val_loss', 
                            save_best_only=True, save_weights_only=False,
                            mode='auto', save_freq='epoch')
reducing_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_loss", factor=0.98, patience = 4, verbose = 1)

model.fit(train_ds, epochs = 20, verbose = 2, validation_data= val_ds, callbacks=[save_mod, reducing_lr])
test_loss, test_acc = model.evaluate(test_ds)
model = keras.models.load_model('best_model')
save_model(model, model_dir, model_lite_dir, spec = [None, 49, 10, 1])
test_loss, test_acc = model.evaluate(test_ds)


