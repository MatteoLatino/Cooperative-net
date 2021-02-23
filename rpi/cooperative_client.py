import sys
import os
os.close(sys.stderr.fileno())
from MyMQTT import MyMQTT
import tensorflow.lite as tflite
import tensorflow as tf
import numpy as np
import datetime
import time
import json
import base64            
                  
zip_path = tf.keras.utils.get_file(origin = "http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip", 
                                   fname = "mini_speech_commands.zip", 
                                   extract = True, 
                                   cache_dir = ".",
                                   cache_subdir  = 'data')
                                   
def preprocess(audio_path):
	frame_length = 640  #16*40
	frame_step = 320    #16*20
	num_mel_bins = 40 
	lower_frequency = 20 #Hz
	upper_frequency = 4000 #Hz
	Num_coeff=10
	sampling_rate = 16000
	num_spectrogram_bins = 321 # spectrogram.shape[-1]
	
	audio_binary = tf.io.read_file(audio_path)
	audio, _ = tf.audio.decode_wav(audio_binary)
	audio = tf.squeeze(audio, axis=1)
	zero_padding = tf.zeros([sampling_rate] - tf.shape(audio), dtype=tf.float32)
	audio = tf.concat([audio, zero_padding], 0)
	audio.set_shape([sampling_rate])


	stft = tf.signal.stft(audio, frame_length = frame_length, 
			  frame_step = frame_step, 
			  fft_length = frame_length)
	spectrogram = tf.abs(stft)
	
	linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, 
			                                                num_spectrogram_bins, 
			                                                sampling_rate, 
			                                                lower_frequency, 
			                                                upper_frequency)

	mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
	mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
	log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
	mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :Num_coeff]
	mfcc = tf.expand_dims(mfcc, -1)
	mfcc = tf.expand_dims(mfcc, 0)
	return mfcc

		

class Cooperative_client():
	def __init__(self, clientID, N_labels, tot):
		self.clientID = clientID
		self.Inferences = {}
		for i in range(tot):
			self.Inferences[i] = np.zeros(N_labels)
		self.myMqttClient = MyMQTT(self.clientID, "mqtt.eclipseprojects.io", 1883, self) 
	
	def run(self):
		print ("running %s" % (self.clientID))
		self.myMqttClient.start()
	
	def end(self):
		print ("ending %s" % (self.clientID))
		self.myMqttClient.stop()
		
	def notify(self, topic, msg): #msg = inference
		data = json.loads(msg)
		score = base64.b64decode(data["e"][0]["v"])
		score = np.frombuffer(score, dtype = np.float32)
		self.Inferences[data["e"][0]["t"]] += score
		
	def get_inference(self):
		return self.Inferences


	
if __name__ == "__main__":
	LABELS =  ['down', 'stop', 'right', 'left', 'up', 'yes', 'no', 'go']
	test_path = "kws_test_split.txt"
	files = []
	for f in open(test_path).readlines():
		files.append(f.replace("\n", "").encode("UTF-8"))
	test_files = tf.convert_to_tensor(files, dtype=tf.string)
	tot = len(test_files)
	
	Ensemble_net = Cooperative_client("cooperative_net", len(LABELS), tot)
	Ensemble_net.run()
	Ensemble_net.myMqttClient.mySubscribe('269317/HW3/inference')
	
	now = datetime.datetime.now()
	timestamp = int(now.timestamp())
	
	t = 0
	labels_id = []
	for audio_path in test_files:
		#get the true label
		parts = tf.strings.split(audio_path, os.path.sep)
		label = parts[-2]
		label_id = tf.argmax(label == LABELS)
		labels_id.append(label_id)  
		
		#pre process the audio
		pre_processed_audio = preprocess(audio_path)
		pre_processed_audio = base64.b64encode(pre_processed_audio).decode()
		audio_e = {"n":"Audio", "u":"/", "t":t, "v":pre_processed_audio} 
		data = {"bn": "192.168.0.108",
			"bt": timestamp,
			"e": [audio_e]} 
		data = json.dumps(data)
		Ensemble_net.myMqttClient.myPublish('269317/HW3_2/audio',data) 

		t += 1
		time.sleep(0.01)
			
	time.sleep(2)
	correct = 0
	inferences = Ensemble_net.get_inference()
	for i in range(len(inferences)):
		lbl = np.argmax(inferences[i])
		if lbl == labels_id[i]:
			correct += 1
		
	print(f"Final accuracy:{correct/tot}")
	Ensemble_net.end()

