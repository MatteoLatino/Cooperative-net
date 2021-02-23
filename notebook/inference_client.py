import sys
import os
os.close(sys.stderr.fileno())
from MyMQTT import MyMQTT
import tensorflow.lite as tflite
import tensorflow as tf
import numpy as np
import datetime
import argparse
import time
import json
import base64  

class Inference_client():
	def __init__(self, clientID, model_path):
		self.clientID = clientID
		self.myMqttClient = MyMQTT(self.clientID, "mqtt.eclipseprojects.io", 1883, self) 
		
		#initialize the interpreter
		self.interpreter = tflite.Interpreter(model_path=model_path)
		self.interpreter.allocate_tensors()
		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()

	
	def run(self):
		print ("running %s" % (self.clientID))
		self.myMqttClient.start()

	def end(self):
		print ("ending %s" % (self.clientID))
		self.myMqttClient.stop ()
	
	def notify(self, topic, msg): #msg = preprocessed audio
		data = json.loads(msg)
		t = data["e"][0]["t"] 
		audio_str = data["e"][0]["v"]
		audio_p = base64.b64decode(audio_str) 
		audio_p = np.frombuffer(audio_p, dtype = np.float32)
		audio_p = tf.constant(audio_p, dtype=tf.float32)
		audio_p = tf.reshape(audio_p, [1,49,10,1])

		#compute the inference
		self.interpreter.set_tensor(self.input_details[0]['index'], audio_p) 
		self.interpreter.invoke()
		my_output = self.interpreter.get_tensor(self.output_details[0]['index'])
		score = base64.b64encode(my_output).decode()
		score_e = {"n":"score",  "u":"/", "t":t, "v":score}
			
		now = datetime.datetime.now()
		timestamp = int(now.timestamp())
		data = {"bn": "192.168.0.105",
			"bt": timestamp,
			"e": [score_e]} 
		#send the scores
		data = json.dumps(data)
		self.myMqttClient.myPublish('269317/HW3/inference', data)


parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str, help='Path of the tflite model', required=True)
args = parser.parse_args()
model_path = args.model

if __name__ == "__main__":
	cl = Inference_client(f"client{model_path}", model_path)
	cl.run()
	cl.myMqttClient.mySubscribe('269317/HW3_2/audio')
	while True:
		time.sleep(0)
			
	#unsubscribe and end loop
	cl.end()







