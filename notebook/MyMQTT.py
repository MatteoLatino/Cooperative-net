import paho.mqtt.client as PahoMQTT

class MyMQTT:
	def __init__(self, clientID, broker, port, notifier):
		self.broker = broker
		self.port = port
		self.notifier = notifier
		self.clientID = clientID

		self._topic = ""
		self._isSubscriber = False

		# create an instance of paho.mqtt.client
		self._paho_mqtt = PahoMQTT.Client(clientID, False) 

		# register the callback
		self._paho_mqtt.on_connect = self.myOnConnect
		self._paho_mqtt.on_message = self.myOnMessageReceived


	def myOnConnect (self, paho_mqtt, userdata, flags, rc):
		print ("Connected to %s with result code: %d" % (self.broker, rc))

	def myOnMessageReceived (self, paho_mqtt , userdata, msg):
		self.notifier.notify (msg.topic, msg.payload)


	def myPublish (self, topic, msg):
		self._paho_mqtt.publish(topic, msg, 0)

	def mySubscribe (self, topic):
		print ("subscribing to %s" % (topic))
		self._paho_mqtt.subscribe(topic, 0)

		# just to remember that it works also as a subscriber
		self._isSubscriber = True
		self._topic = topic

	def start(self):
		self._paho_mqtt.connect(self.broker , self.port)
		self._paho_mqtt.loop_start()

	def stop (self):
		if (self._isSubscriber):
			# remember to unsuscribe if it is working also as subscriber 
			self._paho_mqtt.unsubscribe(self._topic)

		self._paho_mqtt.loop_stop()
		self._paho_mqtt.disconnect()



