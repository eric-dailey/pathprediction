import tensorflow as tf
gf = tf.GraphDef()
gf.ParseFromString(open('./model.pb', 'rb').read())
for n in gf.node:
	print(n.name)

#gf.summary()
