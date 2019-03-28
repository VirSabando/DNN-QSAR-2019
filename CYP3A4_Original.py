#estas lineas son para ignorar unos warnings muy molestos
#1) para ignorar warnings respecto a CPU/GPU optimization de tensowflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#2) para ignorar el dependency bug de h5py
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Lambda 0.0001 y minibatch 400

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import pickle as pkl
from CYP3A4 import train_Activos_data, train_Inactivos_data, train, test, ev

from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import dropout
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

from sys import argv

# Semilla
semilla = argv[1]
tf.set_random_seed(int(semilla))

ruta = "/home/viri/Modelos_Finales/CYP3A4/Originales/he_10_o/"+str(semilla)+"/he.ckpt"
ruta_pickles = "/home/viri/Modelos_Finales/CYP3A4/Originales/he_10_o/Pickles/"

# Tamanios de las capas de la red
n_inputs = 1032	
n_hidden1 = 20
n_hidden2 = 10
n_hidden3 = 5
n_outputs = 2

# Placeholders para los datasets
X = tf.placeholder(tf.float32, shape = (None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name = "y")
is_training =tf.placeholder(tf.bool, shape = (), name = 'is_training')
phase =tf.placeholder(tf.bool, shape = (), name = 'phase')

# Definicion de diccionario de parametros para batch normalization
bn_params = {
	'is_training': is_training,
	'decay': 0.9,
	'updates_collections': tf.GraphKeys.UPDATE_OPS
}

# Learning rate
learning_rate = 0.00001

# Numero de epochs 
n_epochs = np.iinfo(np.int32).max

# Tamanio de mini batch
mini_batch_size = 200

# Coeficiente lambda para Ln regularization
lambda_ln = 0.001

# Coeficiente de Dropout: keep probability
keep_prob_h1 = 0.65
keep_prob_h2 = 0.75
keep_prob_h3 = 0.85


# Regularizador: L2
Ln_reg = tf.contrib.layers.l2_regularizer(lambda_ln)

# Definicion de la arquitectura de red a: batch norm
with tf.variable_scope("dnn"):
	with tf.contrib.framework.arg_scope([fully_connected], normalizer_fn = batch_norm, normalizer_params = bn_params, weights_regularizer = Ln_reg, weights_initializer = tf.keras.initializers.he_normal(seed=None)):
		X_drop = dropout(X, keep_prob_h1, is_training = phase)
		hidden1 = fully_connected(inputs=X_drop, num_outputs=n_hidden1, scope='hidden1')
		hidden1_drop = dropout(hidden1, keep_prob_h1, is_training = phase)
		hidden2 = fully_connected(inputs=hidden1_drop, num_outputs=n_hidden2, scope='hidden2')
		hidden2_drop = dropout(hidden2, keep_prob_h2, is_training = phase)
		hidden3 = fully_connected(inputs=hidden2_drop, num_outputs=n_hidden3, scope='hidden3')
		hidden3_drop = dropout(hidden3, keep_prob_h3, is_training = phase)
		logits = fully_connected(inputs=hidden3_drop, num_outputs=n_outputs, scope='outputs')

# Creacion de funcion de costo: Cross-entropy con la funcion provista
# por TF: sparse_softmax_cross_entropy_with_logits(...)
with tf.name_scope("cost"):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)
	reg_cost = reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	base_cost = tf.reduce_mean(xentropy, name = "base_cost")
	cost = tf.add_n([base_cost] + reg_cost, name = "cost")
	# cost = base_cost

# Definicion de un optimizador: Adam Optimizer, de TF
with tf.name_scope("train"):
	optimizer = tf.train.AdamOptimizer(learning_rate)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		training_op = optimizer.minimize(cost)

# Definicion de criterio de buena clasificacion: fn Softmax
with tf.name_scope("softmax_predict"):
	softmax = tf.nn.softmax(logits=logits, axis = 1, name="softmax")

def predicciones_correctas(p,t):
	pred_activos = p > 0.5
	correctas = ~np.logical_xor(pred_activos, t)
	return correctas

def sensibilidad(p,t):
	pred_activos = p > 0.5
	TP = np.count_nonzero(pred_activos & t)
	FN = np.count_nonzero(~pred_activos & t)
	if((TP + FN) > 0):
		sensitivity = TP / (TP + FN)
	else:
		sensitivity = 1
	return sensitivity

def especificidad(p,t):
	pred_activos = p > 0.5
	TN = np.count_nonzero(~pred_activos & ~t)
	FP = np.count_nonzero(pred_activos & ~t)
	if((TN + FP) > 0):
		specificity = TN / (TN + FP)
	else:
		specificity = 1
	return specificity

# Creacion de nodos para inicializar todas las variables (globales y locales)
init = tf.global_variables_initializer()
# Creacion de nodo checkpointer (a disco)
saver = tf.train.Saver()

# Longitudes totales de datasets
total = len(train[0])
total_Activos = len(train_Activos_data[0])
total_Inactivos = len(train_Inactivos_data[0])

# Arreglos de indices para minibatches
indices_Activos = np.arange(total_Activos)
indices_Inactivos = np.arange(total_Inactivos)

data_Activos = train_Activos_data[0]
labels_Activos = train_Activos_data[1]

data_Inactivos = train_Inactivos_data[0]
labels_Inactivos = train_Inactivos_data[1]

# Funcion para hacer shuffle de indices para minibatches
def next_epoch():
	np.random.shuffle(indices_Activos)
	np.random.shuffle(indices_Inactivos)

# Función para recopilar nuevo minibatch
def next_batch():
	a = np.roll(indices_Activos, mini_batch_size // 2)
	b = np.roll(indices_Inactivos, mini_batch_size // 2)

	muestra_Activos = a[0:mini_batch_size // 2]
	muestra_Inactivos = b[0:mini_batch_size // 2]

	# Tercero, tomo la muestra de datos segun los indices elegidos
	datos_batch_Activos = data_Activos[muestra_Activos,:]
	datos_batch_Inactivos = data_Inactivos[muestra_Inactivos,:]

	labels_Activos = np.ones((mini_batch_size // 2,), dtype = np.int)
	labels_Inactivos = np.zeros((mini_batch_size // 2,), dtype = np.int)

	data_batch = np.concatenate((datos_batch_Activos, datos_batch_Inactivos))
	label_batch = np.concatenate((labels_Activos, labels_Inactivos))

	# Shuffle
	shuf = np.arange(mini_batch_size)
	np.random.shuffle(shuf)

	data_batch = data_batch[shuf,:]
	label_batch = label_batch[shuf]

	return (data_batch, label_batch)
	

#--------------------------------------------------------------------
#			Execution phase
#--------------------------------------------------------------------

# n_batches: cantidad de minibatches
n_batches = len(train[0]) // mini_batch_size

patience_cnt = 0
minimo = np.iinfo(np.int32).max
stream = 20
# Entrenamiento del modelo
with tf.Session() as session:

	all_vars= tf.global_variables()
	
	def get_var(name):
		for i in range(len(all_vars)):
			if all_vars[i].name.startswith(name):
				return all_vars[i]
		return None


	tf.local_variables_initializer().run()
	tf.global_variables_initializer().run()						# Inicializacion de las variables

	for epoch in range(n_epochs):
	
		# Reset costos parciales por epoch
		tec = 0
		# Shuffle de indices
		next_epoch()

		for batch in range(n_batches):
			X_batch, y_batch = next_batch()													
			session.run(training_op, feed_dict = {phase: True, is_training: True, X: X_batch, y: y_batch})
			c = cost.eval(feed_dict = {phase: False, is_training: True, X: X_batch, y: y_batch})	
			tec += c / n_batches	

		# Cada 5 epochs testeo por ES
		if epoch % 5 == 0:
			cv = cost.eval(feed_dict = {phase: False, is_training: False, X: test[0], y: test[1]})
			# early stopping
			patience = 60 # Corta tras 300 epocas de no superar el minimo en 0,001
			min_delta = 0.0005
			if (minimo - cv) > min_delta:
				minimo = cv
				patience_cnt = 0
				save_path = saver.save(session, ruta)
			else:
				patience_cnt += 1

		# Cada 100 epochs imprimo costo de entrenamiento
		if epoch % 100 == 0:
			print ("Epoch %i: %f | %f | %f" % (epoch, tec, cv, minimo))

		if patience_cnt > patience:
			print("Early stopping at epoch: ", epoch)
			print("Costo de validacion minimo: ", minimo)
			break

	dictionary = {phase: False, is_training: False, X: test[0], y: test[1]}
	# dictionary = {phase: False, is_training: False, X: ev[0], y: ev[1]}
	saver.restore(session, ruta)

	# Recuperación de los pesos de cada capa
	pesos1 = get_var('dnn/hidden1/weights')
	pesos2 = get_var('dnn/hidden2/weights')
	pesos3 = get_var('dnn/hidden3/weights')
	pesos4 = get_var('dnn/outputs/weights')
	p1 = session.run(pesos1, feed_dict = dictionary)
	p2 = session.run(pesos2, feed_dict = dictionary)
	p3 = session.run(pesos3, feed_dict = dictionary)
	p4 = session.run(pesos4, feed_dict = dictionary)
	to_dump = (p1, p2, p3, p4)
	pkl.dump(to_dump, open(ruta_pickles + "he_10_o_weights_" +str(semilla)+ ".p", "wb"))
	# pkl.dump(to_dump, open(ruta_pickles + "he_10_o_EV_weights_" +str(semilla)+ ".p", "wb"))
	
	# Construccion de métrica MAP

	# Obtención de arreglos de predccion y targets. Calculo de entropia.
	predicciones = softmax.eval(feed_dict = dictionary)
	predicciones_activos = predicciones[:,1]
	entropia = sc.special.entr(predicciones_activos) + sc.special.entr(1 - predicciones_activos)
	targets = test[1]
	# targets = ev[1]

	indices_ascendente = np.argsort(entropia)
	predicciones_ordenadas = predicciones_activos[indices_ascendente]
	entropia_ordenadas = entropia[indices_ascendente]
	targets_ordenados = targets[indices_ascendente]

	# Elaboracion de lista para MAP
	lista_intervalos = []
	NErr_array = np.zeros(shape = len(targets))

	for i in range(0,len(targets)):
		intervalo_datos = predicciones_ordenadas[0:i+1]
		intervalo_targets = targets_ordenados[0:i+1]
		correctas = predicciones_correctas(intervalo_datos, intervalo_targets)
		Sn = sensibilidad(intervalo_datos,intervalo_targets)
		Sp = especificidad(intervalo_datos,intervalo_targets)
		NErr = (Sn + Sp) / 2
		NErr_array[i] = NErr
		umbral_entropia = entropia_ordenadas[i]
		umbral_certeza = np.abs(0.5 - predicciones_ordenadas[i])
		compuestos_tomados = len(intervalo_datos)
		intervalo = (intervalo_datos, intervalo_targets, correctas, Sn, Sp, NErr, umbral_entropia, umbral_certeza, compuestos_tomados)
		lista_intervalos.append(intervalo)

	pkl.dump(lista_intervalos, open(ruta_pickles + "he_10_o_tuples_" + str(semilla) + ".p", "wb"))
	# pkl.dump(lista_intervalos, open(ruta_pickles + "he_10_o_EV_tuples_" + str(semilla) + ".p", "wb"))

	print(np.mean(NErr_array))