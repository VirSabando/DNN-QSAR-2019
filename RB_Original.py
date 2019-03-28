#estas lineas son para ignorar unos warnings muy molestos
#1) para ignorar warnings respecto a CPU/GPU optimization de tensowflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#2) para ignorar el dependency bug de h5py
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import pickle as pkl
from Todeschini_41 import train_RB_data, train_NRB_data, train_data, test_data, ev


from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import dropout
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

# from tensorflow.contrib.training import stratified_sample

from sys import argv

# Semilla
semilla = argv[1]
tf.set_random_seed(int(semilla))
np.random.seed(42)

ruta = "/home/viri/Modelos_Finales/RB/Originales/mod_2_o/"+str(semilla)+"/sees.ckpt"
ruta_pickles = "/home/viri/Modelos_Finales/RB/Originales/mod_2_o/Pickles/"

# Tamanios de las capas de la red
n_inputs = 41		
n_hidden1 = 10
n_hidden2 = 5
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
learning_rate = 0.0001

# Numero de epochs 
n_epochs = np.iinfo(np.int32).max

# Tamanio de mini batch
mini_batch_size = 200

# Coeficiente lambda para Ln regularization
lambda_ln = 0.001

# Coeficiente de Dropout: keep probability
keep_prob_h1 = 0.8
keep_prob_h2 = 0.9

# Regularizador: L1
Ln_reg = tf.contrib.layers.l2_regularizer(lambda_ln)

# Definicion de la arquitectura de red a: batch norm
with tf.variable_scope("dnn"):
	with tf.contrib.framework.arg_scope([fully_connected], normalizer_fn = batch_norm, normalizer_params = bn_params, weights_regularizer = Ln_reg):
		X_drop = dropout(X, keep_prob_h1, is_training = phase)
		hidden1 = fully_connected(inputs=X_drop, num_outputs=n_hidden1, scope='hidden1')
		hidden1_drop = dropout(hidden1, keep_prob_h1, is_training = phase)
		hidden2 = fully_connected(inputs=hidden1_drop, num_outputs=n_hidden2, scope='hidden2')
		hidden2_drop = dropout(hidden2, keep_prob_h2, is_training = phase)
		logits = fully_connected(inputs=hidden2_drop, num_outputs=n_outputs, scope='outputs')

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
total = len(train_data[0])
total_RB = len(train_RB_data[0])
total_NRB = len(train_NRB_data[0])

# Primero, armo los arreglos de indices
indices_RB = np.arange(total_RB)
indices_NRB = np.arange(total_NRB)

data_RB = train_RB_data[0]
labels_RB = train_RB_data[1]

data_NRB = train_NRB_data[0]
labels_NRB = train_NRB_data[1]

# Función para recopilar nuevo minibatch
def next_batch():

	# Segundo, selecciono los indices para el minibatch
	muestra_RB = np.random.choice(indices_RB, size = mini_batch_size // 2, replace = False)
	muestra_NRB = np.random.choice(indices_NRB, size = mini_batch_size // 2, replace = False)

	# Tercero, tomo la muestra de datos segun los indices elegidos
	datos_batch_RB = data_RB[muestra_RB,:]
	datos_batch_NRB = data_NRB[muestra_NRB,:]

	labels_RB = np.ones((mini_batch_size // 2,), dtype = np.int)
	labels_NRB = np.zeros((mini_batch_size // 2,), dtype = np.int)

	data_batch = np.concatenate((datos_batch_RB, datos_batch_NRB))
	label_batch = np.concatenate((labels_RB, labels_NRB))

	# Shuffle
	shuf = np.arange(200)
	np.random.shuffle(shuf)

	data_batch = data_batch[shuf,:]
	label_batch = label_batch[shuf]

	return (data_batch, label_batch)
	

#--------------------------------------------------------------------
#			Execution phase
#--------------------------------------------------------------------

# n_batches: cantidad de minibatches
n_batches = len(train_data[0]) // mini_batch_size

# Listas de costos y accuracies para componer los plots
train_costs = []
test_costs = []
test_ES = []

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
											
	coordinator = tf.train.Coordinator()
	tf.train.start_queue_runners(session, coord=coordinator)

	for epoch in range(n_epochs):
		# Reset costos parciales por epoch
		tec = 0
		
		for batch in range(n_batches):
			X_batch, y_batch = next_batch()
			# print(epoch , X_batch[0])														
			session.run(training_op, feed_dict = {phase: True, is_training: True, X: X_batch, y: y_batch})
			c = cost.eval(feed_dict = {phase: False, is_training: True, X: X_batch, y: y_batch})	
			tec += c / n_batches	

		# Cada 5 epochs testeo por ES
		if epoch % 5 == 0:
			cv = cost.eval(feed_dict = {phase: False, is_training: False, X: test_data[0], y: test_data[1]})
			# early stopping
			patience = 60 # Corta tras 300 epocas de no superar el minimo en 0,001
			min_delta = 0.001
			if (minimo - cv) > min_delta:
				minimo = cv
				patience_cnt = 0

				save_path = saver.save(session, ruta)
			else:
				patience_cnt += 1
		
		if epoch % stream == 0:
			train_costs.append(tec)
			test_costs.append(cv)
			test_ES.append(minimo)

		# Cada 100 epochs imprimo costo de entrenamiento
		if epoch % 100 == 0:
			print ("Epoch %i: %f | %f " % (epoch, tec, cv))

		if patience_cnt > patience:
			print("Early stopping at epoch: ", epoch)
			print("Costo de validacion minimo: ", minimo)
			break

	dictionary = {phase: False, is_training: False, X: test_data[0], y: test_data[1]}
	# dictionary = {phase: False, is_training: False, X: ev[0], y: ev[1]}
	saver.restore(session, ruta)
	
	# Recuperación de los pesos de cada capa
	pesos1 = get_var('dnn/hidden1/weights')
	pesos2 = get_var('dnn/hidden2/weights')
	pesos3 = get_var('dnn/outputs/weights')
	p1 = session.run(pesos1, feed_dict = dictionary)
	p2 = session.run(pesos2, feed_dict = dictionary)
	p3 = session.run(pesos3, feed_dict = dictionary)
	to_dump = (p1, p2, p3)
	pkl.dump(to_dump, open(ruta_pickles + "mod_2_o_weights_" +str(semilla)+ ".p", "wb"))
	# pkl.dump(to_dump, open(ruta_pickles + "mod_2_o_EV_weights_" +str(semilla)+ ".p", "wb"))

	# Construccion de métrica MAP

	# Obtención de arreglos de predccion y targets. Calculo de entropia.
	predicciones = softmax.eval(feed_dict = dictionary)
	predicciones_activos = predicciones[:,1]
	entropia = sc.special.entr(predicciones_activos) + sc.special.entr(1 - predicciones_activos)
	# targets = test_data[1]
	targets = ev[1]

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

	pkl.dump(lista_intervalos, open(ruta_pickles + "mod_2_o_tuples_" + str(semilla) + ".p", "wb"))
	# pkl.dump(lista_intervalos, open(ruta_pickles + "mod_2_o_EV_tuples_" + str(semilla) + ".p", "wb"))