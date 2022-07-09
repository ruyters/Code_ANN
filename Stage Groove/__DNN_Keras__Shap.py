import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn as sk

from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix, confusion_matrix
from statistics import mean

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from keras import callbacks 

from scipy.stats import pearsonr, yeojohnson
from scipy.special import expit, logit
import coral_ordinal as coral
from keras.utils.np_utils import to_categorical  

import time
import shap
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore',category=DeprecationWarning)
warnings.filterwarnings(action='ignore',category=FutureWarning)
warnings.filterwarnings(action='ignore',category=Warning)

from __Extraction_data import *
from __Main import *
from __RedDim_DNN_Shapley import *
from __Results_Utility_functions import *
from __ExtractDyn import *
from __DNN_keras_addDict__ import *


def def_sample(f, p, r, lobe, nlobe, moy = True):
	
	'''
	Parameters
	----------
	f : 0-6 if moy = True, 0-99 else
	p : Tsubject.
	r : regressor.
	lobe : cerebral zone.
	nlobe : number of lobe.
	moy : default is True.

	Returns
	-------
	X_train, X_test, y_train y_test : balanced samples 
	weights : weight for each class.
	nc : X size (number of chanels).
	'''
	
	
	X, y, _, _ = summary_stat(p, r, lobe, moy, f, nlobe, False, False, True, bande = 'classic')
	
	
	y = pd.DataFrame(y)
	y = y.replace(2, 1)
	y = np.array(y)
	nc = len(X[1])
	
	RFE = False
	if RFE == True:
		
		if nlobe ==1:
			
			goodchanidx = joblib.load('/Users/swann/Université/Stage/Spyder/Analysis/RESULTATS_FINAUX_data/SVM_DT_Delta2hz_Periodicity/idx_SVM_lobes1_Groove.pkl')
			goodchann = goodchanidx[8]
			X = X[:, goodchann]
			
		if nlobe==2:
			goodchanidx = joblib.load('idx_SVM_lobes2_Groove.pkl')
			if LOBES == '["LFidx", "LPidx"]':
				goodchann = goodchanidx[6]
				X = X[:, goodchann]
			else:
				goodchanidx = joblib.load('idx_SVM_lobes2_Groove.pkl')
				goodchann = goodchanidx[11]
				X = X[:, goodchann]
				
		if nlobe==3:
			goodchanidx = joblib.load('/Users/swann/Université/Stage/Spyder/Analysis/RESULTATS_FINAUX_data/SVM_DT_Delta2hz_Periodicity/idx_SVM_lobes3_Groove.pkl')
			goodchann = goodchanidx[4]
			X = X[:, goodchann]
	
	print("SAMPLE SIZE:", nc)
	X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.15, shuffle = True)
	
	weights = class_weight.compute_class_weight(class_weight = 'balanced',
                                                classes =  np.unique(y_train),
                                                y = y_train.reshape(len(y_train)))
	
	if sum(y_test) < 5: 
		print("Unbalanced y_test :(")
		while(sum(y_test) < 5):
			X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.15, shuffle = True)
			weights = class_weight.compute_class_weight(class_weight = 'balanced',
		                                                classes =  np.unique(y_train),
		                                                y = y_train.reshape(len(y_train)))
	
	
	print("  ### Classes Weights", weights)

	return X_train, X_test, y_train, y_test, weights, nc
	

def create_classifier_clf(N_LAYERS, OPTIMIZER, ACTIVATION, NUM_CLASS, LR, NC):
	
	'''
	Parameters
	----------
	N_LAYERS : number of hidden layers (int).
	OPTIMIZER : méthodes de descente du gradient(string).
	ACTIVATION : activation function (string).
	NUM_CLASS : number of class to prodict (int).
	LR : learning rate (int).
	NC : Tinput size (int).

	Returns
	-------
	mod : model with parameters set.

	'''
	# Hyper parameters to custom
	# HIDDEN = [2 * NC, NC, 1.5 * NC, 60]
	HIDDEN = [70, 175, 32]
	DR = [0.2, 0.1, 0.1, 0.2]
	CLF = 'coral'

	# First layer for model
	mod = tf.keras.Sequential()
	mod.add(tf.keras.layers.Dense(units = NC, activation = ACTIVATION[0]))
	
	
	# For each layers : activation function and dropout
	for i in range(N_LAYERS):
		mod.add(tf.keras.layers.Dense(HIDDEN[i], activation = ACTIVATION[i+1]))
		mod.add(tf.keras.layers.Dropout(DR[i]))
	
	# Optimizer choice
	if CLF == 'coral':
		mod.add(coral.CoralOrdinal(num_classes = NUM_CLASS)) # Ordinal variable has 5 labels, 0 through 4.
			
		if OPTIMIZER == 'SGD':
			mod.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=LR, decay = 1e-10, momentum = 0.98),
				 loss = coral.OrdinalCrossEntropy(),
				  metrics = [coral.MeanAbsoluteErrorLabels()])
		if OPTIMIZER == 'adam':
			mod.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=LR,decay = 1e-6),
				 loss = coral.OrdinalCrossEntropy(),
				  metrics = [coral.MeanAbsoluteErrorLabels()])
		if OPTIMIZER == 'rmsprop':
			 mod.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=LR),
				 loss = coral.OrdinalCrossEntropy(),
				  metrics = [coral.MeanAbsoluteErrorLabels()])
		if OPTIMIZER == 'nadam':
			 mod.compile(optimizer = tf.keras.optimizers.Nadam(learning_rate=LR, decay = 1e-6),
				 loss = coral.OrdinalCrossEntropy(),
				  metrics = [coral.MeanAbsoluteErrorLabels()])
		if OPTIMIZER == 'adadelta':
			 mod.compile(optimizer = tf.keras.optimizers.Adadelta(learning_rate=LR),
				 loss = coral.OrdinalCrossEntropy(),
				  metrics = [coral.MeanAbsoluteErrorLabels()])
	
	# If classification without coral-ordinal
	if CLF == "clf":
		for i in range(N_LAYERS):
			mod.add(tf.keras.layers.Dense(HIDDEN[i], activation = ACTIVATION[i+1]))
			mod.add(tf.keras.layers.Dropout(DR[i]))
		mod.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mae'])
	
	return mod	 


def make_dnn(r, f, list_part):
	'''
	Parameters
	----------
	r : regressor.
	f : frequency.
	list_part : subjects.

	Returns
	-------
	Run and evaluation model : 
		df : results
		list_dict_history : details about results
		shap : shapley model
		df_shap : shapley valuews'''
		
	reg_names = ['Periodicity', 'Syncope', 'Groove']
	CLF = 'ordinal'
	
	# Reproductibility
	SEED = 4242
	np.random.seed(SEED)
	tf.random.set_seed(SEED)

	
	# Hyperparameters to custom
	# LOBE = ['LFidx','LPidx']
	LOBE = 'LFidx' # , 'LPidx', 'LTDidx'
	N_LOBE = 1
	NUM_CLASS = 2

	
	EPOCHS = 3000
	BATCH = 40
	N_LAYERS = 3
	OPTIMIZER = 'SGD' #nadam, adam SGD
	ACTIVATION = ['elu', 'elu', 'relu', 'tanh'] #elu relu tanh linear
	LR = 1e-4
	# METRICS = ['mse','mae']


	# Dict classification
	df = {
		'loss': [], 'mae_labels' : [], 
		'val_loss' : [], 'val_mae_labels' : [],
		'val_mae_labels_sd' : [], 'acc' : []
		 }
	
	
	list_dict_history = []
	df_shap = pd.DataFrame()
	
	try : 
####### For each frequency...
		for p in list_part: 

			# Dict classification
			dict_res = {
			'loss': [], 'mae_labels' : [], 
			'val_loss' : [], 'val_mae_labels' : [], 'val_mae_labels_sd': [],
			'acc' : []
			 }
			
			# Run  model 
			X_train, X_test, y_train, y_test, class_weights, NC = def_sample(f, p, r, LOBE, N_LOBE, moy = False)
			
			try: 
				cweights = {0:class_weights[0], 1:class_weights[1], 2:class_weights[2]}
			
			except IndexError:
				cweights = {0:class_weights[0], 1:class_weights[1]}
				
			
			earlystopping = callbacks.EarlyStopping(monitor ="val_loss",  
                                        mode ="min", patience = 20,  
                                        restore_best_weights = True) 
			
			model = create_classifier_clf(N_LAYERS, OPTIMIZER, ACTIVATION, NUM_CLASS, LR, NC)
			

			h = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH, 
				 verbose=1, validation_split=0.1, class_weight = cweights, callbacks = [earlystopping]) 
			
			# Predictions
			if CLF == 'clf': 
				# output = model.evaluate(X_test, y_test, verbose=0) 
				output = model.predict(X_test)
				output[(output < 0.5)] = 0 ; output[(output > 0.5)] = 1
				print("OUTPUT:", output)
				acc = np.mean(output == y_test)
				print("Accuracy Clf: ", acc)
				print(confusion_matrix(y_test, output))
				# print(h.history.keys())


			if CLF == 'ordinal':
				preds = model.predict(X_test) 
				probs = pd.DataFrame(coral.ordinal_softmax(preds).numpy())
				output = np.array(probs.idxmax(axis = 1))
				acc = np.mean(output == y_test)
				
				# Compare to the logit-based cumulative probs
				cum_probs = pd.DataFrame(preds).apply(expit)
				output2 = np.array(cum_probs.apply(lambda x: x > 0.5).sum(axis = 1))
				
				dict_res = add_dict_clf(dict_res, h, acc)
				
				print("---> Difference between predictions:", np.mean(output == output2))
				print("-> Predictions 1 : ", output)
				print("-> Predictions 2 : ", output2)
				print("-> Accuracy of label : ", np.mean(output == y_test), np.mean(output2 == y_test))
				print(confusion_matrix(y_test, output))
				print(confusion_matrix(y_test, output2))
				
			if acc > 0.65:
				if sum(output) > 0:
					# Save confusion matrix
					#cm = multilabel_confusion_matrix(y_test, output)
					cm = confusion_matrix(y_test, output)
					title = 'CM' + '_p' + str(p) + '_f' + str(f) + '.pkl'
					joblib.dump(cm, str(title))
					
					# Save feature importance (shap)
					df_fi = call_shapley_featureImportance(model, X_test, NC)
					title = 'Shap_FI' + '_p' + str(p) + '_f' + str(f) + '.pkl'
					joblib.dump(df_fi, str(title))
					
					# Save model History
					title = 'history' +  str(p) + '_f' + str(f) + '.pkl'
					joblib.dump(pd.DataFrame(h.history), str(title))
				
					# Shapley
					shap_values = call_shapley(model, X_test)
					shap.plots.beeswarm(shap_values, max_display=10) # color 
					shap.waterfall_plot(shap_values = shap_values[0], max_display=20)
					title = "model_p" + str(p) +"_f" + str(f) + ".h5"
					model.save_weights(str(title))
					print("Saved model to disk")
				
				list_dict_history.append(h)
				
				if CLF == 'ordinal': 
					df = add_final_dict_clf(dict_res, df)
					
			df_shap = call_shapley_featureImportance(model, X_test, NC)

		
		return df, list_dict_history, shap, df_shap

	except KeyboardInterrupt:
		
		return df, list_dict_history, shap, df_shap

if __name__ == '__main__':	
	
	r = 2
	f = [6]
	# list_part = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
	#list_part = [0, 1, 7, 21, 23]
	list_part = [1]
	timerStart = time.time()
	df_grv2, dict_h, shap, df_shap = make_dnn(r, f, list_part)
	timerStop = time.time()
	#np.save('results_groove_160522.npy', rslt_grv)
	title = "RESULTS_p1_2306"
	# joblib.dump(df_grv, str(title))
	print("Time : ", (timerStop - timerStart)/60, "min")
	
	print(df_shap.index)
	
	'''
	timerStart = time.time()
	resultat_syncope = make_dnn(data, 1, 28)
	timerStop = time.time()
	np.save('rslt_syncopey.npy', resultat_syncope)
	print("Time : ", (timerStop - timerStart)/60, "min")
	
	
	timerStart = time.time()
	resultat_groove = make_dnn(data = data, r = 2, nb_freq = 28, method = 'regression')
	timerStop = time.time()
	
	val_importance = resultat_groove[3]
	shap_values = resultat_groove[2]
	shap.plots.beeswarm(shap_values, max_display=20) # color 
	shap.waterfall_plot(shap_values = shap_values[0], max_display=250)
	shap.plots.bar(shap_values.abs.mean(0),  max_display=250)
	shap.plots.heatmap(shap_values = values, max_display = 10)
	shap.plots.force(shap_values)
	labels =[]
	feature_idx = np.ndarray([labels['FEATURE'] % str(i) for i in range(240)])
	
	values = pd.DataFrame(np.transpose(shap_values.values.abs.mean(0)))
	print(shap_values.abs)
	print(shap_values.values[instance,feature])
	


	
	red = Color("red")
	colors = list(red.range_to(Color("blue"), 15))
	plot_DNN_clf(res_groove_f50-99, 'groove', colors)
	
	
	# np.save('rslt_groove.npy', resultat_groove)
	# print("Time : ", (timerStop - timerStart)/60, "min")


	
	results = resultat[0]
	history = resultat[1]
	len(history['history'])
	history = resultat[1]['history'][0].history.keys()
	t = history["history"][0]
	print(t.history['loss'])
	'''

	
	'''
	 Score was 0.8148148148148148, and its parameters were:
             {'num_layers': 3, 'optimizer': 'rmsprop', 'activation': 'relu', 'activation0': 'relu', 
              'n_units_l0': 381, 'dropout_rate0': 0.49917623275724726, 'activation1': 'tanh',
              'n_units_l1': 506, 'dropout_rate1': 0.4415360373216983, 'activation2': 'tanh',
              'n_units_l2': 482, 'dropout_rate2': 0.4997487736798049, 'learning_rate': 2.404477580358311e-06,
              'epochs': 1335, 'batch': 26, 'part': 28, 'freq': 3}
	'''
	
	'''
	LP
	-> Predictions 1 :  [1 0 1 0 0 0 0 2 0 0 0 0 0 0 0 2 1 1 1 2 0 1 0 1 0 0 1 2]
	-> Predictions 2 :  [1 0 1 0 0 0 0 2 0 0 0 0 0 0 0 2 1 1 1 2 0 1 0 1 0 0 1 2]
	-> Accuracy of label :  0.42857142857142855 0.42857142857142855
	[[8 5 3]
	 [6 1 1]
	 [2 2 0]]
	Time :  1.890070950984955 min
	
	LF
	-> Predictions 1 :  [0 0 0 2 2 1 0 2 0 0 1 1 0 0 2 0 0 0 0 0 2 2 1 0 0 0 0 2]
	-> Predictions 2 :  [0 0 0 2 2 1 0 2 0 0 1 1 0 0 2 0 0 0 0 0 2 2 1 0 0 0 0 2]
	-> Accuracy of label :  0.42346938775510207 0.42346938775510207
	[[9 3 4]
	 [6 0 2]
	 [2 1 1]]
	Time :  1.9181453029314677 min
	'''

