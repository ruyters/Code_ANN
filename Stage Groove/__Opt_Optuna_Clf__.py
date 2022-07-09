# -*- coding: utf-8 -*-

import optuna
from optuna.integration import KerasPruningCallback
from optuna.trial import TrialState
from optuna.pruners import ThresholdPruner

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix, confusion_matrix
from statistics import mean
from scipy.special import expit, logit

import pandas as pd
import numpy as np
import tensorflow as tf
import coral_ordinal as coral
import sklearn as sk
import keras
from keras import callbacks 

import joblib

from __Extraction_data import *
from __Main import *
from __RedDim_DNN_Shapley import *
from __Results_Utility_functions import *
from __ExtractDyn import *
from __RedDim_InterClass import * 

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def def_input_fn(f, p, r, SKF, pretraining = False):
	'''
	Parameters
	----------
	f : frequency (list).
	p : subject.
	r : Tregressor.
	SKF : if true, stratified kfold.
	trial : object of optuna.
	pretraining : TYPE, optional
		DESCRIPTION. The default is False.

	Returns
	-------
	X_train :. ; X_test :. ; y_train : ; y_test :. ; weights :.; nc : number of channels. '''

	moy = True
	
	#if type(LOBES) == str:
	#	N_LOBES = 1
	nlobes = 3
	lobes = ['LFidx', 'LPidx', 'LTDidx']
	
	X, y, _, _ = summary_stat(p, r, lobes, moy, f, nlobes, False, False, binary = True, bande = 'classic')
	y = pd.DataFrame(y)
	y = y.replace(2, 1)
	y = np.array(y)
	
	RFE = False
	if RFE == True:
		if N_LOBES ==1:
			goodchanidx = joblib.load('idx_SVM_lobes1_Groove.pkl')
			goodchann = goodchanidx[8]
			X = X[:, goodchann]
		if N_LOBES==2:
			goodchanidx = joblib.load('idx_SVM_lobes2_Groove.pkl')
			if LOBES == '["LFidx", "LPidx"]':
				goodchann = goodchanidx[6]
				X = X[:, goodchann]
			else:
				goodchanidx = joblib.load('idx_SVM_lobes2_Groove.pkl')
				goodchann = goodchanidx[11]
				X = X[:, goodchann]
		if N_LOBES==3:
			goodchanidx = joblib.load('idx_SVM_lobes3_Groove.pkl')
			goodchann = goodchanidx[4]
			X = X[:, goodchann]
		
	nc = len(X[1])
	print("SAMPLE SIZE:", nc)
	X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.20, shuffle = True)
	weights = class_weight.compute_class_weight(class_weight = 'balanced',
                                                classes =  np.unique(y_train),
                                                y = y_train.reshape(len(y_train)))
	
	if sum(y_test) < 1: 
		print("Unbalanced y_test :(")
		while(sum(y_test) < 1):
			X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.20, shuffle = True)
			weights = class_weight.compute_class_weight(class_weight = 'balanced',
		                                                classes =  np.unique(y_train),
		                                                y = y_train.reshape(len(y_train)))
	
	print("  ### Classes Weights", weights)
	return X_train, X_test, y_train, y_test, weights, nc


def create_classifier(trial, nc, NUM_CLASS):
	n_layers = trial.suggest_int("num_layers", LAYERS[0], LAYERS[1])
	opt = trial.suggest_categorical('optimizer', OPTIMIZER)
	act = trial.suggest_categorical('activation', ACTIVATION)
	
	mod = tf.keras.Sequential()
	mod.add(tf.keras.layers.Dense(units = nc, activation = act))
	
	for i in range(n_layers):
		act = trial.suggest_categorical('activation{}'.format(i), ACTIVATION)
		NH = trial.suggest_int("n_units_l{}".format(i), NUM_HIDDEN[0], NUM_HIDDEN[1], log=True)
		mod.add(tf.keras.layers.Dense(NH, activation = act))
		mod.add(tf.keras.layers.Dropout(trial.suggest_uniform('dropout_rate{}'.format(i),DR[0], DR[1])))

	clf = "coral"
	if clf == "coral":
		mod.add(coral.CoralOrdinal(num_classes = NUM_CLASS)) # Ordinal variable has 5 labels, 0 through 4.
		print("Coral : done")
		
		if opt == 'SGD':
			mod.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=trial.suggest_loguniform('learning_rate', 1e-10, 0.001), 
												momentum = trial.suggest_uniform('momentum', MOMENTUM[0], MOMENTUM[1])),
				 loss = coral.OrdinalCrossEntropy(),
				  metrics = [coral.MeanAbsoluteErrorLabels()])
		if opt == 'adam':
			mod.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=trial.suggest_loguniform('learning_rate', 1e-10, 0.001)),
				 loss = coral.OrdinalCrossEntropy(),
				  metrics = [coral.MeanAbsoluteErrorLabels()])
		if opt == 'rmsprop':
			 mod.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=trial.suggest_loguniform('learning_rate', 1e-10, 0.001)),
				 loss = coral.OrdinalCrossEntropy(),
				  metrics = [coral.MeanAbsoluteErrorLabels()])
		if opt == 'nadam':
			 mod.compile(optimizer = tf.keras.optimizers.Nadam(learning_rate=trial.suggest_loguniform('learning_rate', 1e-10, 0.001)),
				 loss = coral.OrdinalCrossEntropy(),
				  metrics = [coral.MeanAbsoluteErrorLabels()])
		if opt == 'adadelta':
			 mod.compile(optimizer = tf.keras.optimizers.Adadelta(learning_rate=trial.suggest_loguniform('learning_rate', 1e-10, 0.001)),
				 loss = coral.OrdinalCrossEntropy(),
				  metrics = [coral.MeanAbsoluteErrorLabels()])
		 
	if clf == "clf":
		mod.add(tf.keras.layers.Dense(1, activation = act))
		mod.compile(loss="binary_crossentropy", optimizer = 'adam', metrics = 'binary_accuracy')
		
		
	return mod	 


def objective(trial):
	
	keras.backend.clear_session()
	t = trial.number
	
	# PARAMS TO CUSTOM
	SKF = False 
	r = 2
	NB_CLASS = [2, 2, 2]
	NUM_CLASSES = NB_CLASS[r]
	THRESHOLD = 0.65
	
	# Fit the model on the training data
	EPOCHS = trial.suggest_int("epochs", EPOCH[0], EPOCH[1])
	BATCH_SIZE = trial.suggest_int("batch", BATCH[0], BATCH[1])
	p = trial.suggest_categorical("part", PART)
	f = FREQ
	
	i = 0
	if SKF == False:
		
		X_train, X_test, y_train, y_test, weights, nc = def_input_fn(f, p, r, False, pretraining = False)
		model = create_classifier(trial, nc, NUM_CLASSES)
		
		try:
			cweights = {0:weights[0], 1:weights[1], 2:weights[2]}
			
		except IndexError:
			cweights = {0:weights[0], 1:weights[1]}	
		
		earlystopping = callbacks.EarlyStopping(monitor ="val_loss",  
                                       mode ="min", patience = 500,  
                                       restore_best_weights = True) 
		h = model.fit(
				X_train,
				y_train,
				batch_size = BATCH_SIZE,
				callbacks = [optuna.integration.TFKerasPruningCallback(trial, "val_loss")],
				# callbacks = [earlystopping],
				epochs = EPOCHS,
				validation_data = (X_test, y_test),
				verbose = 1,
				class_weight = cweights
			)
		
		
		clf = 'coral'
		if clf == 'coral':
			# Predictions
			preds = model.predict(X_test) 
			probs = pd.DataFrame(coral.ordinal_softmax(preds).numpy())
			output = np.array(probs.idxmax(axis = 1))
			acc = np.mean(output == y_test)
			
			# Compare to the logit-based cumulative probs
			cum_probs = pd.DataFrame(preds).apply(expit)
			output2 = np.array(cum_probs.apply(lambda x: x > 0.5).sum(axis = 1))
			print("Difference between predictions:", np.mean(output == output2))
						
			print("-> Predictions 1 : ", output)
			print("-> Predictions 2 : ", output2)
			print("-> Accuracy of label : ", np.mean(output == y_test), np.mean(output2 == y_test))
			
			
			if acc > THRESHOLD:
				# Save confusion matrix
				#cm = multilabel_confusion_matrix(y_test, output)
				cm = confusion_matrix(y_test, output)
				title = 'CM' + '_p' + str(p) + '_t' + str(t) + '_f' + str(f) + '.pkl'
				joblib.dump(cm, str(title))
				
				# Save feature importance (shap)
				df_fi = call_shapley_featureImportance(model, X_test, nc)
				title = 'Shap_FI' + str(nc) +'ch' + '_p' + str(p) + '_t' + str(t) + '_f' + str(f) + '.pkl'
				joblib.dump(df_fi, str(title))
				i = i + 1
				
				# Save model History
				title = 'history' +  str(p) + '_t' + str(t) + '_f' + str(f) + '.pkl'
				joblib.dump(pd.DataFrame(h.history), str(title))
		
		if clf == 'clf':
			output = model.predict(X_test) 
			acc = np.mean(output == y_test)
			print(y_test)
			print(output)
	
	if SKF == True: 
		X, y, nc = def_input_fn(f, p, r, True, pretraining = False)
		skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 22)
		acc_list = []
		model = create_classifier(trial, nc, NUM_CLASS)

		i = 0
		for train_index, test_index in skf.split(X, y):
			
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			
			print("  Shape y_train : ", y_train.shape)
			print(" - - - - Model fitting - - - -")
			
			h = model.fit(
					X_train,
					y_train,
					batch_size = BATCH_SIZE,
					callbacks = [optuna.integration.TFKerasPruningCallback(trial, "loss")],
					epochs = EPOCHS,
					validation_split=0.1,
					verbose = 0
				)
			
			preds = model.predict(X_test) 
			probs = pd.DataFrame(coral.ordinal_softmax(preds).numpy())
			output = np.array(probs.idxmax(axis = 1))
			acc_kfold = np.mean(output == y_test)
			acc_list.append(acc_kfold)
			print("OUTPUT:", output)
			print("TRUE:", y_test)
			
			if acc_kfold > THRESHOLD:
				
				if sum(output) > 0: 
					# Save confusion matrix
					cm = multilabel_confusion_matrix(y_test, output)
					title = 'CMv2' + '_p' + str(p) + '_t' + str(t) + '_f' + str(f) + '.pkl'
					joblib.dump(cm, str(title))
					
					# Save feature importance (shap)
					df_fi = call_shapley_featureImportance(model, X_test, nc)
					title = 'Shap_Fv2I' + str(nc) +'ch' + '_p' + str(p) + '_t' + str(t) + '_f' + str(f) + '.pkl'
					joblib.dump(df_fi, str(title))
					i = i + 1
					
					# Save model History
					title = 'historyv2' +  str(p) + '_t' + str(t) + '_f' + str(f) + '.pkl'
					joblib.dump(pd.DataFrame(h.history), str(title))

		acc = mean(acc_list)

		# ('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))   
	
	print(" - - - -  Accuracy (Mean) == ", acc)
	print("SUM OUTPUT:", sum(output))
	
	if acc > THRESHOLD:
		if sum(output)>0:
			shap_values = call_shapley(model, X_test)
			shap.plots.beeswarm(shap_values, max_display=10) # color 
			shap.waterfall_plot(shap_values = shap_values[0], max_display=20)
			title = "model" + str(t) + "_" + str(i) + ".h5"
			model.save_weights(str(title))
			print("Saved model to disk")
	
	if sum(output) > 3:
		print(1-abs(acc))
		return (1-abs(acc))
	
	else:
		return 1
	

if __name__ == "__main__":
	
	'''
Notes : penser à changer "binary" dans l'appel à summary lorsque la classification 
		est binaire (uniquement les classes "low" et "high" pour le groove et la syncope'
	'''
	
	
	LR = [1e-5,1e-3]
	EPOCH = [200, 1500]
	BATCH = [15, 35]
	LAYERS = [1, 3]
	NUM_HIDDEN = [25, 300]
	OPTIMIZER = ['SGD'] # 'SGD', 'adam', 'rmsprop', 'nadam', 'adadelta'
	ACTIVATION = ['relu', 'tanh', 'elu', 'softmax'] # ['relu', 'tanh', 'sigmoid', 'elu', 'linear']
	DR = [0.1, 0.4]
	MOMENTUM = [0.5, 0.98]
	# PART = [0, 1, 4, 6, 7, 9, 10, 12, 13, 14, 18, 20, 21, 22, 23, 25, 26, 27]
	
	LOSS = 'mae'
	METRICS = ['mae']
	
	PART = [1]
	FREQ = [6, 3, 3]

	NUM_CLASS = 2
	TRIALS = 500
	
	# GROOVE - POUR P1 SELON LES RESULTATS SVM / Decision Tree	à 2Hz
	LOBE_3 = [["LFidx", "LPidx", "LTDidx"]]
	LOBE_2 = [["LFidx", "LPidx"], ["LTGidx", "LTDidx"]]
	LOBE_1 = ["LTDidx"]
	
	study = optuna.create_study(direction="minimize")
	#study.optimize(objective, timeout=300, show_progress_bar=True)
	study.optimize(objective, n_trials=TRIALS, show_progress_bar=True)
	title =  'GrooveLobe_noRFE_3L_p1.pkl'
	joblib.dump(study, str(title))
	# Trials storage
	df = study.trials_dataframe()
	
	'''	
	for LOBE in LOBE_1:
		LOBES = LOBE
		N_LOBES = 1 
		study = optuna.create_study(direction="minimize")
		#study.optimize(objective, timeout=300, show_progress_bar=True)
		study = study.optimize(objective, n_trials=TRIALS, show_progress_bar=True)
		title =  'GrooveLobe_RFE_' + str(LOBES) +'_p1.pkl'
		joblib.dump(study, str(title))
	
	
	for LOBE in LOBE_2:
		LOBES = LOBE
		N_LOBES = 2
		study = optuna.create_study(direction="minimize")
		#study.optimize(objective, n_trials=TRIALS, show_progress_bar=True)
		study = study.optimize(objective, n_trials=TRIALS, show_progress_bar=True)
		title =  'GrooveLobe_RFE_' + str(LOBES) +'_p1.pkl'
		joblib.dump(study, str(title))
		

	
	for LOBE in LOBE_3:
		LOBES = LOBE
		N_LOBES = 3 
		study = optuna.create_study(direction="minimize")
		#study.optimize(objective, timeout=300, show_progress_bar=True)
		study = study.optimize(objective, n_trials=TRIALS, show_progress_bar=True)
		title =  'GrooveLobe_RFE_' + str(LOBES) +'_p1.pkl'
		joblib.dump(study, str(title))
	
	'''
	# study = joblib.load("GrooveLobeLFLPLTD2_p1.pkl")
	
	
	##### Load Studie ####
	# study = joblib.load("RESULTS2305.pkl")

	# Print main information
	trial = study.best_trial
	pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
	complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]
	
	print("Study statistics: ")
	print("  Number of finished trials: ", len(study.trials))
	print("  Number of pruned trials: ", len(pruned_trials))
	print("  Number of complete trials: ", len(complete_trials))
	print("Best trial:")
	
	trial = study.best_trial
	print("  Value: ", trial.value)
	print("  Params: ")
	for key, value in trial.params.items():
		print("    {}: {}".format(key, value))
	
	
	number_of_trials = len(study.trials)
	first_trial = study.trials[0]
	best_score_so_far = first_trial.value
	
	for report_index in range(1, number_of_trials):
	    trial_to_report = study.trials[report_index]
	    score_of_trial_to_report = trial_to_report.value
	    improved_score = (score_of_trial_to_report <= best_score_so_far)
	    if improved_score:
	        best_score_so_far = score_of_trial_to_report
	        print('\nTrial {}:'.format(trial_to_report.number), end=' ')
	        print('began at {}.'.format(trial_to_report.datetime_start))
	        print('Score was {},'.format(trial_to_report.value), end=' ')
	        print('and its parameters were: {}\n'.format(trial_to_report.params))
			 
	# Main / global Visu
	optuna.visualization.matplotlib.plot_param_importances(study)
	optuna.visualization.matplotlib.plot_optimization_history(study)
	optuna.visualization.matplotlib.plot_slice(study)
	optuna.visualization.matplotlib.plot_slice(study, ['learning_rate'])
	# optuna.visualization.matplotlib.plot_slice(study, ['learning_rate', 'freq'])
	optuna.visualization.matplotlib.plot_contour(study, ['learning_rate', 'optimizer'])
	# optuna.visualization.matplotlib.plot_contour(study, ['freq', 'part'])
	#optuna.visualization.matplotlib.plot_contour(study, ['learning_rate','freq'])
	#optuna.visualization.matplotlib.plot_contour(study, ['freq','batch'])
	optuna.visualization.matplotlib.plot_contour(study, ['epochs','optimizer'])
	optuna.visualization.matplotlib.plot_intermediate_values(study)
	

	'''
	optuna.visualization.plot_param_importances(study)
	optuna.visualization.plot_optimization_history(study)
	optuna.visualization.plot_slice(study)
	optuna.visualization.plot_slice(study, ['n_units_l1'])
	optuna.visualization.plot_contour(study, ['lr', 'optimizer'])
	4.04.22
	Study statistics: 
	  Number of finished trials:  100
	  Number of pruned trials:  76
	  Number of complete trials:  24
	Best trial:
	  Value:  0.5286458134651184
	  Params: 
	    num_layers: 5
	    optimizer: Adam
	    activation: sigmoid
	    n_units_l0: 277
	    dropout_rate: 0.4423560472299739
	    n_units_l1: 453
	    n_units_l2: 370
	    n_units_l3: 159
	    n_units_l4: 381
	    learning_rate: 5.0507417342874895e-06
	    epochs: 635
	    batch: 20
	    part: 2
	    freq: 20
	
	Trial 637 finished with value: 0.5185185185185185 and parameters: {'num_layers': 7, 'optimizer': 'rmsprop', 'activation': 'tanh', 'activation0': 'sigmoid', 'n_units_l0': 508, 'dropout_rate0': 0.48590344709964745, 'activation1': 'sigmoid', 'n_units_l1': 117, 'dropout_rate1': 0.4524635558251613, 'activation2': 'tanh', 'n_units_l2': 125, 'dropout_rate2': 0.39251611151022403, 'activation3': 'relu', 'n_units_l3': 114, 'dropout_rate3': 0.2550939630491904, 'activation4': 'sigmoid', 'n_units_l4': 163, 'dropout_rate4': 0.33207007553363704, 'activation5': 'relu', 'n_units_l5': 464, 'dropout_rate5': 0.4238155248622878, 'activation6': 'sigmoid', 'n_units_l6': 138, 'dropout_rate6': 0.44422976776460127, 'learning_rate': 6.9994856042890055e-06, 'epochs': 1506, 'batch': 23, 'part': 28, 'freq': 33}. Best is trial 199 with value: 0.8148148148148148.
	'''
	
	
	
	'''
	def input saved:
		nlobe = 1
		lobe = 'LFidx'
		moy = True
		lobe_list = ['LFidx', 'LPidx', 'LTGidx', 'LTDidx', 'LOidx']
		
		if pretraining == True:
			
			x = data[0][:, 0:140, 0:246, f]
			X = x.reshape(4200, 246)
			X = StandardScaler().fit_transform(X)
			
			Y = pd.DataFrame()
			for i in range(0,30):
				y = data[1][i, 0:140, r] ; y = y.reshape(140, 1)
				y = pd.DataFrame(y) ; y = round(y, 6) ; y = y.astype(str)
				l = np.unique(y) ; y = y.replace(list(l), list(range(0,len(l))))
				y = y.replace([0, 1, 2], [0, 0, 0]) ; y = y.replace([3], [1]) ; y = y.replace([4, 5], [2, 2])
				Y = Y.append(y)
			Y = np.array(Y)
			y = (Y.astype(float))
			print("Shape y ", y.shape)
		
		else:
			X, y, _, _ = summary_stat(p, r, lobe, moy, f, nlobe, False)
			nc = len(X[1])

		if SKF == False:
			X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.20, shuffle = True)
			weights = class_weight.compute_class_weight(class_weight = 'balanced',
		                                                classes =  np.unique(y_train),
		                                                y = y_train.reshape(len(y_train)))
			print("  ### ", weights)
			return X_train, X_test, y_train, y_test, weights, nc
		
		else:
			
			return X, y, nc
		
	'''
	