#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 19:08:58 2022

@author: swann
"""
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from statistics import mean
from sklearn.feature_selection import RFE, RFECV


from colour import Color
import pandas as pd
import numpy as np
import sklearn as sk

from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

from __Extraction_data import *
from __Main import *
from __RedDim_DNN_Shapley import *
from __Results_Utility_functions import *
from __ExtractDyn import *
from __DNN_keras_addDict__ import *

import joblib

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def def_sample(f, p, r, lobe, nlobe, moy):

	X, y, _, _ = summary_stat(p, r, lobe, moy, f, nlobe, False, False, True, bande = 'classic')
	y = pd.DataFrame(y)
	y = y.replace(2, 1)
	y = np.array(y)
	nc = len(X[1])
	print("SAMPLE SIZE:", nc)
	
	X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.1, shuffle = True)
	
	weights = class_weight.compute_class_weight(class_weight = 'balanced',
                                                classes =  np.unique(y_train),
                                                y = y_train.reshape(len(y_train)))
	print("  ### Classes Weights", weights)
	
	if sum(y_test) < 5: 
		print("Unbalanced y_test :(")
		while(sum(y_test) < 5):
			X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.1, shuffle = True)
			weights = class_weight.compute_class_weight(class_weight = 'balanced',
		                                                classes =  np.unique(y_train),
		                                                y = y_train.reshape(len(y_train)))
	
	
	
	return X_train, X_test, y_train, y_test, weights, nc


def getimportance(fitted_estimator):                                                           
	return fitted_estimator.coef_
	

def create_DecisionTree(MAX_DEPTH, MIN_SAMPLES, LOBE, N_LOBES, f, p ,r, SEED, MFS): 
	np.random.seed(SEED)

	X_train, X_test, y_train, y_test, class_weights, NC = def_sample(f, p, r, LOBE, N_LOBES, moy = True)
	cweights = {0:class_weights[0], 1:class_weights[1]}	
	clf_DT = DecisionTreeClassifier(criterion = "entropy", max_depth = MAX_DEPTH, min_samples_leaf = MIN_SAMPLES, class_weight = cweights, random_state = SEED)

	print(" - - - - Model fitting - - - -")
	dt = clf_DT.fit(X_train,y_train)
	print(" - - - - Model predicting - - - -")
	y_pred = clf_DT.predict(X_test)
	print("Y_test : ", y_test)
	print("Y_pred : ", y_pred)
	acc = metrics.accuracy_score(y_test, y_pred)
	print("Accuracy without RFE:", acc)
	plot_tree(dt, filled = True)
	plt.show()
	cm = confusion_matrix(y_test, y_pred)
	print("Confusion Matrix: \n", cm)

	if rf == True:
		
		print(" - - - - RFE fitting - - - -")
		cv = StratifiedKFold(CV, shuffle = True, random_state = SEED)
		min_features_to_select = MFS # Minimum number of features to consider len(X_train[1]) / 2
		
		rfecv = RFECV(
			estimator = clf_DT,
			step = 1,
			cv = cv,
			scoring = "accuracy",
			min_features_to_select = min_features_to_select,
			verbose = 0
		)
		
		rfecv_train = rfecv.fit(X_train, y_train)
		print("Optimal number of features : %d" % rfecv_train.n_features_)
		rfe_pred = rfecv.predict(X_test)
	
		print("Y_test : ", y_test) ; print("Y_pred : ", rfe_pred)
		acc_rfe = metrics.accuracy_score(y_test, rfe_pred)
		cm = confusion_matrix(y_test, rfe_pred)
		print("Accuracy with RFE : ", acc_rfe) ; print("Confusion Matrix :\n", cm)
		
		title = "Accuracy for Decision Tree (Part: " +str(p) + ", Freq: "+ str(f[0])+", Lobe(s): " + str(LOBE) + ")"
		titlefig = "DT_p" + str(p) + "_f"+ str(f[0]) + "_nbL" + str(LOBE) + ".svg"
		# fig = plt.figure(plt.figure(figsize=(10, 5)))
		plt.xlabel("Number of features selected")
		plt.ylabel("Cross validation score (accuracy)")
		plt.plot(
			range(min_features_to_select, len(rfecv_train.grid_scores_) + min_features_to_select),
		    rfecv_train.grid_scores_, linestyle = "dashed")
		plt.plot(
		    range(min_features_to_select, len(rfecv_train.cv_results_['mean_test_score']) + min_features_to_select),
		    rfecv_train.cv_results_['mean_test_score'], color = "black")
		plt.fill_between(range(min_features_to_select, len(rfecv_train.cv_results_['mean_test_score']) + min_features_to_select), 
				   rfecv_train.cv_results_['mean_test_score'] - rfecv_train.cv_results_['std_test_score'], 
				   rfecv_train.cv_results_['mean_test_score'] + rfecv_train.cv_results_['std_test_score'], color= 'black', alpha=0.2)
		plt.axvline(x=rfecv_train.n_features_, color = "black")
		plt.legend(["StratifiedKFold " + str(i+1) for i in range(0,CV)], loc='upper left', bbox_to_anchor=(1, 0.5))
		plt.savefig(titlefig, format="svg")
		plt.title(str(title))
		plt.show()		


	return acc, acc_rfe, rfecv_train.cv_results_, rfecv_train.get_support(indices = True), rfecv_train.n_features_
			

def create_SVM(CV, C, KERNEL, DEGREE, LOBE, N_LOBES, f, p ,r, rf , SEED, min_features_to_select): 

	from sklearn import svm
	import matplotlib.pyplot as plt
	
	np.random.seed(SEED)
	
	X_train, X_test, y_train, y_test, class_weights, NC = def_sample(f, p, r, LOBE, N_LOBES, moy = True)
	cweights = {0:class_weights[0], 1:class_weights[1]}
	y_train = y_train.reshape(len(y_train),) ; y_test = y_test.reshape(len(y_test),)


	print(" - - - - Model fitting and predicting - - - -")
	clf_SVC = svm.SVC(kernel = KERNEL, C = C, degree = DEGREE, class_weight = cweights, random_state = SEED)
	svm = clf_SVC.fit(X_train,y_train)
	y_pred = clf_SVC.predict(X_test) ; print("Y_test : ", y_test) ; print("Y_pred : ", y_pred)
	acc = metrics.accuracy_score(y_test, y_pred)
	cm = confusion_matrix(y_test, y_pred)
	print("Accuracy without RFE : ", acc) ; print("Confusion Matrix :\n", cm)
	
	if rf == True:
		print(" - - - - RFE fitting - - - -")
		cv = StratifiedKFold(CV, shuffle = True, random_state = SEED)
		# min_features_to_select = 50 # Minimum number of features to consider len(X_train[1]) / 2
		
		rfecv = RFECV(
			estimator = clf_SVC,
			step = 1,
			cv = cv,
			scoring = "accuracy",
			min_features_to_select=min_features_to_select,
			verbose = 0
		)
		
		rfecv_train = rfecv.fit(X_train, y_train)
		print("Optimal number of features : %d" % rfecv_train.n_features_)
		# rfe = RFE(estimator=clf_SVC, n_features_to_select = rfecv_train.n_features_, step=1)
		# rfecv = rfecv.transform(X_test)
		rfe_pred = rfecv.predict(X_test)
		# rfe_train = RF_tree_featuresTrain.loc[:, rfecv.get_support()]
		# rfe_test = RF_tree_featuresTest.loc[:, rfecv.get_support()]
	
	
		print("Y_test : ", y_test) ; print("Y_pred : ", rfe_pred)
		acc_rfe = metrics.accuracy_score(y_test, rfe_pred)
		cm = confusion_matrix(y_test, rfe_pred)
		print("Accuracy with RFE : ", acc_rfe) ; print("Confusion Matrix :\n", cm)
		# print("/!\   ", rfecv_train.get_support(indices = True))
		# print("/!\   ", rfecv_train.cv_results_)

		
		title = "Accuracy for Support Vector Machine (Part: " +str(p) + ", Freq: "+ str(f[0])+", Nb Lobe(s): " + str(LOBE) + ")"
		titlefig = "SVM_p" + str(p) + "_f"+ str(f[0]) + "_nbL" + str(LOBE) + ".svg"
		# fig = plt.figure(plt.figure(figsize=(10, 5)))
		plt.xlabel("Number of features selected")
		plt.ylabel("Cross validation score (accuracy)")
		plt.plot(
			range(min_features_to_select, len(rfecv_train.grid_scores_) + min_features_to_select),
		    rfecv_train.grid_scores_, linestyle = "dashed")
		plt.plot(
		    range(min_features_to_select, len(rfecv_train.cv_results_['mean_test_score']) + min_features_to_select),
		    rfecv_train.cv_results_['mean_test_score'], color = "black")
		plt.fill_between(range(min_features_to_select, len(rfecv_train.cv_results_['mean_test_score']) + min_features_to_select), 
				   rfecv_train.cv_results_['mean_test_score'] - rfecv_train.cv_results_['std_test_score'], 
				   rfecv_train.cv_results_['mean_test_score'] + rfecv_train.cv_results_['std_test_score'], color= 'black', alpha=0.2)
		plt.axvline(x=rfecv_train.n_features_, color = "black")
		plt.legend(["StratifiedKFold " + str(i+1) for i in range(0,CV)], loc='upper left', bbox_to_anchor=(1, 0.5))
		plt.savefig(titlefig, format="svg")
		plt.title(str(title))
		plt.show()		


	return acc, acc_rfe, rfecv_train.cv_results_, rfecv_train.get_support(indices = True), rfecv_train.n_features_

def plot_global_acc(list_rfecv, f, p, N_LOBES, min_features_to_select, nfeature, col):
	'''Parameters
	----------
	list_rfecv : result for each fold.
	f : frequency.
	p : subject.
	N_LOBES : numberr lobe.
	min_features_to_select : in the name.
	nfeature : optimal number of feature.
	col : list color.
	
	Returns
	-------
	plot.

	'''
	title = "Accuracy for Support Vector Machine (Part: " +str(p) + ", Freq: "+ str(f[0])+", Lobe(s): " + str(LOBE) + ")"
	titlefig = "SVM_p" + str(p) + "_f"+ str(f[0]) + "_nbL" + str(LOBE) + ".svg" 
	
	j = 0
	alpha = 0.05
	
	for rfecv_train in list_rfecv: 
		leg  = "StratifiedKFold, Subject " + str(p[j])
		plt.plot(
		    range(min_features_to_select, len(rfecv_train['mean_test_score']) + min_features_to_select),
		    rfecv_train['mean_test_score'], color = col[j])
		plt.fill_between(range(min_features_to_select, len(rfecv_train['mean_test_score']) + min_features_to_select), 
				   rfecv_train['mean_test_score'] - rfecv_train['std_test_score'], 
				   rfecv_train['mean_test_score'] + rfecv_train['std_test_score'], color= col[j], alpha=alpha)
		plt.axvline(x=nfeature[j], color = col[j])
		j = j + 1
		alpha = alpha + 0.1
	
	plt.legend(["StratifiedKFold " + str(p[i]) for i in range(0,len(p))], loc='upper left', bbox_to_anchor=(1, 0.5))
	plt.savefig(titlefig, format="svg")
	plt.title(str(title))
	plt.show()		
	
def make_bestParams(dict_params_tree, dict_params_svm, CV, SEED, LOBE, N_LOBES, f, p, r):
	'''Parameters
	----------
	dict_params_tree : hyperparameters decision tree.
	dict_params_svm : hyperparameters svm.

	Returns
	-------
	Best parameters.'''
	
	from sklearn.model_selection import GridSearchCV
	from sklearn import svm

	np.random.seed(SEED)
	
	X_train, X_test, y_train, y_test, class_weights, NC = def_sample(f, p, r, LOBE, N_LOBES, moy = True)
	cweights = {0:class_weights[0], 1:class_weights[1]}
	y_train = y_train.reshape(len(y_train),) ; y_test = y_test.reshape(len(y_test),)
	
	cv = StratifiedKFold(CV, shuffle = True, random_state = SEED)

	# Decision Tree
	dict_params_tree['class_weight'] = [cweights]
	clf_tree = GridSearchCV(DecisionTreeClassifier(), dict_params_tree, cv=cv, 
						 scoring = "accuracy")
	clf_tree.fit(X_train, y_train)
	best_params_tree = clf_tree.best_params_
	# print(clf_tree.get_params().keys())
	
	
	# SVM
	dict_params_svm['class_weight'] = [cweights]
	clf_svm = GridSearchCV(svm.SVC(), dict_params_svm, cv=cv, 
						scoring = "accuracy")
	clf_svm.fit(X_train, y_train)
	best_params_svm = clf_svm.best_params_
	
	return best_params_tree, best_params_svm


	
def make_dict_lobe(dict_SVM, dict_tree):

	for key in dict_SVM.keys():
		dict_SVM[key] = []
		for key in dict_tree.keys():
			dict_tree[key] = []
	
	return dict_SVM, dict_tree


	
def chann_intersect(list_idx):
	f = []
	for i in range(0, len(list_idx) - 1, 2):
		f = f + list(list_idx[i]) + list(list_idx[i+1])
	return f

if __name__ == "__main__":

	# Runnning plan for lobes
	N_LOBES = 4
	LOBE_4 = [["LFidx", "LPidx", "LTDidx", "LTGidx"]]
	
	N_LOBES = 3 
	LOBE_3 = [["LFidx", "LPidx", "LTDidx"], ["LFidx", "LPidx", "LTGidx"], 
			["LFidx", "LTDidx", "LTGidx"], ["LPidx", "LTGidx", "LTDidx"]]
	
	N_LOBES = 2 
	LOBE_2 = [["LFidx", "LPidx"], ["LFidx", "LTGidx"], ["LFidx", "LTDidx"],
			["LPidx", "LTDidx"], ["LPidx", "LTGidx"], ["LTGidx", "LTDidx"]]

	N_LOBES = 1 
	LOBE_1 = ["LFidx", "LTGidx", "LTDidx", "LPidx", "LOidx"]
	
	dict_SVM = {
		'list_rfecv_SVM' : [],
		'list_idx_SVM' : [],
		'features_SVM' : [],
		'l_acc_rfe_SVM' : [],
		'l_acc_SVM' : []
		}
	
		
	dict_tree = {
		'list_rfecv_tree' : [],
		'list_idx_tree' : [],
		'features_tree' : [],
		'l_acc_rfe_tree' : [],
		'l_acc_tree' : []
		}	
	
	params_SVC = {
		'C' : [10, 50, 100, 500], 
		'kernel' : ['linear'],
		"degree" : [2, 3, 4, 5],
		'random_state' : [4242]
		}
	
	params_DT = {
		'max_depth' : [5, 10, 15, 20, 30],
		'min_samples_leaf' : [5, 10, 20, 30, 40, 50, 60],
		'random_state' : [4242]
		}  
	
	# Global parameters
	SEED = 4242
	CV = 5
	rf = True
	
	MFS = int(15 * N_LOBES) # Min_feature_to_select
	f = [2, 2, 2, 2]
	p = 1 	# part = [0, 1, 7, 21, 23]
	r = 2
	
	df = pd.DataFrame(columns=['part', 'lobes', 'freq', 'acc_tree', 'acc_svm', 
						 'acc_rfe_tree', 'acc_rfe_svm', 'features_svm', 'features_tree', ])
	 
	list_idx_SVM = []
	list_idx_tree = []
	
	listfreq = [[0, 0], [1,1], [1,1], [1,1], [2,2], [3,3], [4,4], [6,6], [7,7]] # [0, 4, 9, 13, 30, 41, 61, 81, 100]
	
	LOBE_2 = [
			["LTGidx", "LPidx"], # delta
			["LFidx", "LTDidx"], ["LFidx", "LPidx"],["LPidx", "LTDidx"], # theta 
			["LTGidx", "LTDidx"], # alpha
			["LPidx", "LTDidx"], # beta1
			['LFidx', 'LPidx'], #beta2
			["LFidx", "LTDidx"], # g1
			["LPidx", "LTDidx"] # g2
			]
	
	# Pour les participants sélectionnnés
	#for f in listfreq: 
	i = 0
	for LOBE in LOBE_2:
		
		N_LOBES = 2
		f = listfreq[i]
		
		MFS = int(20 * N_LOBES)
		best_params_tree, best_params_svm =  make_bestParams(params_DT, params_SVC, CV, SEED, LOBE, N_LOBES, f, p, r)
		
		# Tree parameters
		MAX_DEPTH = best_params_tree['max_depth']
		MIN_SAMPLES = best_params_tree['min_samples_leaf'] 
		
		# SVM parameters
		C = best_params_svm['C']
		KERNEL = best_params_svm['kernel']
		DEGREE = best_params_svm['degree']
		
		acc_SVM, acc_rfe_SVM, rfecv_train_SVM, idx_SVM, nfeature_SVM = create_SVM(CV, C, KERNEL, DEGREE, LOBE, N_LOBES, f, p ,r, rf , SEED, MFS)
		acc_tree, acc_rfe_tree, rfecv_train_tree, idx_tree, nfeature_tree = create_DecisionTree(MAX_DEPTH, MIN_SAMPLES, LOBE, N_LOBES, f, p ,r, SEED, MFS)
		
		row_to_add = {'part':p, 'lobes': LOBE, 'freq': f[0],  
					'acc_tree': acc_tree, 'acc_svm' : acc_SVM,
					'acc_rfe_tree': acc_rfe_tree, 'acc_rfe_svm':acc_rfe_SVM,
					'features_svm' : nfeature_SVM, 'features_tree' : nfeature_tree}
		
		df = df.append(row_to_add, ignore_index=True)
		list_idx_SVM.append(list(idx_SVM))
		list_idx_tree.append(list(idx_tree))
		
		i = i+1

	
	joblib.dump(df, 'df_lobes2_allfreq.pkl')
	joblib.dump(list_idx_SVM, 'idx_SVM_lobes2_allfreq.pkl')
	joblib.dump(list_idx_tree, 'idx_tree_lobes2_allfreq.pkl')
	
	
	'''
	joblib.dump(df, 'df_lobe4_Groove.pkl')
	joblib.dump(list_idx_SVM, 'idx_SVM_lobes4_Groove.pkl')
	joblib.dump(list_idx_tree, 'idx_tree_lobes4_Groove.pkl')

	# df_4 = joblib.load('df_lobes4.pkl')
	joblib.dump(df, 'df_lobes2.pkl')
	joblib.dump(list_idx_SVM, 'idx_SVM_lobes2.pkl')
	joblib.dump(list_idx_tree, 'idx_tree_lobes2.pkl')

	df_4 = joblib.load('df_lobes4.pkl')
	
	joblib.dump(df, 'df_lobes2_delta.pkl')
	joblib.dump(list_idx_SVM, 'idx_SVM_lobes2_delta.pkl')
	joblib.dump(list_idx_tree, 'idx_tree_lobes2_delta.pkl')
	
	
	joblib.dump(df, 'df_lobes3_delta.pkl')
	joblib.dump(list_idx_SVM, 'idx_SVM_lobes3_delta.pkl')
	joblib.dump(list_idx_tree, 'idx_tree_lobes3_delta.pkl')
	

	joblib.dump(df, 'df_lobes4_delta.pkl')
	joblib.dump(list_idx_SVM, 'idx_SVM_lobes4_delta.pkl')
	joblib.dump(list_idx_tree, 'idx_tree_lobes4_delta.pkl')
	
	# a = joblib.load('global_results_SVM_tree.pkl')
	
			

	# Global parameters
	SEED = 4242
	
	LOBE = ["LFidx", "LPidx", "LTDidx"]
	LOBE = ["LFidx", "LPidx", "LTGidx"]
	LOBE = ["LFidx", "LTDidx", "LTGidx"]
	LOBE = ["LPidx", "LTGidx", "LTDidx"]
	
	LOBE = ["LFidx", "LPidx"]
	LOBE = ["LFidx", "LTGidx"]
	LOBE = ["LFidx", "LTDidx"]
	LOBE = ["LPidx", "LTDidx"]
	LOBE = ["LPidx", "LTDidx"]
	LOBE = ["LTGidx", "LTDidx"]
	
	MFS = 10 # Min_feature_to_select
	# LOBE = "LFidx"
	N_LOBES = 2
	f = [2, 2, 2, 2]
	part = [0, 1, 7, 21, 23]
	p = 0
	r = 0
	CV = 5
	
	params_SVC = {
		'C' : [10, 50, 100, 500], 
		'kernel' : ['linear'],
		"degree" : [2, 3, 4, 5],
		'random_state' : [4242]
		}
	
	params_DT = {
		'max_depth' : [5, 10, 15, 20],
		'min_samples_leaf' : [10, 20, 30, 40, 50],
		'random_state' : [4242]
		}  
	
	best_params_tree, best_params_svm =  make_bestParams(params_DT, params_SVC, CV, SEED, LOBE, N_LOBES, f, p, r)
	
	# Tree parameters
	MAX_DEPTH = best_params_tree['max_depth']
	MIN_SAMPLES = best_params_tree['min_samples_leaf'] 
	
	# SVM parameters
	C = best_params_svm['C']
	KERNEL = best_params_svm['kernel']
	DEGREE = best_params_svm['degree']
	CV = 5
	rf = True
	
	# red = Color("red")
	# colors_list = list(red.range_to(Color("blue"), 7))
	colors_list = ['cyan', 'red', 'blue', 'orange', 'purple']
	
	# Run decision tree
	# acc_tree = create_DecisionTree(MAX_DEPTH, MIN_SAMPLES, LOBE, N_LOBES, f, p ,r)
	# print("Tree Accuracy: ", acc_tree)
	
	list_rfecv = []
	list_idx = []
	features = []
	
	for p in part:
		acc, acc_rfe, rfecv_train, idx, nfeature = create_SVM(CV, C, KERNEL, DEGREE, LOBE, N_LOBES, f, p ,r, rf , SEED, MFS)
		list_rfecv.append(rfecv_train)
		list_idx.append(idx)
		features.append(nfeature)
	
	p = [0, 0, 1, 1, 7, 7, 21, 21, 23, 23]
	MFS = 10
	plot_global_acc(list_rfecv, f, p, N_LOBES, MFS, features, colors_list)
	# print("SVM Accuracy: ", acc_SVM)
	# f = chann_intersect(list_idx)
	# f = np.unique(f)
	'''
	
	
	
	