#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Apr 11 14:53:01 2022

@author: Swann
"""


import numpy as np 

def add_dict_clf(dict_res, h, acc):
	# print(h.history.keys())
	dict_res['loss'].append(np.mean(h.history['loss']))
	dict_res['mae_labels'].append(np.mean(h.history['mean_absolute_error_labels']))
	dict_res['val_mae_labels'].append(np.nanmean(h.history['val_mean_absolute_error_labels']))
	dict_res['val_mae_labels_sd'].append(np.nanstd(h.history['val_mean_absolute_error_labels']))
	dict_res['val_loss'].append(np.mean(h.history['val_loss']))
	dict_res['acc'].append(acc)
	print(dict_res['acc'])
	return dict_res

def add_final_dict_clf(dict_res, df):
	
	loss = dict_res['loss']
	mean_absolute_error = dict_res['mae_labels']
	val_mean_absolute_error = dict_res['val_mae_labels']
	val_mae_labels_sd = dict_res['val_mae_labels_sd']
	val_loss = dict_res['val_loss']
	acc = dict_res['acc']
	
	df['loss'].append(list(loss))
	df['mae_labels'].append(list(mean_absolute_error))
	df['val_mae_labels'].append(list(val_mean_absolute_error))
	df['val_mae_labels_sd'].append(list(val_mae_labels_sd))
	df['val_loss'].append(list(val_loss))
	df['acc'].append(list(acc))
	
	return df

def add_dict_reg(dict_res, h, pearsonr):
	# print(h.history.keys())
	dict_res['loss'].append(np.mean(h.history['loss']))
	dict_res['mse'].append(np.mean(h.history['mse']))
	dict_res['mae'].append(np.mean(h.history['mae']))
	dict_res['val_loss'].append(np.mean(h.history['val_loss']))
	dict_res['val_loss_sd'].append(np.std(h.history['val_loss']))
	dict_res['val_mse'].append(np.mean(h.history['val_mse']))
	dict_res['val_mse_sd'].append(np.std(h.history['val_mse']))
	dict_res['val_mae'].append(np.mean(h.history['val_mae']))
	dict_res['pearsonr'].append(pearsonr)
	
	return dict_res

def add_final_dict_reg(dict_res, df):
	
	loss = dict_res['loss']
	mean_squared_error = dict_res['mse']
	mean_absolute_error = dict_res['mae']
	val_loss = dict_res['val_loss']
	val_loss_sd = dict_res['val_loss_sd']
	val_mean_squared_error = dict_res['val_mse']
	val_mean_absolute_error = dict_res['val_mae']
	val_sd_absolute_error = dict_res['val_mse_sd']
	pearsonr = dict_res['pearsonr']
	
	df['loss'].append(list(loss))
	df['mse'].append(list(mean_squared_error))
	df['mae'].append(list(mean_absolute_error))
	df['val_loss'].append(list(val_loss))
	df['val_loss_sd'].append(list(val_loss_sd))
	df['val_mse'].append(list(val_mean_squared_error))
	df['val_mse_sd'].append(list(val_sd_absolute_error))
	df['val_mae'].append(list(val_mean_absolute_error))
	df['pearsonr'].append(list(pearsonr))
	
	return df