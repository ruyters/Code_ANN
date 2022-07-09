# -*- coding: utf-8 -*-
import shap
import numpy as np
import pandas as pd

def call_shapley(model, data):
	keras_explainer = shap.KernelExplainer(model, data)
	keras_shap_values = keras_explainer.shap_values(data)
	values = keras_shap_values[0]
	base_values = [keras_explainer.expected_value[0]]*len(keras_shap_values[0])
	explanation = shap.Explanation(values = np.array(values, dtype=np.float32),
				base_values = np.array(base_values, dtype=np.float32),
				data = np.array(data))
	
	# Plot waterfall
	# shap.waterfall_plot(shap_values = explanation[0])
	# shap.plots.beeswarm(explanation)
	
	return explanation
	
	# See https://shap-lrjball.readthedocs.io/en/latest/examples.html#plots


def call_shapley_featureImportance(model, data, nc):
	
	keras_explainer = shap.KernelExplainer(model, data)
	keras_shap_values = keras_explainer.shap_values(data)
	
	values = np.abs(keras_shap_values[0]).mean(0)
	feature_idx = np.array([str(i+1) for i in range(nc)])

	feature_importance = pd.DataFrame(list(zip(feature_idx, values)),columns=['col_name','feature_importance_vals'])
	feature_importance.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)

	return feature_importance	


def merge_shapley_df(fi, fi_add):
	fif = fi.append(fi_add)
	return fif