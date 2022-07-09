# -*- coding: utf-8 -*-

import plotly.graph_objects as go
import numpy as np
import joblib

def plot_acc(x_done, df, p, r, f, title):
	# Si 1 lobe
	x1 = ['Frontal Lobe', 'Temporal Left Lobe', 'Temporal Right Lobe', 'Parietal Lobe', 'Occipital Lobe']
	
	# Si 2 Lobes 
	x2 = ['Frontal - Parietal', 
		 'Frontal - Temporal Left', 
		 'Frontal - Temporal Right', 
		 'Parietal - Temporal Right', 
		 'Parietal - Temporal Left',
		 'Temporal Left - Temporal Right'
		 ]
	
	# Si 3 Lobes 
	x3 = ['Frontal - Parietal - Temporal Left', 
		 'Frontal - Parietal - Temporal Right',
		 'Frontal- Temporal Left/Right',
		 'Parietal - Temporal Right/Left']
	
	# Si 4 Lobes
	x4 = ['Frontal - Parietal - Temporal Left/Right']
	
	if x_done == "x1": 
		x = x1 
	if x_done == "x2": 
		x = x2
	if x_done == "x3": 
		x = x3
	if x_done == "x4": 
		x = x4
		
		
		
	# On convertit la colonne en string
	df['lobes'] = df['lobes'].astype(str)
	
	
	y1 = df['acc_svm'].loc[np.where(df['part'] == p)]
	y2 = df['acc_tree'].loc[np.where(df['part'] == p)]
	y3 = df['acc_rfe_svm'].loc[np.where(df['part'] == p)]
	y4 = df['acc_rfe_tree'].loc[np.where(df['part'] == p)]
	
	
	fig = go.Figure()
	
	fig.add_trace(go.Bar(
	    x=x,
	    y=y1.tolist(),
	    name='Accuracy SVM without RFE',
	    marker_color='indianred'
	))
	
	fig.add_trace(go.Bar(
	    x=x,
	    y=y3.tolist(),
	    name='Accuracy SVM with RFE',
	    marker_color='lightsalmon'
	))
	
	fig.add_trace(go.Bar(
	    x=x,
	    y=y2.tolist(),
	    name='Accuracy Decision Tree without RFE',
	    marker_color='steelblue'
	))
	
	
	fig.add_trace(go.Bar(
	    x=x,
	    y=y4.tolist(),
	    name='Accuracy Decision Tree with RFE',
	    marker_color='paleturquoise'
	))
	
	fig.update_layout(
	    title=title,
	    yaxis_range=[0,0.9])
	
	fig.add_hline(y=0.5, line_dash="dash") 
	
	titlesvg = "BarPlot_Groove_nLobes_" + str(x_done) + "_Reg" + str(r) + "_p" + str(p) + "_f" + str(f) + ".svg"
	fig.write_image(str(titlesvg))
	fig.show()
	
def plot_acc_freq(x_done, df, p, r, f, title):
	# Si 1 lobe
	x1 = ['Frontal Lobe', 'Temporal Left Lobe', 'Temporal Right Lobe', 'Parietal Lobe', 'Occipital Lobe']
	
	# Si 2 Lobes 
	if f == 0: 
		x2 = ['Temporal Left - Parietal']
	if f == 1:
		x2 = ['Frontal - Temporal Right', 
		 'Frontal - Parietal', 
		 'Parietal - Temporal Right']
	if f == 2:
		x2 = ['Temporal Right/Left']
	if f == 3:
		x2 = ['Parietal - Temporal Right']
	if f == 4:
		x2 = ['Frontal - Pareital']
	if f == 6:
		x2 = ['Frontal - Temporal Right']
	if f == 7:
		 x2 = ['Parietal - Tempora right ']
	
	# Si 3 Lobes 
	x3 = ['Frontal - Parietal - Temporal Left', 
		 'Frontal - Parietal - Temporal Right',
		 'Frontal- Temporal Left/Right',
		 'Parietal - Temporal Right/Left']
	
	# Si 4 Lobes
	x4 = ['Frontal - Parietal - Temporal Left/Right']
	
	if x_done == "x1": 
		x = x1 
	if x_done == "x2": 
		x = x2
	if x_done == "x3": 
		x = x3
	if x_done == "x4": 
		x = x4
		
		
		
	# On convertit la colonne en string
	df['lobes'] = df['lobes'].astype(str)
	

	y1 = df['acc_svm'].loc[np.where(df['freq'] == f)]
	y2 = df['acc_tree'].loc[np.where(df['freq'] == f)]
	y3 = df['acc_rfe_svm'].loc[np.where(df['freq'] == f)]
	y4 = df['acc_rfe_tree'].loc[np.where(df['freq'] == f)]

	fig = go.Figure()
	
	fig.add_trace(go.Bar(
	    x=x,
	    y=y1.tolist(),
	    name='Accuracy SVM without RFE',
	    marker_color='indianred'
	))
	
	fig.add_trace(go.Bar(
	    x=x,
	    y=y3.tolist(),
	    name='Accuracy SVM with RFE',
	    marker_color='lightsalmon'
	))
	
	fig.add_trace(go.Bar(
	    x=x,
	    y=y2.tolist(),
	    name='Accuracy Decision Tree without RFE',
	    marker_color='steelblue'
	))
	
	
	fig.add_trace(go.Bar(
	    x=x,
	    y=y4.tolist(),
	    name='Accuracy Decision Tree with RFE',
	    marker_color='paleturquoise'
	))
	
	fig.update_layout(
	    title=title,
	    yaxis_range=[0,0.9])
	
	fig.add_hline(y=0.5, line_dash="dash") 
	
	titlesvg = "BarPlot_Groove_nLobes_" + str(x_done) + "_Reg" + str(r) + "_p" + str(p) + "_f" + str(f) + ".svg"
	fig.write_image(str(titlesvg))
	fig.show()
	


if __name__ == '__main__':

	freq = [0, 1, 1, 1, 2, 3, 4, 6, 7]
	p = 1
	df_1 = joblib.load('df_lobes2_allfreq.pkl')
	# df_2 = joblib.load('SAVED/df_lobe2_Groove.pkl')
	# df_3 = joblib.load('SAVED/df_lobe3_Groove.pkl')
	# df_4 = joblib.load('SAVED/df_lobe4_Groove.pkl')
	
	
	for f in freq:
		title = 'Accuracy Models - Lobe for Groove\n (f :' + str(f)  + '_P:' + str(p) + ')'
		plot_acc_freq('x2', df_1, p, 2, f, str(title))
		
	'''
	title = 'Accuracy Models - Combinaison of 2 Lobes for Groove\n (Freq :' + str(f) +  '_P:' + str(p) + ')'
	plot_acc('x2', df_2, p, 2, f, str(title))
	
	title = 'Accuracy Models - Combinaison of 3 Lobes for Groove\n (Freq :' + str(f) +  '_P:' + str(p) + ')'
	plot_acc('x3', df_3, p, 2, f, str(title))
	
	title = 'Accuracy Models - 4 Lobes for Groove\n (Freq :' + str(f) +  '_P:' + str(p) + ')'
	plot_acc('x4', df_4, p, 2, f, str(title))
	'''
	

	
	
	
	
	
	
	