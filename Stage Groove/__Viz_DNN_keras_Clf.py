import matplotlib.pyplot as plt
import numpy as np
from colour import Color

############  Visualization Comparaison functions ############

def plot_DNN_clf(dm, reg, col, p, f):
	fig = plt.figure(figsize=(10, 5))
	title = "Accuracy for each Subject (" + str(reg) + ")"
	l = list(range(0, p))
	
	for f in range(0, f):
		plt.plot(l, dm['acc'][f][:], color = col[f].hex)
	plt.xticks(np.arange(0,30, step = 1))
	plt.axis([0, p, 0, 0.8])
	plt.xlabel("Subjects" )
	plt.ylabel('Accuracy')
	plt.title(str(title), fontsize=10)
	plt.grid(linestyle = '--')
	plt.legend()
	plt.show()


def call_plot_errxfreq(dcm, dcsd, fd, ff):
	title = "Error (MSE, MAE) for each Frequency" 
	mean_mse = np.array(dcm['mse'][0][fd:ff])
	mean_mae = np.array(dcm['mae'][0][fd:ff])
	mean_mae_val = np.array(dcm['val_mae'][0][fd:ff])
	std = np.array(dcsd['val_mae_sd'][0][fd:ff] [fd:ff])
	l = list(range(fd, ff))
	
	plt.plot(l, mean_mse, '--', color = 'green', label = 'Mean (MSE) (' + str(np.round(np.mean(mean_mse), 2)) + ')')
	plt.plot(l, mean_mae, '--', color = 'red', label = 'Mean (MAE) (' + str(np.round(np.mean(mean_mae), 2)) + ')')
	plt.plot(l, mean_mae_val, color = 'red', label = 'Mean Validation (MAE) (' + str(np.round(np.mean(mean_mae_val), 2)) + ')')
	plt.fill_between(l, mean_mae_val - std, mean_mae_val + std, color= 'red', alpha=0.2)
	plt.axis([fd, ff, 0, 2])
	plt.xlabel("Frenquencies" )
	plt.ylabel('MSE, MAE')
	plt.title(str(title), fontsize=10)
	plt.grid(linestyle = '--')
	plt.legend()


