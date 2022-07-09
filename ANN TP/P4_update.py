# coding: utf8
# !/usr/bin/env python
# ------------------------------------------------------------------------
# Perceptron en pytorch (en utilisant juste les tenseurs)
# Écrit par Mathieu Lefort
#
# Distribué sous licence BSD.
# ------------------------------------------------------------------------

# https://towardsdatascience.com/multi-layer-neural-networks-with-sigmoid-function-deep-learning-for-rookies-2-bf464f09eb7f

import gzip, numpy as np, torch
from numpy.core.numeric import outer
import matplotlib.pyplot as plt
import pandas as pd


####################################################################
# Modeles
####################################################################

###########
### MLP ###
###########
class MLP(torch.nn.Module):

	def __init__(self, input_dim, output_dim, hidden_1, hidden_2):
		self.hidden_1 = hidden_1
		self.hidden_2 = hidden_2
		super(MLP, self).__init__()

		self.hidden1 = torch.nn.Linear(input_dim, hidden_1)
		self.hidden2 = torch.nn.Linear(hidden_1, hidden_2)
		self.output = torch.nn.Linear(hidden_2, output_dim)
		self.sigmoid = torch.nn.Sigmoid()


	def forward(self, x):
		#check this site
		#https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/forwardpropagation_backpropagation_gradientdescent/
		x = self.hidden1(x)
		x = self.sigmoid(x)
		x = self.hidden2(x)
		x = self.sigmoid(x)
		x = self.output(x)
		return x

	# https://blog.paperspace.com/pytorch-101-advanced/
	def init_weights(self):
		for module in self.modules():
			if isinstance(module, torch.nn.Linear):
				torch.nn.init.normal_(module.weight, mean=0, std=1)


####################################################################
# Train function
####################################################################
def train_step(model, loss_func, optimizer):

	# on initiliase la fonction de coût (l) et l'optimiseur (o)
	def train_step1(x,t):
		output = model.forward(x) # Forward pass
		loss = loss_func(t,output) # MàJ des poids
		loss.backward() # Propagation
		optimizer.step() # clear the gradients of all optimized variables
		optimizer.zero_grad()

		return loss.item()

	return train_step1	


####################################################################
#  Visualization Comparaison function
####################################################################
def plot_total_accuracy(it, accuracy_list, len, col):
	i = 0
	for acc in accuracy_list:
		names = 'Model ' + str(i)
		plt.plot(it, acc, color= col[i], label= names)
		i = i+1
	plt.axis([0, (len - 1), 0, 1])
	plt.xlabel('Number of Epochs')
	plt.ylabel('Accuracy')
	plt.title('Total Accuracy')
	plt.legend()
	plt.show()


def plot_total_loss(it, losses_test, losses_train, len, col):

	i = 0
	for ltrain in losses_train:
		names = 'Model_train ' + str(i)
		plt.plot(it, ltrain, color= col[i], label= names,  linestyle='dashed')
		i = i + 1

	j = 0 	
	for ltest in losses_test:
		names = 'Model_test ' + str(j)
		plt.plot(it, ltest, color= col[j], label= names)
		j = j + 1

	plt.axis([0,( len - 1), 0, 15])
	plt.xlabel('Number of Epochs')
	plt.ylabel('Losses')
	plt.title('Loss Function')
	plt.legend()
	plt.show()



####################################################################
# Main
####################################################################

if __name__ == '__main__':

####################################################################
# Dictionnaires d'hyperparamètres 
# TODO : GridSearchCV
####################################################################
	plot = {
		'colors' : ['red', 'green', 'purple', 'blue', 'orange', 'pink', 'black', 'brown', 'yellow', 'gray']
	}

	params_1 = {
	'batch_size' : 5,
	'nb_epochs' : 10,
	'eta' : 0.00001,
	'hidden_1' : 128,
	'hidden_2' : 128,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}

	params_2 = {
	'batch_size' : 5,
	'nb_epochs' : 10,
	'eta' : 0.0001,
	'hidden_1' : 128,
	'hidden_2' : 128,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}

	params_3 = {
	'batch_size' : 5,
	'nb_epochs' : 10,
	'eta' : 0.001,
	'hidden_1' : 128,
	'hidden_2' : 128,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}

	params_4 = {
	'batch_size' : 5,
	'nb_epochs' : 10,
	'eta' : 0.01,
	'hidden_1' : 128,
	'hidden_2' : 128,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}


	list_tt = [params_1, params_2, params_3, params_4]


	# list_tt = [params_1, params_2, params_5, params_6]
	# list_tt = [params_1, params_2]
	# list_tt = [params_1]

# On initialise une liste vide pour y stocker l'évolution de l'apprentissage
	
	visu_losses_train_tt = []
	visu_losses_test_tt = []
	visu_acc_tt = []
	visu_iteration = []

	for element in list_tt:

		visu_acc = []
		visu_losses_train = []
		visu_losses_test = []
		train_losses = 0.
		test_losses = 0.

		# on met le bon dictionnaire 
		params = element

####### Chargement des données
		((data_train,label_train),(data_test,label_test)) = torch.load(gzip.open('mnist.pkl.gz'))
		train_dataset = torch.utils.data.TensorDataset(data_train,label_train)
		test_dataset = torch.utils.data.TensorDataset(data_test,label_test)

		# On créait les lecteurs de données
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

		# Création du modèle en faisant appel à MLP
		model = MLP(data_train.shape[1], label_train.shape[1], params['hidden_1'], params['hidden_2'])

		# Initialize Weight
		model.init_weights()# Prefer Xavier initialization to Sigmoid and Tanh

		# TODO : Optimizer
		optim =  torch.optim.SGD(model.parameters(), lr=params['eta'])

		# Appel de la fonction train 
		train_step_res = train_step(model, params['loss_func'], optim)

		#########################################################################################################

####### For each epoch...
		for n in range(params['nb_epochs']):
			
			# loss = 0.
########### TRAIN ########### 
			for x,t in train_loader:
				loss = train_step_res(x, t)
				train_losses = train_losses + loss

			# Evite de calculer des gradients inutiles
			# Ne marche pas
			# with torch.no_grad:
########### TEST ###########
			acc = 0.
			# loss = 0.
			for x,t in test_loader:
				#model.eval() # Eval définit le model en mode évaluation 
				output = model(x)
				loss = params['loss_func'](output,t)
				#print(loss)
				test_losses = test_losses + loss.item()
				acc += torch.argmax(output,1) == torch.argmax(t,1) # On regarde si la sortie est correcte

	# On stocke les données pour visualisation
			print('Accuracy', acc)
			visu_acc.append((acc/data_test.shape[0]).item())
			visu_losses_test.append((test_losses/data_test.shape[0]))
			visu_losses_train.append((train_losses/data_train.shape[0]))
			visu_iteration.append(n)

			
		visu_losses_train_tt.append(list(visu_losses_train))
		visu_losses_test_tt.append(list(visu_losses_test))
		visu_acc_tt.append(list(visu_acc))

			# Hyperparameters
		print("Model's state_dict:")
		for param_tensor in model.state_dict():
			print(param_tensor, "\t", model.state_dict()[param_tensor].size())

		
	acc = list(visu_acc_tt)
	it = list(set(visu_iteration))

	# Visualization	
	plot_total_loss(it, visu_losses_test_tt, visu_losses_train_tt, len(it), plot['colors'])
	plot_total_accuracy(it, acc, len(it), plot['colors'])
	

	# Debug
	print("it = ", it)
	print("train_losses LEN = ", len(train_losses))
	print("test_losses LEN = ", len(test_losses))
	print("train_losses = ", train_losses)
	print("test_losses = ", test_losses)

	# TODO : stocker les données avec pickle 










	


	