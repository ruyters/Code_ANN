# coding: utf8
# !/usr/bin/env python
# ------------------------------------------------------------------------
# Perceptron en pytorch (en utilisant juste les tenseurs)
# Écrit par Mathieu Lefort
#
# Distribué sous licence BSD.
# ------------------------------------------------------------------------


import gzip, numpy, torch
from numpy.core.numeric import outer
import matplotlib.pyplot as plt


####################################################################
# Modele MLP
####################################################################

class MLP(torch.nn.Module):
	# TODO 
	hidden_layer_dim = [64, 128, 256]

	def __init__(self, input_dim, output_dim, i):
		super().__init__()

		self.hidden = torch.nn.Linear(input_dim, self.hidden_layer_dim[i])
		self.output = torch.nn.Linear(self.hidden_layer_dim[i], output_dim)
		self.sigmoid = torch.nn.Sigmoid()

	
	def forward(self, x):
		x = self.hidden(x)
		x = self.sigmoid(x)
		x = self.output(x)
		return x

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
# Visualization Comparaison function
####################################################################
def plot_total_accuracy(it, accuracy_list, len, col):
	i = 0
	for acc in accuracy_list:
		names = 'Model ' + str(i)
		plt.plot(it, acc, color= col[i], label= names)
		i = i+1
	plt.axis([0, (len - 1), 0, 1])
	plt.xlabel('Number of batch')
	plt.ylabel('Accuracy')
	plt.title('Total Accuracy')
	plt.legend()
	plt.show()


if __name__ == '__main__':
	
	plot = {
		'colors' : ['red', 'green', 'purple', 'blue', 'orange', 'pink', 'black', 'brown', 'yellow', 'gray']
	}

	batch_size = 5 # nombre de données lues à chaque fois
	nb_epochs = 10 # nombre de fois que la base de données sera lue
	# eta_liste = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10] # taux d'apprentissage
	eta = 0.0001
	visu_acc_tt = []
	l = [0, 1, 2]
	n  = 0 

	for i in l:
		# on lit les données
		((data_train,label_train),(data_test,label_test)) = torch.load(gzip.open('mnist.pkl.gz'))

		#On créait les lecteurs de données
		train_dataset = torch.utils.data.TensorDataset(data_train,label_train)
		test_dataset = torch.utils.data.TensorDataset(data_test,label_test)
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

		# on initialise le modèle et ses poids en faisant appel à la fonction MLP
		model = MLP(data_train.shape[1], label_train.shape[1], i)
		#print(model)

		torch.nn.init.uniform_(model.hidden.weight,-0.1,0.1)
		torch.nn.init.uniform_(model.output.weight,-0.1,0.1)
		'''
		if i == 1:
		# MODELE 1
			torch.nn.init.uniform_(model.hidden.weight,-0.1,0.1)
			torch.nn.init.uniform_(model.output.weight,-0.1,0.1)

		if i == 2:
		# MODELE 2 
			torch.nn.init.uniform_(model.hidden.weight,-0.00001,0.00001)
			torch.nn.init.uniform_(model.output.weight,-0.1,0.1)
		
		if i == 3:
		# MODELE 3 
			torch.nn.init.uniform_(model.hidden.weight,-0.1,0.1)
			torch.nn.init.uniform_(model.output.weight,-0.00001,0.00001)

		if i == 4:
		# MODELE 4 
			torch.nn.init.uniform_(model.hidden.weight,-0.00001,0.00001)
			torch.nn.init.uniform_(model.output.weight,-0.00001,0.00001)
		'''
		

		# on initiliase l'optimiseur
		loss_func = torch.nn.MSELoss(reduction='sum')
		optim = torch.optim.SGD(model.parameters(), lr=eta)
		#print(loss_func)
		#print(optim)

		visu_acc = []
		visu_iteration = []

		for n in range(nb_epochs):
			visu_iteration.append(n)
			# on lit toutes les données d'apprentissage : x=data, t=target
			for x,t in train_loader:
				# clear the gradients of all optimized variables
				optim.zero_grad()
				# Forward pass
				output = model.forward(x)
				# on met à jour les poids
				loss = loss_func(t,output)
				# on propage
				loss.backward()
				optim.step()
				
			# test du modèle (on évalue la progression pendant l'apprentissage)
			acc = 0.

			# on lit toutes les donnéees de test
			for x,t in test_loader:
				output = model(x)
				loss = loss_func(output,t)
				# on regarde si la sortie est correcte
				acc += torch.argmax(output,1) == torch.argmax(t,1)
			# on affiche le pourcentage de bonnes réponses
			print(acc)
			print(acc/data_test.shape[0])
			visu_acc.append((acc/data_test.shape[0]).item())

		visu_acc_tt.append(list(visu_acc))
		print(' n = ', n)
		print(visu_acc_tt)
		n = n + 1

	it = list(set(visu_iteration))
	acc = list(visu_acc_tt)
	plot_total_accuracy(it, acc, len(it), plot['colors'])

	

	