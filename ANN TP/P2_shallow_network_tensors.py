# coding: utf8
# !/usr/bin/env python
# ------------------------------------------------------------------------
# Perceptron en pytorch (en utilisant juste les tenseurs)
# Écrit par Mathieu Lefort
#
# Distribué sous licence BSD.
# ------------------------------------------------------------------------

import gzip, numpy, torch
    
if __name__ == '__main__':
	batch_size = 5 # nombre de données lues à chaque fois
	nb_epochs = 10 # nombre de fois que la base de données sera lue
	eta = 0.01 # taux d'apprentissage
	hidden_layer_size = 64 # taille de la couche cachée
	
	# on lit les données
	((data_train,label_train),(data_test,label_test)) = torch.load(gzip.open('mnist.pkl.gz'))

	#print(f'Shape of data_train: {data_train.shape}')
	#print(f'Shape of label_train: {label_train.shape}')
	#print(f'Shape of data_test: {data_test.shape}')
	#print(f'Shape of label_test: {label_test.shape}')

	# on initialise le modèle et ses poids
	w1 = torch.empty((data_train.shape[1],hidden_layer_size),dtype=torch.float)
	b1 = torch.empty((1,hidden_layer_size),dtype=torch.float)
	torch.nn.init.uniform_(w1,-0.001,0.001)
	torch.nn.init.uniform_(b1,-0.001,0.001)
	w2 = torch.empty((hidden_layer_size, label_train.shape[1]),dtype=torch.float)
	b2 = torch.empty((1,label_train.shape[1]),dtype=torch.float)
	torch.nn.init.uniform_(w2,-0.01,0.01)
	torch.nn.init.uniform_(b2,-0.001,0.001)

	#print(f'Shape of w1: {w1.shape}')
	#print(f'Shape of b1: {b1.shape}')
	#print(f'Shape of w2: {w2.shape}')
	#print(f'Shape of b2: {b2.shape}')

	nb_data_train = data_train.shape[0]
	nb_data_test = data_test.shape[0]
	indices = numpy.arange(nb_data_train,step=batch_size)

	#print(f'Size of nb_data_train: {nb_data_train}')
	#print(f'Size of nb_data_test: {nb_data_test}')
	#print(f'Size of indices: {indices[0]}')

	pass

	for n in range(nb_epochs):
		# on mélange les indices
		numpy.random.shuffle(indices)

		# on lit les données d'apprentissage
		for i in indices:

			# on récupère les entrées
			x = data_train[i:i+batch_size]
			# on calcule la sortie y1 du modèle (couche cachée)
			y1 = 1 / (1 + torch.exp(-torch.mm(x, w1) + b1))
			#y1 = torch.div(torch.ones(batch_size, hidden_layer_size), torch.ones(batch_size, hidden_layer_size) + torch.exp(-torch.mm(x,w1))) + b1
			# on calcule la sortie y2 du modèle (couche sortie)
			y2 = torch.mm(y1, w2) + b2

			
			# on regarde les vrais labels
			t = label_train[i:i+batch_size]
			# on met à jour les poids
			grad2 = (t-y2)
			grad1 = y1 * (1 - y1) * torch.mm(grad2, w2.T)
			
			w2 += eta * torch.mm(y1.T, grad2)
			b2 += eta * grad2.sum(axis=0)
			w1 += eta * torch.mm(x.T, grad1)
			b1 += eta * grad1.sum(axis=0)
			
			
		
		#print(f'Shape of x: {x.shape}')
		#print(f'Shape of y1: {y1.shape}')
		#print(f'Shape of y2: {y2.shape}')
		#print(f'Shape of grad1: {grad1.shape}')
		#print(f'Shape of grad2: {grad2.shape}')
		#print(f'Shape of t: {t.shape}')
		

		
		# test du modèle (on évalue la progression pendant l'apprentissage)
		acc = 0.
		# on lit toutes les donnéees de test
		for i in range(nb_data_test):
			# on récupère les entrées
			x = data_train[i:i+1]
			# on calcule la sortie y1 du modèle
			y1 = 1 / (1 + torch.exp(-torch.mm(x, w1) + b1))
			#y1 = torch.div(torch.ones(batch_size, hidden_layer_size), torch.ones(batch_size, hidden_layer_size) + torch.exp(-torch.mm(x,w1))) + b1
			# on calcule la sortie y2 du modèle
			y2 = torch.mm(y1, w2) + b2
			# on regarde les vrais labels
			t = label_train[i:i+1]
			# on regarde si la sortie est correcte
			acc += torch.argmax(y2,1) == torch.argmax(t,1)

		#print(f'Shape of x: {x.shape}')
		#print(f'Shape of y1: {y1.shape}')
		#print(f'Shape of y2: {y2.shape}')
		#print(f'Shape of t: {t.shape}')
		# on affiche le pourcentage de bonnes réponses
		print(acc/nb_data_test)
		




