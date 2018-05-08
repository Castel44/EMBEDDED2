import matplotlib.pyplot as plt

# TRAINING TIME COMPARISON
neur = [100, 500, 1000, 5000, 8000, 15000]

#hpelm
cifar_hpelm = [2.16960930824, 2.36896840731, 2.38313651085, 4.32588251432, 7.59313519796, 29.553097407]
mnist_hpelm =[0.911782344182, 0.926784912745, 0.99379269282, 2.64933697383, 6.27313391368, 29.6792712212]

#tfelm
cifar_tfelm = [1.00946116447, 1.16798170408, 1.33900284767, 2.2006131808, 3.53294857343, 8.88313102722]
mnist_tfelm = [0.559736967087, 0.732259750366, 0.815937042236, 1.74838757515, 3.0245505174, 9.08882133166]

plt.figure(1)
plt.title('tfelm vs hpelm: training time')
plt.plot(neur, mnist_hpelm, color='red', marker='^', label='MNIST hpelm')
plt.plot(neur, mnist_tfelm, color='blue', marker='^', label='MNIST tfelm')
plt.plot(neur, cifar_hpelm, color='red', marker='o', label='CIFAR10 hpelm')
plt.plot(neur, cifar_tfelm, color='blue', marker='o', label='CIFAR10 tfelm')
plt.xlabel('Hidden neuron number')
plt.ylabel('Training time (s)')
plt.legend()
plt.grid()
plt.show()

###########################################################################################