"""Test de réception : réseau du chapitre 1 de Nielsen.
Usage : python test_reception.py [epoques]   (défaut 30 ; 1 = essai rapide)"""
import os, sys, time
src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nielsen", "src")
os.chdir(src)  # mnist_loader lit ../data/mnist.pkl.gz
sys.path.insert(0, src)
import mnist_loader, network

epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10])
t = time.time()
net.SGD(training_data, epochs, 10, 3.0, test_data=test_data)
print("duree : %.0f s" % (time.time() - t))
