"""Réseau du chapitre 3 de Nielsen (network2), étendu aux autres concepts du chapitre :

- activation des couches cachées : sigmoïde, tanh ou ReLU
- couche de sortie : sigmoïde (comme network2) ou softmax + coût log-vraisemblance
- dropout sur les couches cachées
- initialisation des poids : N(0,1), N(0, 1/n) ou N(0, 2/n) (« He », adaptée à ReLU)

network2.py n'est pas modifié : ReseauEtendu en hérite et ne redéfinit que
feedforward (utilisé pour mesurer la précision) et backprop. La mise à jour des
poids (moyenne sur le mini-batch + régularisation L2) reste celle de network2.
"""
import numpy as np

import network2
from network2 import sigmoid, sigmoid_prime


# ---------------------------------------------------------------- activations

def _relu(z):
    return np.maximum(z, 0.0)


def _relu_prime(z):
    return (z > 0).astype(float)


def _tanh_prime(z):
    return 1.0 - np.tanh(z) ** 2


ACTIVATIONS = {
    "sigmoide": (sigmoid, sigmoid_prime),
    "tanh": (np.tanh, _tanh_prime),
    "relu": (_relu, _relu_prime),
}


def softmax(z):
    e = np.exp(z - np.max(z))          # stable numériquement
    return e / np.sum(e)


# ---------------------------------------------------------------- coût de la softmax

class LogLikelihoodCost(object):
    """C = -ln(a_y) : le logarithme de la probabilité donnée à la bonne classe.
    Associé à une sortie softmax, il donne δᴸ = a - y, comme la cross-entropy
    associée à une sortie sigmoïde."""

    @staticmethod
    def fn(a, y):
        return float(-np.log(a[int(np.argmax(y)), 0] + 1e-12))

    @staticmethod
    def delta(z, a, y):
        return a - y


COUTS = {"cross-entropy": network2.CrossEntropyCost,
         "quadratique": network2.QuadraticCost}


# ---------------------------------------------------------------- réseau

class ReseauEtendu(network2.Network):
    """network2.Network avec activation cachée, sortie, dropout et init au choix.

    activation : "sigmoide", "tanh" ou "relu" (couches cachées uniquement)
    sortie     : "sigmoide" (coût cross-entropy ou quadratique, comme network2)
                 ou "softmax" (coût log-vraisemblance imposé)
    dropout    : probabilité d'éteindre chaque neurone caché pendant
                 l'entraînement (0 = pas de dropout ; Nielsen prend 0,5)
    init       : "1/sqrt(n)" (défaut de network2), "N(0,1)" ou "sqrt(2/n)"

    Dropout « inversé » : à l'entraînement, les neurones gardés sont multipliés
    par 1/(1 - p). La sortie moyenne d'un neurone est ainsi la même qu'à
    l'inférence, où tous les neurones sont actifs et rien n'est modifié.
    C'est équivalent à la description de Nielsen (diviser les poids par deux
    au moment du test quand p = 0,5).
    """

    def __init__(self, sizes, cout="cross-entropy", activation="sigmoide",
                 sortie="sigmoide", dropout=0.0, init="1/sqrt(n)"):
        if activation not in ACTIVATIONS:
            raise ValueError("ACTIVATION doit valoir 'sigmoide', 'tanh' ou 'relu'")
        if sortie not in ("sigmoide", "softmax"):
            raise ValueError("SORTIE doit valoir 'sigmoide' ou 'softmax'")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("DROPOUT doit être compris entre 0 et 1 (exclu)")
        if sortie == "softmax":
            cost = LogLikelihoodCost
        elif cout in COUTS:
            cost = COUTS[cout]
        else:
            raise ValueError("COUT doit valoir 'cross-entropy' ou 'quadratique'")

        super().__init__(sizes, cost=cost)       # tire biais N(0,1) et poids N(0, 1/n)
        if init == "N(0,1)":
            self.large_weight_initializer()
        elif init == "sqrt(2/n)":
            self.weights = [np.random.randn(y, x) * np.sqrt(2.0 / x)
                            for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        elif init != "1/sqrt(n)":
            raise ValueError("INIT doit valoir '1/sqrt(n)', 'N(0,1)' ou 'sqrt(2/n)'")

        self.activation = activation
        self.sortie = sortie
        self.dropout = dropout
        self.f, self.f_prime = ACTIVATIONS[activation]
        morceaux = [{"sigmoide": "sigmoïde", "tanh": "tanh", "relu": "ReLU"}[activation]]
        if sortie == "softmax":
            morceaux.append("softmax")
        if dropout:
            morceaux.append("dropout %g" % dropout)
        self.variante = " · ".join(morceaux)

    # -- sortie
    def _sortie(self, z):
        return softmax(z) if self.sortie == "softmax" else sigmoid(z)

    # -- masque de dropout (méthode séparée pour pouvoir la fixer lors des tests)
    def _masque(self, forme):
        p = self.dropout
        return (np.random.rand(*forme) >= p) / (1.0 - p)

    def feedforward(self, a):
        """Inférence : tous les neurones actifs, pas de dropout."""
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = self.f(np.dot(w, a) + b)
        return self._sortie(np.dot(self.weights[-1], a) + self.biases[-1])

    def backprop(self, x, y):
        """Gradient du coût pour un exemple, avec dropout éventuel.
        Même structure que network2.backprop, écrite avec des indices positifs."""
        L = len(self.weights)                    # nombre de couches de poids
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # propagation avant, en gardant z, les activations et les masques
        activations, zs, masques = [x], [], []
        a = x
        for i in range(L - 1):                   # couches cachées
            z = np.dot(self.weights[i], a) + self.biases[i]
            zs.append(z)
            a = self.f(z)
            m = self._masque(a.shape) if self.dropout else None
            if m is not None:
                a = a * m
            masques.append(m)
            activations.append(a)
        z = np.dot(self.weights[-1], a) + self.biases[-1]
        zs.append(z)
        activations.append(self._sortie(z))

        # erreur en sortie (BP1), puis rétropropagation (BP2)
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for i in range(L - 2, -1, -1):           # couche cachée i
            delta = np.dot(self.weights[i + 1].transpose(), delta) * self.f_prime(zs[i])
            if masques[i] is not None:
                delta = delta * masques[i]       # un neurone éteint ne transmet rien
            nabla_b[i] = delta
            nabla_w[i] = np.dot(delta, activations[i].transpose())
        return (nabla_b, nabla_w)
