"""Petits outils pour regarder à l'intérieur des réseaux de Nielsen.

Usage (depuis un notebook de experiences/) :
    import sys, os
    sys.path.insert(0, os.path.abspath("../outils"))
    from visu import charger_mnist, afficher_poids, entrainer_en_direct, comparer
    import network
"""
import operator
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_NIELSEN = os.path.join(RACINE, "nielsen", "src")

# Rend "import network", "import network2", "import mnist_loader" possibles.
if SRC_NIELSEN not in sys.path:
    sys.path.insert(0, SRC_NIELSEN)

# Couleurs des courbes (ordre fixe : le n-ième essai a toujours la n-ième couleur)
COULEURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
MARQUEURS = ["o", "s", "^", "D"]
GRILLE = "#e6e5e0"
TEXTE_2 = "#52514e"


def charger_mnist():
    """Charge MNIST avec le mnist_loader de Nielsen.

    mnist_loader lit '../data/mnist.pkl.gz' par rapport au dossier courant :
    on se place dans nielsen/src le temps du chargement, puis on revient.
    Renvoie (training_data, validation_data, test_data).
    """
    import mnist_loader
    ici = os.getcwd()
    os.chdir(SRC_NIELSEN)
    try:
        return mnist_loader.load_data_wrapper()
    finally:
        os.chdir(ici)


# ---------------------------------------------------------------- poids

def _grille_poids(conteneur, W, echelle, colonnes):
    """Dessine les lignes de W (une par neurone) en images 28×28 dans conteneur
    (une figure ou une sous-figure)."""
    n = W.shape[0]
    colonnes = min(colonnes, n)
    lignes = int(np.ceil(n / colonnes))
    m = echelle if echelle is not None else max(np.abs(W).max(), 1e-12)
    axes = conteneur.subplots(lignes, colonnes, squeeze=False)
    for j, ax in enumerate(axes.flat):
        ax.axis("off")
        if j < n:
            im = ax.imshow(W[j].reshape(28, 28), cmap="RdBu_r", vmin=-m, vmax=m)
            ax.set_title(str(j), fontsize=9)
    conteneur.colorbar(im, ax=axes, shrink=0.8)


def _poids_a_afficher(net, reference, max_neurones, permutation=None):
    W = net.weights[0]
    if W.shape[1] != 784:
        raise ValueError("La première couche doit avoir 784 entrées (images 28×28).")
    if reference is not None:
        W = W - reference
    if permutation is not None:          # remet chaque poids sous son pixel d'origine
        W_ordre = np.empty_like(W)
        W_ordre[:, permutation] = W
        W = W_ordre
    return W[:max_neurones]


def afficher_poids(net, reference=None, titre=None, colonnes=10, max_neurones=100,
                   echelle=None, permutation=None):
    """Affiche les poids de la première couche : une image 28×28 par neurone.

    Chaque neurone de la première couche a 784 poids, un par pixel d'entrée.
    On les remet en image 28×28 pour voir quels pixels il « regarde ».

    net        : un Network de network.py ou network2.py
    reference  : une copie antérieure de net.weights[0] (par ex. W0 avant
                 entraînement). Si fournie, on affiche la différence
                 poids actuels − reference : ce que l'entraînement a changé.
    titre      : titre de la figure
    colonnes   : nombre d'images par ligne
    max_neurones : on n'affiche que les premiers neurones au-delà
    echelle    : borne de la barre de couleurs (±echelle). Par défaut, le plus
                 grand |poids| affiché. La fixer permet de comparer des images
                 successives sans que l'échelle bouge.
    permutation : si le réseau a été entraîné sur des pixels permutés, la
                 permutation utilisée : les poids sont remis sous leur pixel
                 d'origine avant affichage (None = affichage tel quel)

    Rouge = poids positif, bleu = négatif, blanc = 0.
    Même échelle pour toutes les images, symétrique autour de zéro.
    """
    W = _poids_a_afficher(net, reference, max_neurones, permutation)
    n = W.shape[0]
    c = min(colonnes, n)
    lignes = int(np.ceil(n / c))
    fig = plt.figure(layout="constrained", figsize=(1.5 * c + 1.2, 1.6 * lignes + 0.5))
    _grille_poids(fig, W, echelle, colonnes)
    if titre is None:
        titre = ("Variation des poids depuis la référence" if reference is not None
                 else "Poids de la première couche")
    fig.suptitle("%s  (%d neurones)" % (titre, n))
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------- précision

def _precision(net, donnees, max_exemples=10000):
    """Part des exemples correctement classés (entre 0 et 1).
    Accepte les deux formats de Nielsen : étiquette entière (test, validation)
    ou vecteur one-hot (entraînement)."""
    donnees = donnees[:max_exemples]
    bons = 0
    for x, y in donnees:
        cible = y if np.ndim(y) == 0 else int(np.argmax(y))
        bons += int(np.argmax(net.feedforward(x)) == cible)
    return bons / len(donnees)


def _style_axe(ax):
    ax.grid(axis="y", color=GRILLE, linewidth=0.8)
    ax.set_axisbelow(True)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    for cote in ("left", "bottom"):
        ax.spines[cote].set_color(GRILLE)
    ax.tick_params(colors=TEXTE_2, labelsize=9)
    ax.set_xlabel("époques", color=TEXTE_2)
    ax.set_ylabel("précision (%)", color=TEXTE_2)


def _tracer_historique(ax, historique, couleur, nom=None, marqueur="o"):
    """Courbe test en trait plein, courbe entraînement (si suivie) en pointillés."""
    ep = [p["epoque"] for p in historique]
    tous = max(1, len(ep) // 20)            # au plus ~20 marqueurs par courbe
    ax.plot(ep, [100 * p["test"] for p in historique], color=couleur, lw=2,
            marker=marqueur, ms=4, markevery=tous,
            label=(nom + " — test") if nom else "test")
    if historique and historique[0].get("entrainement") is not None:
        ax.plot(ep, [100 * p["entrainement"] for p in historique], color=couleur,
                lw=2, ls="--", marker=marqueur, ms=4, markevery=tous, mfc="white",
                label=(nom + " — entraînement") if nom else "entraînement")


# ---------------------------------------------------------------- entraînement

# ---------------------------------------------------------------- transformations

GRAINE_PERMUTATION = 1234        # toujours la même permutation, quelle que soit GRAINE


def permutation_fixe(graine=GRAINE_PERMUTATION):
    """Une permutation des 784 pixels, toujours la même pour une graine donnée."""
    return np.random.RandomState(graine).permutation(784)


def _decalage(decalage):
    """None ou 0 -> (0, 0) ; un entier k -> (k, 0) ; un couple (dx, dy) tel quel."""
    if not decalage:
        return 0, 0
    if np.isscalar(decalage):
        return int(decalage), 0
    return int(decalage[0]), int(decalage[1])


def decaler_image(x, dx, dy):
    """Décale une image 784×1 de dx pixels vers la droite et dy vers le bas
    (valeurs négatives : vers la gauche / le haut). Ce qui sort du cadre est
    perdu, ce qui entre est noir (0)."""
    img = x.reshape(28, 28)
    out = np.zeros_like(img)
    out[max(dy, 0):28 + min(dy, 0), max(dx, 0):28 + min(dx, 0)] = \
        img[max(-dy, 0):28 + min(-dy, 0), max(-dx, 0):28 + min(-dx, 0)]
    return out.reshape(784, 1)


def preparer_donnees(training_original, test_data, n_images=None, permuter=False,
                     decalage_test=None):
    """Prépare les données d'un essai.

    n_images      : None = toutes les images d'entraînement, sinon les n premières
    permuter      : applique la même permutation fixe des pixels à TOUTES les
                    images (entraînement et test)
    decalage_test : décale les images de TEST seulement (entier = vers la
                    droite, ou couple (dx, dy)) ; l'entraînement reste intact

    Renvoie (donnees_entrainement, donnees_test, permutation ou None).
    Les données d'origine ne sont jamais modifiées.
    """
    train = list(training_original if n_images is None else training_original[:n_images])
    test = list(test_data)
    dx, dy = _decalage(decalage_test)
    if dx or dy:                              # décalage dans l'espace de l'image...
        test = [(decaler_image(x, dx, dy), y) for x, y in test]
    perm = None
    if permuter:                              # ...puis permutation éventuelle
        perm = permutation_fixe()
        train = [(x[perm], y) for x, y in train]
        test = [(x[perm], y) for x, y in test]
    return train, test, perm


def preparer_validation(validation_data, permuter=False):
    """Validation préparée comme l'entraînement : permutée si besoin, jamais décalée."""
    if not permuter:
        return list(validation_data)
    perm = permutation_fixe()
    return [(x[perm], y) for x, y in validation_data]


def apercu_transformations(test_data, index=0, decalages=(2, 4), chemin=None):
    """Figure : une image de test, ses versions décalées, sa version permutée,
    et la version permutée remise en ordre. Enregistrée si chemin est donné."""
    x, y = test_data[index]
    perm = permutation_fixe()
    images = [(x, "référence (chiffre %d)" % y)]
    for k in decalages:
        images.append((decaler_image(x, k, 0), "décalée de %d px vers la droite" % k))
    images.append((x[perm], "pixels permutés\n(ce que voit le réseau)"))
    remise = np.empty_like(x); remise[perm] = x[perm]
    images.append((remise, "permutée puis remise en ordre\n(identique à la référence)"))

    # superposition référence / premier décalage : rouge = référence seule,
    # bleu = décalée seule, gris foncé = encre aux mêmes pixels
    ref = x.reshape(28, 28); dec = decaler_image(x, decalages[0], 0).reshape(28, 28)
    rgb = np.stack([1 - 0.8 * dec, 1 - 0.8 * np.maximum(ref, dec), 1 - 0.8 * ref], axis=-1)
    commun = np.minimum(ref, dec).sum() / ref.sum()

    fig, axes = plt.subplots(1, len(images) + 1, figsize=(2.6 * (len(images) + 1), 3.3),
                             layout="constrained")
    for ax, (img, titre) in zip(axes, images):
        ax.imshow(img.reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(titre, fontsize=9)
    axes[-1].imshow(np.clip(rgb, 0, 1))
    axes[-1].set_title("référence (rouge) et décalée de %d px (bleu)\n%d %% de l'encre "
                       "reste sur les mêmes pixels" % (decalages[0], round(100 * commun)),
                       fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for c in ax.spines.values():
            c.set_color(GRILLE)
    if chemin:
        fig.savefig(chemin, dpi=130, facecolor="white")
    plt.show()
    plt.close(fig)


def creer_reseau(architecture, reseau="network", cout="cross-entropy", init="1/sqrt(n)",
                 activation="sigmoide", sortie="sigmoide", dropout=0.0):
    """Crée un réseau de Nielsen.

    reseau : "network"  -> chapitre 1 : sigmoïde, coût quadratique, poids N(0,1)
             "network2" -> chapitre 3 : coût et initialisation au choix
             "etendu"   -> network2 + activation cachée, sortie softmax et dropout
                           (outils/reseau_etendu.py)
    cout   : (network2, etendu) "cross-entropy" ou "quadratique"
    init   : (network2, etendu) "1/sqrt(n)" ou "N(0,1)" ; (etendu) aussi "sqrt(2/n)"
    activation, sortie, dropout : (etendu) voir reseau_etendu.ReseauEtendu
    """
    if reseau == "network":
        import network
        return network.Network(list(architecture))
    if reseau == "network2":
        import network2
        couts = {"cross-entropy": network2.CrossEntropyCost,
                 "quadratique": network2.QuadraticCost}
        if cout not in couts:
            raise ValueError("COUT doit valoir 'cross-entropy' ou 'quadratique'")
        net = network2.Network(list(architecture), cost=couts[cout])
        if init == "N(0,1)":
            net.large_weight_initializer()
        elif init != "1/sqrt(n)":
            raise ValueError("Avec network2, INIT doit valoir '1/sqrt(n)' ou 'N(0,1)' "
                             "('sqrt(2/n)' demande RESEAU = 'etendu')")
        return net
    if reseau == "etendu":
        from reseau_etendu import ReseauEtendu
        return ReseauEtendu(list(architecture), cout=cout, activation=activation,
                            sortie=sortie, dropout=dropout, init=init)
    raise ValueError("RESEAU doit valoir 'network', 'network2' ou 'etendu'")


def etiquette_essai(reseau, init, graine, continuer, note="", permuter=False,
                    decalage_test=None):
    """Complète le nom automatique avec ce que les paramètres ne disent pas."""
    morceaux = []
    if permuter:
        morceaux.append("pixels permutés")
    dx, dy = _decalage(decalage_test)
    if dx or dy:
        morceaux.append("test décalé (%+d, %+d) px" % (dx, dy))
    if reseau in ("network2", "etendu"):
        morceaux.append("init " + init)
    if graine and not continuer:              # en suite, la graine n'est pas réappliquée
        morceaux.append("graine %s" % graine)
    if continuer:
        morceaux.append("suite")
    if note:
        morceaux.append(note)
    return " · ".join(morceaux)


class Historique(list):
    """Liste des points mesurés pendant un essai, plus ses paramètres.

    .params : architecture, nombre d'images, époques, taille du mini-batch,
              eta, eta/m (et coût + lambda pour network2)
    .nom    : nom automatique construit à partir de ces paramètres
    """

    def __init__(self, params):
        super().__init__()
        self.params = params
        morceaux = [str(params["architecture"])]
        if params["n_images"] != 50000:
            morceaux.append("%d images" % params["n_images"])
        morceaux += ["%d ép." % params["epoques"], "m=%d" % params["taille_batch"],
                     "η=%g" % params["eta"], "η/m=%.3g" % params["eta_sur_m"]]
        if params.get("planning"):
            morceaux.append(params["planning"])
        if params.get("cout"):
            morceaux.append(params["cout"])
            if params.get("lmbda"):
                morceaux.append("λ=%g" % params["lmbda"])
        if params.get("variante"):
            morceaux.append(params["variante"])
        if params.get("etiquette"):
            morceaux.append(params["etiquette"])
        self.nom = " · ".join(morceaux)


def planning_eta(type_planning="constant", paliers=None, echauffement=0, eta_fin=0.0,
                 patience=5, facteur=2, arret=128):
    """Décrit l'évolution de η pendant l'entraînement (à passer à entrainer_en_direct).

    "constant" : η fixe
    "paliers"  : paliers = [(époque de début, η), ...], ex. [(0, 3), (30, 0.5), (130, 0.1)]
    "cosinus"  : montée linéaire de 0 à η pendant `echauffement` époques, puis
                 décroissance en cosinus jusqu'à eta_fin à la dernière époque
    "plateau"  : règle de Nielsen : η est divisé par `facteur` quand la précision
                 de VALIDATION n'a pas progressé depuis `patience` époques ;
                 l'entraînement s'arrête quand η < η_initial / arret
    """
    if type_planning not in ("constant", "paliers", "cosinus", "plateau"):
        raise ValueError("PLANNING doit valoir 'constant', 'paliers', 'cosinus' ou 'plateau'")
    if type_planning == "paliers" and not paliers:
        raise ValueError("PLANNING = 'paliers' demande une liste PALIERS")
    return {"type": type_planning, "paliers": [tuple(p) for p in (paliers or [])],
            "echauffement": echauffement, "eta_fin": eta_fin,
            "patience": patience, "facteur": facteur, "arret": arret}


def _decrire_planning(pl):
    t = pl.get("type", "constant")
    if t == "paliers":
        return "paliers η %s (ép. %s)" % ("→".join("%g" % v for _, v in pl["paliers"]),
                                           "/".join("%g" % d for d, _ in pl["paliers"]))
    if t == "cosinus":
        return "cosinus (échauff. %g ép., η fin %g)" % (pl["echauffement"], pl["eta_fin"])
    if t == "plateau":
        return "plateau (÷%g, patience %d, arrêt η/%g)" % (pl["facteur"], pl["patience"],
                                                          pl["arret"])
    return ""


def entrainer_en_direct(net, training_data, epoques, taille_batch, eta,
                        test_data=None, images_par_epoque=5, reference=None,
                        echelle=None, lmbda=0.0, aussi_entrainement=False,
                        etiquette="", permutation=None, planning=None,
                        validation_data=None, voir_poids=True):
    """Entraîne le réseau en réaffichant les poids et la courbe de précision.

    C'est exactement la boucle de Network.SGD de Nielsen (mélange, découpage
    en mini-batches, update_mini_batch), avec un affichage intercalé.

    images_par_epoque : nombre d'affichages par époque. Peut être < 1 :
                        0.1 = un affichage toutes les 10 époques (utile pour
                        les longs entraînements sur peu d'images)
    reference, echelle : comme pour afficher_poids (reference=W0 pour ne voir
                        que ce que l'entraînement a changé)
    lmbda             : régularisation L2, utilisée seulement avec network2
    aussi_entrainement : suit aussi la précision sur les données d'entraînement
                        (sur 10 000 exemples au plus). Indispensable pour voir
                        le sur-apprentissage.
    etiquette         : texte ajouté au nom automatique de l'essai
    permutation       : permutation des pixels (voir preparer_donnees), pour
                        afficher les poids dans l'ordre d'origine
    planning          : évolution de η (voir planning_eta) ; None = η constant.
                        eta est alors la valeur de départ ou maximale.
    validation_data   : données de validation, nécessaires au planning "plateau"
    voir_poids        : False = ni calcul ni tracé des images de poids pendant
                        l'entraînement, seulement les courbes (précision, η).
                        Les poids finaux restent visibles avec afficher_poids.

    Chronomètre séparément l'apprentissage seul (mélange + mini-batches), la
    mesure de la précision et le tracé des figures ; les trois sont affichés
    à la fin.

    Renvoie l'historique : liste de points {"epoque", "test", "entrainement",
    "temps_calcul", "eta"}, avec les paramètres de l'essai (.params) et un nom
    automatique (.nom). Il se passe tel quel à comparer().
    """
    import random
    from math import cos, pi
    from time import perf_counter
    from IPython.display import clear_output

    pl = planning or {"type": "constant"}
    type_pl = pl.get("type", "constant")
    if type_pl == "plateau" and not validation_data:
        raise ValueError("Le planning 'plateau' demande validation_data")
    eta_nom = pl["paliers"][0][1] if type_pl == "paliers" else eta

    n = len(training_data)
    M = int(np.ceil(n / taille_batch))          # mini-batches par époque
    pas = M / images_par_epoque                 # mini-batches entre deux affichages
    total = epoques * M
    prochain = pas
    historique = Historique({
        "architecture": list(net.sizes), "n_images": n, "epoques": epoques,
        "taille_batch": taille_batch, "eta": eta_nom, "eta_sur_m": eta_nom / taille_batch,
        "planning": _decrire_planning(pl),
        "cout": net.cost.__name__.replace("Cost", "") if hasattr(net, "cost") else None,
        "lmbda": lmbda if hasattr(net, "cost") else None,
        "variante": getattr(net, "variante", None),
        "etiquette": etiquette})

    etat = {"eta": eta, "meilleur": -1.0, "attente": 0}   # pour le planning "plateau"

    def eta_a(t):
        """η à appliquer quand t époques ont été parcourues (t réel)."""
        if type_pl == "paliers":
            valeur = pl["paliers"][0][1]
            for debut_palier, v in pl["paliers"]:
                if t >= debut_palier - 1e-9:
                    valeur = v
            return valeur
        if type_pl == "cosinus":
            w, fin = pl.get("echauffement", 0), pl.get("eta_fin", 0.0)
            if w > 0 and t < w:
                return eta * (t + 1.0 / M) / w
            if epoques <= w:
                return eta
            return fin + 0.5 * (eta - fin) * (1 + cos(pi * (t - w) / (epoques - w)))
        if type_pl == "plateau":
            return etat["eta"]
        return eta

    faits = 0                                   # mini-batches traités
    dernier_affiche = -1
    eta_k = eta_a(0.0)
    # chronomètres : apprentissage seul / mesure de la précision / tracé des figures
    temps = {"calcul": 0.0, "mesure": 0.0, "affichage": 0.0}
    debut = perf_counter()
    arreter = False
    trace = historique.params["trace_eta"] = []    # η au début de chaque époque
    for e in range(epoques):
        trace.append([e, eta_a(float(e))])
        random.shuffle(training_data)
        for k in range(0, n, taille_batch):
            mini_batch = training_data[k:k + taille_batch]
            eta_k = eta_a(faits / M)
            if hasattr(net, "cost"):            # network2.py
                net.update_mini_batch(mini_batch, eta_k, lmbda, n)
            else:                               # network.py
                net.update_mini_batch(mini_batch, eta_k)
            faits += 1
            if faits >= prochain - 1e-9 or faits == total:
                temps["calcul"] += perf_counter() - debut
                prochain += pas
                _afficher_etat(net, historique, faits / M, faits * taille_batch,
                               training_data, test_data, aussi_entrainement,
                               reference, echelle, clear_output, temps, permutation,
                               eta_k, voir_poids)
                dernier_affiche = faits
                debut = perf_counter()

        if type_pl == "plateau":                # règle de Nielsen, en fin d'époque
            temps["calcul"] += perf_counter() - debut
            t0 = perf_counter()
            p_val = _precision(net, validation_data)
            temps["mesure"] += perf_counter() - t0
            if p_val > etat["meilleur"]:
                etat["meilleur"], etat["attente"] = p_val, 0
            else:
                etat["attente"] += 1
                if etat["attente"] >= pl.get("patience", 5):
                    etat["eta"] /= pl.get("facteur", 2)
                    etat["attente"] = 0
                    if etat["eta"] < eta / pl.get("arret", 128):
                        arreter = True
            if arreter:
                if dernier_affiche != faits:
                    _afficher_etat(net, historique, faits / M, faits * taille_batch,
                                   training_data, test_data, aussi_entrainement,
                                   reference, echelle, clear_output, temps,
                                   permutation, eta_k, voir_poids)
                print("Arrêt à l'époque %d : η est descendu sous η/%g "
                      "(meilleure précision de validation %.2f %%)."
                      % (e + 1, pl.get("arret", 128), 100 * etat["meilleur"]))
                debut = perf_counter()
                break
            debut = perf_counter()
    temps["calcul"] += perf_counter() - debut
    trace.append([faits / M, eta_k])
    print("Temps d'apprentissage seul : %.1f s   (mesure de la précision : %.1f s, "
          "affichage : %.1f s, total : %.1f s)"
          % (temps["calcul"], temps["mesure"], temps["affichage"], sum(temps.values())))
    return historique


def _courbe_eta(historique):
    """(époques, η) : la trace époque par époque si elle existe, sinon les points."""
    trace = (getattr(historique, "params", None) or {}).get("trace_eta")
    if trace:
        return [t for t, _ in trace], [v for _, v in trace]
    pts = [p for p in historique if p.get("eta") is not None]
    return [p["epoque"] for p in pts], [p["eta"] for p in pts]


def _eta_variable(historique):
    _, etas = _courbe_eta(historique)
    return len(etas) > 1 and max(etas) - min(etas) > 1e-12


def _tracer_eta(ax, historique, couleur, marqueur="o"):
    ep, et = _courbe_eta(historique)
    if ep:
        ax.plot(ep, et, color=couleur, lw=2, marker=marqueur, ms=3,
                markevery=max(1, len(ep) // 20), drawstyle="steps-post")


def _style_axe_eta(ax):
    _style_axe(ax)
    ax.set_ylabel("η", color=TEXTE_2)


def _afficher_etat(net, historique, epoque, vus, training_data, test_data,
                   aussi_entrainement, reference, echelle, clear_output, temps,
                   permutation=None, eta_actuel=None, voir_poids=True):
    from time import perf_counter
    t0 = perf_counter()
    titre = "%s\nÉpoque %.1f — %d exemples vus — apprentissage %.1f s" % (
        historique.nom, epoque, vus, temps["calcul"])
    if eta_actuel is not None:
        titre += " — η = %.3g" % eta_actuel
    if test_data:
        point = {"epoque": epoque, "test": _precision(net, test_data),
                 "entrainement": (_precision(net, training_data)
                                  if aussi_entrainement else None),
                 "temps_calcul": temps["calcul"], "eta": eta_actuel}
        historique.append(point)
        titre += "\nprécision test %.1f %%" % (100 * point["test"])
        if aussi_entrainement:
            titre += ", entraînement %.1f %%" % (100 * point["entrainement"])
    t1 = perf_counter()
    temps["mesure"] += t1 - t0

    avec_eta = _eta_variable(historique)
    if not voir_poids and not historique:      # rien à tracer : texte seul
        clear_output(wait=True)
        print(titre)
        temps["affichage"] += perf_counter() - t1
        return
    if voir_poids:
        W = _poids_a_afficher(net, reference, 100, permutation)
        c = min(10, W.shape[0])
        lignes = int(np.ceil(W.shape[0] / c))
        h_poids = 1.6 * lignes + 0.5
        h_courbe = (3.0 + (1.4 if avec_eta else 0.0)) if historique else 0.0
        fig = plt.figure(layout="constrained", figsize=(1.5 * c + 1.2, h_poids + h_courbe))
        if historique:
            haut, bas = fig.subfigures(2, 1, height_ratios=[h_poids, h_courbe])
        else:
            haut, bas = fig, None
        _grille_poids(haut, W, echelle, 10)
    else:                                      # courbes seules : figure plus compacte
        fig = plt.figure(layout="constrained",
                         figsize=(10, 4.6 + (1.8 if avec_eta else 0.0)))
        bas = fig
    if bas is not None:
        if avec_eta:
            ax, ax_eta = bas.subplots(2, 1, sharex=True, height_ratios=[3.0, 1.4])
        else:
            ax, ax_eta = bas.subplots(), None
        _tracer_historique(ax, historique, COULEURS[0])
        _style_axe(ax)
        if aussi_entrainement:
            ax.legend(frameon=False, fontsize=9)
        if ax_eta is not None:
            ax.set_xlabel("")
            _tracer_eta(ax_eta, historique, COULEURS[0])
            _style_axe_eta(ax_eta)
    fig.suptitle(titre)
    clear_output(wait=True)
    plt.show()
    plt.close(fig)
    temps["affichage"] += perf_counter() - t1


# ---------------------------------------------------------------- comparaison

_CLES_NUM = {"η/m": "eta_sur_m", "eta/m": "eta_sur_m", "η": "eta", "eta": "eta",
             "m": "taille_batch", "λ": "lmbda", "lambda": "lmbda", "lmbda": "lmbda",
             "époques": "epoques", "epoques": "epoques", "ep": "epoques",
             "images": "n_images"}
_CRITERE_NUM = re.compile(r"^\s*(η/m|eta/m|η|eta|m|λ|lambda|lmbda|époques|epoques|ep|images)"
                          r"\s*(<=|>=|=|<|>)\s*([0-9.eE+-]+)\s*$", re.IGNORECASE)
_OPERATEURS = {"<=": operator.le, ">=": operator.ge, "<": operator.lt, ">": operator.gt}


def _sans_espaces(texte):
    return texte.replace(" ", "").lower()


def _critere_ok(nom, params, critere):
    """Un critère du filtre :
    - numérique sur un paramètre : "m=10", "eta=3", "eta/m>=0.25", "lambda<1",
      "ep=10", "images=1000" (opérateurs = < > <= >=)
    - motif sur le nom, * = n'importe quoi : "[784, 30, *]", "[784, *, 10]",
      "CrossEntropy". Les espaces sont ignorés ; le motif doit correspondre à
      un morceau entier du nom (les morceaux sont séparés par « · »)."""
    num = _CRITERE_NUM.match(critere)
    if num:
        valeur = (params or {}).get(_CLES_NUM[num.group(1).lower()])
        if valeur is None:
            return False
        cible, op = float(num.group(3)), num.group(2)
        if op == "=":
            return abs(valeur - cible) <= 1e-9 * max(1.0, abs(cible))
        return _OPERATEURS[op](valeur, cible)
    motif = re.escape(_sans_espaces(critere)).replace(r"\*", ".*").replace(r"\?", ".")
    return any(re.fullmatch(motif, _sans_espaces(morceau)) for morceau in nom.split("·"))


def _filtrer(tous, filtre):
    """tous : liste de (numéro, nom, historique)."""
    if filtre is None:
        return tous
    if isinstance(filtre, str):
        criteres = [c for c in filtre.split(";") if c.strip()]
    elif all(isinstance(c, str) for c in filtre):
        criteres = list(filtre)
    else:                                   # liste de numéros
        numeros = set(filtre)
        return [e for e in tous if e[0] in numeros]
    return [(num, nom, h) for num, nom, h in tous
            if all(_critere_ok(nom, getattr(h, "params", {}), c) for c in criteres)]


def comparer(essais, titre="Comparaison des essais", ylim=(70, 100), filtre=None):
    """Trace la précision de plusieurs essais sur un même graphique.

    essais : une liste d'historiques renvoyés par entrainer_en_direct
             (nommés automatiquement d'après leurs paramètres) :
                 essais.append(historique)
             ou un dictionnaire {nom choisi: historique} :
                 essais["mon essai"] = historique
    ylim   : bornes de l'ordonnée en %, (70, 100) par défaut ; None = automatique.
             Les essais entièrement sous la borne basse sont signalés en titre.
    filtre : pour n'afficher qu'une partie des essais :
             - un ou plusieurs critères séparés par « ; » (tous doivent être
               vrais), par exemple :
                 "[784, 30, *]"            architecture 784, 30 puis n'importe quoi
                 "[784, *, 10]"            toutes les architectures 784 … 10
                 "m=10; eta=3"             paramètres exacts
                 "[784, *, 10]; eta/m>=0.25"
               Paramètres numériques : m, eta (ou η), eta/m, lambda, ep, images,
               avec = < > <= >=.
             - une liste de numéros, ex. [3, 7, 12] (numéros de la légende)

    Chaque essai garde son numéro, sa couleur et son marqueur quel que soit le
    filtre. Trait plein = test, pointillés = entraînement (si suivi).
    Entre parenthèses : temps d'apprentissage seul de l'essai.
    La légende est sous le graphe : la zone de tracé garde toujours la même taille.
    """
    vus, tous = set(), []
    for nom, h in _paires(essais):
        if id(h) not in vus:
            vus.add(id(h))
            tous.append((len(tous) + 1, nom, h))      # numéro stable

    choisis = _filtrer(tous, filtre)
    if not choisis:
        print("Aucun essai ne correspond au filtre.")
        return

    entrees = sum(2 if (h and h[0].get("entrainement") is not None) else 1
                  for _, _, h in choisis)
    plus_long = max(len(nom) for _, nom, _ in choisis) + 35   # + numéro, temps, « — test »
    colonnes = 2 if plus_long <= 90 else 1                     # noms longs : une colonne
    lignes_legende = int(np.ceil(entrees / colonnes))
    avec_eta = any(_eta_variable(h) for _, _, h in choisis)
    h_eta = 1.8 if avec_eta else 0.0
    fig = plt.figure(layout="constrained",
                     figsize=(13, 4.5 + h_eta + 0.24 * lignes_legende + 0.3))
    if avec_eta:
        ax, ax_eta = fig.subplots(2, 1, sharex=True, height_ratios=[4.5, h_eta])
    else:
        ax, ax_eta = fig.add_subplot(), None
    hors_cadre = []
    for num, nom, h in choisis:
        i = num - 1
        couleur = COULEURS[i % len(COULEURS)]
        marqueur = MARQUEURS[(i // len(COULEURS)) % len(MARQUEURS)]
        etiquette = "#%d  %s" % (num, nom)
        if h and h[-1].get("temps_calcul") is not None:
            etiquette += " (%.1f s)" % h[-1]["temps_calcul"]
        _tracer_historique(ax, h, couleur, etiquette, marqueur)
        if ax_eta is not None:
            _tracer_eta(ax_eta, h, couleur, marqueur)
        if ylim and h and max(p["test"] for p in h) * 100 < ylim[0]:
            hors_cadre.append("#%d" % num)
    _style_axe(ax)
    if ax_eta is not None:
        ax.set_xlabel("")
        _style_axe_eta(ax_eta)
    if ylim:
        ax.set_ylim(*ylim)
    sous_titre = ""
    if hors_cadre:
        sous_titre = "   (sous %g %%, hors cadre : %s)" % (ylim[0], ", ".join(hors_cadre))
    ax.set_title(titre + sous_titre, loc="left")
    fig.legend(loc="outside lower center", ncol=colonnes, frameon=False, fontsize=9)
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------- sauvegarde

DOSSIER_RESULTATS = os.path.join(RACINE, "experiences", "resultats")

_COLONNES = ["essai", "architecture", "n_images", "epoques", "taille_batch", "eta",
             "eta_sur_m", "cout", "lambda", "planning", "variante", "epoque",
             "precision_test_pct",
             "precision_entrainement_pct", "temps_apprentissage_s"]


def _paires(essais):
    if isinstance(essais, dict):
        return list(essais.items())
    return [(getattr(h, "nom", "essai %d" % (i + 1)), h) for i, h in enumerate(essais)]


def exporter(essais, nom_fichier="essais", ecraser=False):
    """Enregistre les essais dans experiences/resultats/ :

    <nom_fichier>.csv  : une ligne par point mesuré, lisible dans Excel
                         (séparateur « ; », virgule décimale)
    <nom_fichier>.json : tout l'historique, rechargeable avec charger_essais()
                         après un redémarrage du noyau

    Sécurité : si le fichier existant contient plus d'essais que la série en
    mémoire, rien n'est écrit (il faut ecraser=True). La version précédente est
    toujours gardée dans <nom_fichier>.precedent.json / .csv.
    """
    import csv
    import json
    import shutil
    os.makedirs(DOSSIER_RESULTATS, exist_ok=True)

    def fr(x, chiffres=None):
        if x is None:
            return ""
        if chiffres is not None:
            x = round(x, chiffres)
        return str(x).replace(".", ",")

    chemin_csv = os.path.join(DOSSIER_RESULTATS, nom_fichier + ".csv")
    chemin_json = os.path.join(DOSSIER_RESULTATS, nom_fichier + ".json")
    sauvegarde, vus = [], set()
    for nom, h in _paires(essais):
        if id(h) not in vus:
            vus.add(id(h))
            sauvegarde.append({"nom": nom, "params": getattr(h, "params", {}) or {},
                               "points": list(h)})

    if os.path.exists(chemin_json):
        with open(chemin_json, encoding="utf-8") as f:
            deja = len(json.load(f))
        if deja > len(sauvegarde) and not ecraser:
            print("ATTENTION : %s contient %d essais, la série en mémoire seulement %d.\n"
                  "Rien n'a été écrit. Recharge la série avec charger_essais(%r), ou "
                  "force avec exporter(..., ecraser=True)." % (
                      os.path.basename(chemin_json), deja, len(sauvegarde), nom_fichier))
            return
        for chemin in (chemin_json, chemin_csv):
            if os.path.exists(chemin):
                racine, ext = os.path.splitext(chemin)
                shutil.copyfile(chemin, racine + ".precedent" + ext)

    with open(chemin_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(_COLONNES)
        for e in sauvegarde:
            p = e["params"]
            for pt in e["points"]:
                w.writerow([e["nom"], p.get("architecture", ""), p.get("n_images", ""),
                            p.get("epoques", ""), p.get("taille_batch", ""),
                            fr(p.get("eta")), fr(p.get("eta_sur_m")), p.get("cout") or "",
                            fr(p.get("lmbda")), p.get("planning") or "",
                            p.get("variante") or "", fr(pt["epoque"], 4),
                            fr(100 * pt["test"], 2),
                            fr(None if pt.get("entrainement") is None
                               else 100 * pt["entrainement"], 2),
                            fr(pt.get("temps_calcul"), 2)])
    with open(chemin_json, "w", encoding="utf-8") as f:
        json.dump(sauvegarde, f, ensure_ascii=False, indent=1, default=float)
    print("Série « %s » enregistrée : %d essais (resultats/%s.csv et .json)"
          % (nom_fichier, len(sauvegarde), nom_fichier))


def charger_essais(nom_fichier="essais"):
    """Recharge les essais enregistrés par exporter() (liste d'historiques,
    utilisable telle quelle avec comparer et exporter)."""
    import json
    with open(os.path.join(DOSSIER_RESULTATS, nom_fichier + ".json"), encoding="utf-8") as f:
        sauvegarde = json.load(f)
    essais = []
    for e in sauvegarde:
        p = dict(e["params"])
        if "architecture" in p:
            h = Historique(p)
        else:                       # essai sans paramètres connus
            h = Historique.__new__(Historique)
            list.__init__(h)
            h.params = p
        h.nom = e["nom"]
        h.extend(e["points"])
        essais.append(h)
    return essais


def charger_serie(nom_fichier="essais"):
    """Recharge une série enregistrée, ou renvoie une série vide si elle
    n'existe pas encore."""
    if os.path.exists(os.path.join(DOSSIER_RESULTATS, nom_fichier + ".json")):
        essais = charger_essais(nom_fichier)
        print("Série « %s » : %d essais rechargés." % (nom_fichier, len(essais)))
        return essais
    print("Série « %s » : nouvelle série." % nom_fichier)
    return []
