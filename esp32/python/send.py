"""Étape 5 : envoie des images à la carte cnn-test par l'API ESPHome et compare à Python.

    python esp32\\python\\send.py                     # 10 images du jeu de test MNIST, tirées au hasard
    python esp32\\python\\send.py --index 9729 4601   # images précises du jeu de test
    python esp32\\python\\send.py --png chiffre.png   # une image à toi (fond clair ou foncé), cadrée comme MNIST

L'image part du PC sous forme brute (784 entiers 0..255, ligne par ligne), par
l'action API « classify » : c'est l'ESP qui fait le prétraitement et l'inférence.
Le script calcule la même chose en Python (float32, poids du modèle exporté)
et vérifie que l'ESP donne la même prédiction.

Clé de chiffrement de l'API : variable d'environnement CNN_TEST_API_KEY, ou ligne
cnn_test_api_key: "..." dans esp32/esphome/secrets.yaml (ignoré par git).
"""
import argparse
import asyncio
import inspect
import json
import os
import re
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(os.path.dirname(ICI))
sys.path.insert(0, ICI)
from donnees import charger_u8, pretraiter            # noqa: E402

MODELE = os.path.join(RACINE, "esp32", "modeles", "mnist_14x14_196_30_10.npz")
SECRETS = os.path.join(RACINE, "esp32", "esphome", "secrets.yaml")
NOMS = {"prediction": "Chiffre prédit", "confiance": "Confiance", "marge": "Marge",
        "temps": "Temps d'inférence"}


# ---------------------------------------------------------------- référence Python

def charger_modele(chemin):
    z = np.load(chemin)
    meta = json.loads(str(z["meta"]))
    couches = [(z["W%d" % i].astype(np.float32), z["b%d" % i].astype(np.float32), act)
               for i, act in enumerate(meta["activations"])]
    return couches, meta


def predire(couches, meta, image_u8):
    """Même calcul que export.py : float32, poids arrondis en float32."""
    a = pretraiter(image_u8[None], meta["pretraitement"]["facteur"])[0]
    for W, b, act in couches:
        z = (W @ a + b).astype(np.float32)
        if act == "sigmoide":
            a = (np.float32(1) / (np.float32(1) + np.exp(-z))).astype(np.float32)
        else:
            e = np.exp(z - z.max())
            a = (e / e.sum()).astype(np.float32)
    return int(a.argmax()), float(a.max())


def image_png(chemin, cadrage="mnist"):
    """Image quelconque -> 28×28 uint8 au format MNIST : chiffre clair sur fond noir.

    cadrage "simple" : réduction directe en 28×28.
    cadrage "mnist"  : comme la préparation de MNIST (LeCun et al.) : fond ramené
    à 0, chiffre réduit pour tenir dans 20×20 en gardant ses proportions, puis
    placé de sorte que son centre de masse tombe au centre de l'image 28×28.
    """
    from PIL import Image
    im = np.asarray(Image.open(chemin).convert("L"), dtype=np.float64)
    bord = np.median(np.r_[im[0], im[-1], im[:, 0], im[:, -1]])
    if bord > 127:                                    # fond clair : on inverse
        im, bord = 255 - im, 255 - bord
    if cadrage == "simple":
        return np.asarray(Image.fromarray(im.astype(np.uint8)).resize((28, 28), Image.LANCZOS))

    im = np.clip(im - bord, 0, None)                  # fond à 0
    im = 255 * im / im.max()                          # contraste étiré jusqu'à 255
    lignes, cols = np.where(im > 0.1 * 255)           # boîte englobant le chiffre
    im = im[lignes.min():lignes.max() + 1, cols.min():cols.max() + 1]
    h, w = im.shape
    k = 20.0 / max(h, w)
    petit = Image.fromarray(im.astype(np.uint8)).resize(
        (max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS)
    petit = np.asarray(petit, dtype=np.float64)
    yy, xx = np.indices(petit.shape)                  # centre de masse du chiffre réduit
    cy, cx = (yy * petit).sum() / petit.sum(), (xx * petit).sum() / petit.sum()
    y0 = int(np.clip(round(14 - cy), 0, 28 - petit.shape[0]))
    x0 = int(np.clip(round(14 - cx), 0, 28 - petit.shape[1]))
    sortie = np.zeros((28, 28))
    sortie[y0:y0 + petit.shape[0], x0:x0 + petit.shape[1]] = petit
    return np.clip(np.round(sortie), 0, 255).astype(np.uint8)


def apercu(images, chemin):
    """Planche des images 28×28 telles qu'envoyées (agrandies ×8), pour vérifier le cadrage."""
    from PIL import Image
    planche = np.concatenate([np.pad(im, 1, constant_values=128) for _, im, *_ in images], axis=1)
    Image.fromarray(planche).resize((planche.shape[1] * 8, planche.shape[0] * 8), Image.NEAREST).save(chemin)
    print("Aperçu des images envoyées :", chemin)


# ---------------------------------------------------------------- API ESPHome

def cle_api(args):
    if args.key:
        return args.key
    if os.environ.get("CNN_TEST_API_KEY"):
        return os.environ["CNN_TEST_API_KEY"]
    if os.path.exists(SECRETS):
        m = re.search(r'^cnn_test_api_key:\s*"?([^"\s]+)"?', open(SECRETS, encoding="utf-8").read(), re.M)
        if m:
            return m.group(1)
    sys.exit("Clé API introuvable : variable CNN_TEST_API_KEY, option --key, ou cnn_test_api_key dans "
             "esp32/esphome/secrets.yaml (même valeur que dans le secrets.yaml de HA).")


async def peut_attendre(resultat):
    """Certaines versions d'aioesphomeapi rendent ces appels asynchrones, d'autres non."""
    if inspect.isawaitable(resultat):
        return await resultat
    return resultat


async def envoyer(args, images):
    from aioesphomeapi import APIClient

    client = APIClient(args.host, 6053, None, noise_psk=cle_api(args))
    await client.connect(login=True)
    entites, actions = await client.list_entities_services()
    action = next((a for a in actions if a.name == "classify"), None)
    if action is None:
        sys.exit("Action « classify » absente : la config de la carte a-t-elle bien le bloc api: actions: ?")
    cles = {}
    for e in entites:                                 # clé numérique de l'entité -> rôle
        for role, nom in NOMS.items():
            if e.name == nom:
                cles[e.key] = role
    if len(cles) < len(NOMS):
        print("Attention : entités trouvées :", sorted(cles.values()))

    etats, recu = {}, asyncio.Event()

    def sur_etat(etat):
        role = cles.get(etat.key)
        if role:
            etats[role] = etat.state
            if len(etats) == len(NOMS):
                recu.set()

    await peut_attendre(client.subscribe_states(sur_etat))
    await asyncio.sleep(1.0)                          # laisse passer l'envoi initial des états

    resultats = []
    for nom, image, etiquette, py_pred, py_conf in images:
        etats.clear()
        recu.clear()
        await peut_attendre(client.execute_service(action, {"image": [int(v) for v in image.ravel()]}))
        try:
            await asyncio.wait_for(recu.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            print("%-14s pas de réponse complète de la carte (reçu : %s)" % (nom, etats))
            continue
        esp = int(round(etats["prediction"]))
        resultats.append(esp == py_pred)
        print("%-14s étiquette %s  Python %d (%5.1f %%)  ESP %d (%5.1f %%, marge %5.1f %%)  %4.0f µs  %s"
              % (nom, "?" if etiquette is None else etiquette, py_pred, 100 * py_conf, esp,
                 etats["confiance"], etats["marge"], etats["temps"],
                 "ok" if esp == py_pred else "DIFFÉRENT"))
        await asyncio.sleep(args.pause)
    await client.disconnect()
    return resultats


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--host", default="cnn-test.local", help="nom ou adresse IP de la carte")
    p.add_argument("--key", default=None, help="clé de chiffrement de l'API (base64)")
    p.add_argument("--index", type=int, nargs="+", help="indices dans le jeu de test MNIST")
    p.add_argument("--n", type=int, default=10, help="nombre d'images tirées au hasard")
    p.add_argument("--graine", type=int, default=None)
    p.add_argument("--png", nargs="+", help="images à envoyer (converties en 28×28)")
    p.add_argument("--cadrage", choices=("mnist", "simple"), default="mnist",
                   help="préparation des images --png (défaut : comme MNIST)")
    p.add_argument("--apercu", default=None, help="enregistre une planche PNG des images envoyées")
    p.add_argument("--sans-carte", action="store_true", help="prédiction Python seulement, sans l'ESP")
    p.add_argument("--pause", type=float, default=0.5, help="secondes entre deux images")
    p.add_argument("--modele", default=MODELE)
    args = p.parse_args()

    couches, meta = charger_modele(args.modele)
    images = []
    if args.png:
        for chemin in args.png:
            im = image_png(chemin, args.cadrage)
            images.append((os.path.basename(chemin)[:14], im, None) + predire(couches, meta, im))
    else:
        img, lab = charger_u8()["test"]
        idx = args.index or list(np.random.default_rng(args.graine).choice(len(img), args.n, replace=False))
        for i in idx:
            images.append(("test[%d]" % i, img[i], int(lab[i])) + predire(couches, meta, img[i]))

    if args.apercu:
        apercu(images, args.apercu)
    if args.sans_carte:
        for nom, _, etiquette, py_pred, py_conf in images:
            print("%-14s étiquette %s  Python %d (%5.1f %%)" % (nom, "?" if etiquette is None else etiquette,
                                                             py_pred, 100 * py_conf))
        return 0
    print("Modèle Python : %s, carte : %s" % (meta["nom"], args.host))
    resultats = asyncio.run(envoyer(args, images))
    print("\n%d/%d prédictions identiques entre l'ESP et Python" % (sum(resultats), len(resultats)))
    return 0 if resultats and all(resultats) else 1


if __name__ == "__main__":
    sys.exit(main())
