# Sémio : IA médicale locale

Le site fait tourner **Gemma 4** (modèle open source de Google) directement dans le navigateur, grâce à Transformers.js et WebGPU. Il n'y a pas d'API, pas de clé et pas de serveur.

## Fichiers
- `index.html` : l'interface et les 4 modules (Cas clinique, Cours, Mécanisme, Recherche)
- `worker.js` : charge le modèle et génère les réponses, en arrière-plan

## Mettre en ligne sur GitHub Pages (gratuit)
1. Crée un nouveau dépôt sur GitHub, par exemple `semio`.
2. Clique sur **Add file → Upload files** et dépose `index.html` et `worker.js`, puis valide avec **Commit**.
3. Va dans **Settings → Pages**, choisis **Source : Deploy from a branch**, puis **Branch : main / (root)**, et clique sur **Save**.
4. Une à deux minutes plus tard, le site est en ligne sur `https://<ton-compte>.github.io/semio/`.

## Tester sur ton PC avant de le mettre en ligne
Le site ne fonctionne pas si tu ouvres le fichier en double-cliquant dessus, car le worker a besoin d'un serveur. Dans le dossier du site, lance :
```
python -m http.server 8000
```
puis ouvre http://localhost:8000 dans Chrome ou Edge.

## Ce qu'il faut pour l'utiliser
- Un ordinateur avec **Chrome ou Edge à jour** (WebGPU).
- Une carte graphique récente aide beaucoup. Avec un GPU intégré, ça marche, mais plus lentement.
- Le premier chargement télécharge le modèle (plusieurs Go). Il est ensuite gardé en cache.
- Le module Recherche a besoin d'Internet (Europe PMC). Les trois autres modules fonctionnent hors ligne une fois le modèle chargé.

## Modèles
- **Gemma 4 E2B** : le plus léger et le plus rapide.
- **Gemma 4 E4B** : plus précis, mais plus lourd.
