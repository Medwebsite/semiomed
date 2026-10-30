# Sémio, l'IA médicale créée par YM

Sémio est un assistant médical gratuit, en français, en darija et en anglais. Il sert à réviser et à comprendre la médecine : cas cliniques, cours, mécanismes, QCM, lecture d'examens et recherche dans la littérature.

**Site : https://medwebsite.github.io/semiomed/**

> Outil d'information et d'apprentissage. Sémio peut se tromper : vérifie toujours auprès d'un professionnel de santé. Il ne remplace pas une consultation. En cas d'urgence, appelle le SAMU (141 au Maroc, 15 en France).

## Ce que fait le site

| Onglet | Rôle |
|---|---|
| **Discussion** | Questions libres. Sémio garde le fil de la conversation et répond dans la langue de la personne, darija comprise. |
| **Cas clinique** | Analyse d'une observation (hypothèses, examens, traitement), avec une relecture automatique pour limiter les erreurs et les oublis. |
| **Examens & imagerie** | Lecture de radios, scanners, IRM, échographies, ECG et bilans biologiques. Le type d'examen est détecté automatiquement. La lecture des ECG reste expérimentale. |
| **Cours** | Rédaction d'un cours sur un sujet, ou explication page par page de ton propre cours en PDF, avec les points clés et des QCM. |
| **Mécanisme** | Explication d'une physiopathologie étape par étape. |
| **Recherche** | Recherche de vrais articles sur PubMed, puis synthèse en français avec les références. |
| **Ma bibliothèque** | Livres en PDF ajoutés par l'utilisateur. Ils restent dans son navigateur et ne sont jamais envoyés. Une bibliothèque commune de manuels libres de droits est aussi disponible. |

## Les moteurs

| Moteur | Où il tourne | Usage |
|---|---|---|
| **Sémio-Pro** (Gemma 4 26B) | Serveur Cloudflare Workers AI | Moteur par défaut : le plus précis, et il lit les images. |
| **Sémio-2b** (Mistral-7B + LoRA) | Serveur Cloudflare Workers AI | Modèle entraîné par YM, spécialisé dans les QCM. |
| **Sémio-1** (Gemma 4 E2B affiné) | Navigateur de l'utilisateur | Mode hors ligne, plus lent. |

Aucune clé d'API n'apparaît dans le site. Les requêtes passent par un petit serveur (Worker Cloudflare) qui accepte seulement les appels venant du site.

### Scores mesurés sur des QCM jamais vus

| Test | Sémio-Pro | Sémio-2b |
|---|---|---|
| ECN (MediQAl) | environ 81 % | 70 % |
| FMPC / EDN | 92 % | 48 % |
| Darija | 99 % | 30 % |

Les échantillons vont de 60 à 170 questions selon le test et le moteur.

## Fichiers du dépôt

- `index.html` : tout le site (interface, onglets, appels aux moteurs).
- `coi-serviceworker.js` : active l'isolation nécessaire au mode hors ligne (Sémio-1).
- `training/semio2_train.py` : entraînement de Sémio-2.
- `training/semio2b_train.py` : entraînement de Sémio-2b.
- `training/semio1_finetune.ipynb` : entraînement de Sémio-1.

## Modèles et données

- Sémio-1 : [Medwebsite/semio-1-E2B-GGUF](https://huggingface.co/Medwebsite/semio-1-E2B-GGUF)
- Sémio-2b : [Medwebsite/semio-2b-mistral7b-lora](https://huggingface.co/Medwebsite/semio-2b-mistral7b-lora)
- Bibliothèque commune : [Medwebsite/semio-bibliotheque](https://huggingface.co/datasets/Medwebsite/semio-bibliotheque) (OpenStax Anatomy & Physiology 2e, CC BY 4.0)

Données d'entraînement, avec nos remerciements :
- [MediQAl](https://huggingface.co/datasets/ANR-MALADES/MediQAl) (CC BY 4.0)
- [FrenchMedMCQA](https://huggingface.co/datasets/uy-rrodriguez/FrenchMedMCQA-extended) (Apache 2.0)

Les sources complètes de Sémio-2b sont indiquées sur sa page Hugging Face.

Modèles de base : Gemma 4 (Google), Mistral-7B-Instruct-v0.2 (Mistral AI).

## Licence et usage

Projet pédagogique et gratuit. Sémio-2b ne peut pas être utilisé commercialement, à cause de la licence d'une partie de ses données d'entraînement.
