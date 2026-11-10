# Sécurité de Sémio

## Signaler une faille

Si tu trouves une faille de sécurité dans Sémio (site, serveur ou comptes),
**ne la publie pas** dans une issue publique.

Utilise le signalement privé de GitHub : onglet **Security** du dépôt →
**Report a vulnerability**. Décris le problème, comment le reproduire et son
impact possible. Une réponse est donnée dès que possible.

## Ce qui est couvert

- Le site `semiomed.pages.dev` (et `medwebsite.github.io/semiomed`)
- Le serveur d'API Sémio (Cloudflare Workers)
- Les comptes et l'historique des utilisateurs

## Mesures en place

- Vérification anti-robot (Cloudflare Turnstile) avant toute question et à la connexion
- Limite du nombre de requêtes par personne et par minute
- Clés et secrets stockés uniquement côté serveur, jamais dans le code du site
- Accès aux données limité à chaque utilisateur (Row Level Security)
- En-têtes de sécurité HTTP (HTTPS obligatoire, pas d'affichage dans un autre site)

Merci de ne pas tester la sécurité en surchargeant le service : les quotas sont
partagés par tous les utilisateurs.
