# captchaOpener

Projet cassage de captcha pour un challenge root-me.org.

Beaucoup de traitement d'image et un petit peu de prédiction par un réseau de neurones.

Etapes:
- On cherche d'abord à séparer tout les lettres du captcha via différentes méthode de traitement d'image.
- On vient ensuite soumettre les lettres individuellement à un réseau de neuronne qui a subit un apprentissage supervisé.

Le programme final est rootMeChall.py qui effectue de nombreuses tentatives car le pourcentages de cassage "complet" est assez bas.
Problèmes de différenciation o/O i/I/l. Problèmes de repérage lié à la variabilité de la taille des caractères des captcha proposés par le générateur automatique.
