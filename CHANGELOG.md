# Changelog

## 0.8.0 — 2026-10-08

Première version préparée pour la publication de `was-disaggregation`.

- Corrige l'assemblage Python, ajoute Pixi, les commandes de vérification et la licence GPL-3.0-only.
- Harmonise les calendriers saisonniers et l'exclusion du 29 février.
- Préserve les valeurs manquantes et les masques propres à chaque classe/membre.
- Corrige les probabilités, l'occurrence, les traces pluvieuses et plusieurs contrôles de validation.
- Rend les sorties Dask reproductibles entre tuiles et lots de membres.
- Vérifie les méthodes paramétriques, par rééchantillonnage, GLM/dynamiques et neuronales avec des tests ciblés.
- Verrouille les environnements Pixi pour Linux x86-64, Windows x64 et macOS Intel/Apple Silicon; ajoute les contrôles GitHub Actions pour ces quatre cibles.
- Permet les hindcasts sous Windows et macOS avec exécution séquentielle lorsque `n_jobs>1`.

La fiabilité des prévisions régionales exige une vérification sur hindcasts indépendants; voir [l'audit scientifique](docs/AUDIT_FR.md).
