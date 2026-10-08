# Audit de cohérence — WAS Disaggregation 0.8.0

Audit réalisé le 8 octobre 2026 sur l'archive `was_disaggregation.zip` fournie
par Mandela HOUNGNIBO. La version 0.8.0 est conservée pour cette préparation de
première publication; le nom de distribution choisi est `was-disaggregation`,
avec la licence GPL-3.0-only. L'import reste `was_disaggregation`.

## Verdict et portée

L'archive initiale ne pouvait pas produire une distribution correctement
installable : elle déclarait un dossier `src` inexistant et une commande visant
un module `was_weather` absent. L'assemblage corrigé construit un wheel et un
sdist, réussit la validation Twine et fonctionne après installation hors du
dossier source. Les erreurs de calcul confirmées ci-dessous sont corrigées et
couvertes par des régressions.

Le package reste un logiciel de recherche en développement. Cet audit examine
le code, les interfaces, les calendriers, les formes de tableaux, la séparation
apprentissage/validation, des cas numériques et des exemples synthétiques.
L'archive fournie ne contient pas les observations ni les prévisions réelles;
aucune compétence opérationnelle sur le Bénin, le Sahel ou l'Afrique de l'Ouest
n'a donc été mesurée. Les tests ne certifient pas une reproduction exhaustive
des articles cités dans les docstrings.

## Assemblage et publication

| Défaut initial | Effet | Correction |
| --- | --- | --- |
| Découverte setuptools sous `src`, absent | Construction impossible ou package absent du wheel | Découverte explicite du dossier plat `was_disaggregation` |
| Commande vers `was_weather.cli` | Échec de la commande installée | `was-disaggregation` et alias `was-weather` vers `was_disaggregation.cli` |
| Nom/version dispersés dans les fichiers | Diagnostics et instructions incohérents | Nom choisi harmonisé; `_version.py` devient la source unique de version |
| README, licence et environnement de publication absents | Documentation PyPI pauvre et préparation manuelle | README, GPLv3, manifeste Pixi, verrou de dépendances, guide français |
| Aucune suite de tests fournie | Régressions non détectées | Suites ciblées par famille plus tests de CLI et de tuilage |

Le script `scripts/release.py` reconstruit proprement les distributions, vérifie
la présence de tous les modules et de la licence, utilise
`twine check --strict`, puis installe le wheel dans un environnement temporaire.
L'import doit provenir de cet environnement, hors des sources. Les dépendances
scientifiques de l'environnement Pixi sont réutilisées pour ce contrôle : il
vérifie le wheel, sans refaire une résolution indépendante des dépendances.

L'envoi utilise Twine à travers les tâches Pixi et sélectionne exactement le
wheel et le sdist de la version courante. Aucun identifiant privé n'est inclus.
La publication distante n'a pas été effectuée dans cette session.
Le dépôt Git suit les sources et la configuration; les fichiers construits, caches
et environnements locaux sont exclus par `.gitignore`.

## Cohérence des entrées et du générateur paramétrique

| Module | Défaut confirmé / risque concret | Résultat de la correction |
| --- | --- | --- |
| `data.py` | `.T` d'une DataArray désignait sa transposée au lieu du temps | Accès explicite à la coordonnée `['T']` |
| `conditioning.py` | Probabilités en pourcentage admises pour les poids mais non normalisées pour la densité | PDF-ratio cohérent avec fractions et pourcentages |
| `conditioning.py` / `model.py` | Repli climatologique perdu en PDF-ratio lorsque des classes historiques étaient vides | Respect explicite de `empty_policy` et poids complets valides |
| `rainfall.py` | Troncature des pluies de trace réduisait leur moyenne ajustée | Loi bornée préservant la moyenne et moments cohérents |
| `rainfall.py` | Séries totalement sèches/humides et runs censurés retombaient sur une fréquence artificielle | Repli fondé sur états/transitions observés; prior nul traité correctement |
| `model.py` | `class_draw='fitted'` en spatial indépendant partageait le même uniforme | Tirages de classes indépendants entre cellules |
| `model.py` / `multivariate.py` / `rainfall.py` | Une variable ou un mois de pluie manquant dans une classe pouvait masquer toutes les classes | Validité appliquée aux membres/classes concernés, y compris avec un cumul JAS et une génération mai-octobre |
| `model.py` / `spatial.py` | Options incompatibles ou invalides parfois ignorées | Erreurs explicites; cohérence des méthodes de dépendance et des mois |
| `scalable.py` | Classes de mélange perdues dans la sortie Dask | Conservation du diagnostic entier `tercile_class` |
| `cli.py` | `--total-window` seul était ignoré; choix du premier tableau multidimensionnel pour une contrainte | Fenêtre appliquée; variable de probabilités identifiée et validée |

La suite vérifie notamment l'invariance des réalisations entre tuiles et lots
de membres, avec innovations indépendantes ou noyaux spatiaux partagés.
Les étiquettes de classe restent entières dans les NetCDF produits par la CLI.

## Attributs, rééchantillonnage et vérification

| Module | Défaut confirmé / risque concret | Résultat de la correction |
| --- | --- | --- |
| `attributes.py` | Calendrier incompatible avec l'exclusion du 29 février par les générateurs | Une politique commune sans 29 février, y compris pour les fenêtres |
| `attributes.py` | Fenêtres d'onset/dry-spell et cessation interannuelle mal bornées | Bornes/look-ahead contrôlés et traitement des fenêtres passant au nouvel an |
| `mre.py` | Tolérances nulles, tableaux mal formés ou non finis, indicateurs de convergence obsolètes | Validation des entrées, du support et de la convergence finale |
| `mre.py` | Une contrainte onset sans probabilités reprenait les terciles de pluie totale | `probabilities=None` réservé à `SeasonalTotal` |
| `nonparametric.py` | Certains priors ne laissaient aucun donneur; limite arbitraire de tentatives | Support vérifié et tirage sur donneurs effectivement disponibles |
| `diagnostics.py` | Seuils et membres comparés après aplatissement de grilles ordonnées différemment | Contrôle explicite de l'égalité des coordonnées |
| `diagnostics.py` / `nonparametric.py` | Champs partiellement manquants admissibles comme membres/donneurs | Éligibilité fondée sur les données requises complètes |
| `validation.py` | Données manquantes classées sèches ou probabilités artificielles | Masques et dénominateurs fondés sur les échantillons valides |
| `validation.py` | Score spatial échouant sur une seule cellule; NaN remplacés par zéro dans les corrélations | Cas mono-cellule pris en charge, corrélations sur paires valides |
| `validation.py` | CRPS avec allocation quadratique en nombre de membres | Calcul exact à partir des échantillons triés, sans tableau M×M×sites |

Les terciles de la prévision doivent correspondre à la période de référence
du producteur. Cette référence fixe n'est pas interchangeable avec les années
d'apprentissage d'un pli de validation. Les transformations ajustées sur les
données et les modèles doivent, eux, exclure les années testées.

Un contrôle numérique indépendant des solveurs sur 20 petits problèmes convexes
par méthode, comparés à SLSQP, donne un écart maximal d'objectif de
4,03×10⁻⁹ pour MRE et 3,76×10⁻¹⁰ pour Croley. Les tests NHMM/NHSMM comparent
les probabilités forward/backward à une énumération exhaustive de trajectoires.
Ces contrôles portent sur les formulations effectivement codées; ils ne changent
pas les limites d'interprétation scientifique décrites ci-dessous.

## GLM, modèles dynamiques et réseaux neuronaux

| Module | Défaut confirmé / risque concret | Résultat de la correction |
| --- | --- | --- |
| `glm.py` | Harmoniques de dates mal alignées entre années | Dates/harmoniques calculées sur les véritables saisons |
| `glm.py` / `dynamical.py` | Acceptation implicite de grilles/predictors non alignés | Contrôles des coordonnées et de la disponibilité des prédicteurs |
| `dynamical.py` | Blocs saisonniers mal alignés et paramètres Gamma non finis pour séries sèches | Alignement sans 29 février; initialisation sèche valide |
| `dynamical.py` | `amounts=False` produisait encore des montants Gamma | Sortie d'occurrence cohérente avec l'option |
| `dynamical.py` | Calibrations manquantes devenaient de la pluie nulle; rangs incomplets ambigus | NaN préservés, préconditions ECC explicites |
| `dynamical.py` | Demande de plus de membres que de templates disponible | Validation/support des templates et traitement explicite |
| `dynamical.py` | Postérieurs retournés différents des paramètres après dernière mise à jour | Recalcul final correspondant aux paramètres retournés |
| `generative.py` | Une année réservée à validation pouvait revenir dans l'apprentissage | Années distinctes/triées et absence de repli fuyant |
| `generative.py` | Coordonnées membres/temps/grilles perdues lors de conversions NumPy | Validation avant conversion et préservation du masque d'observation |
| `generative.py` | Perfect prognosis sans prédicteurs observés, correction modèle appliquée indûment | Voie de prédicteurs observés et correction modèle désactivée |
| `generative.py` | Diffusion/flow sur données constantes avec échelle nulle; AR1 négatif incorrect | Plancher numérique et bruit temporel corrigé |

L'import du package fonctionne sans PyTorch. Les tests neuronaux sur CPU
exercent les méthodes `crps`, `cgan`, `diffusion`, `flow`, leurs sorties et les
checkpoints. Une époque sur des données synthétiques contrôle le fonctionnement;
elle ne mesure ni convergence utile, ni fiabilité probabiliste.

## Limites scientifiques conservées et explicitées

1. **Conditionnement des terciles.** Mélanger ou moyenner des paramètres de
   pluie n'impose pas exactement les fréquences des cumuls saisonniers générés.
   Les classes de paramètres ne sont pas les classes finales des cumuls.
2. **Variance du mélange.** Le réglage automatique utilise des moments mensuels
   approchés; les transitions entre mois et l'occurrence semi-Markov doivent être
   évaluées empiriquement. Il ne garantit pas une égalité exacte de variance.
3. **Contraintes MRE/Croley.** Les pénalités portent sur les poids historiques.
   Des contraintes incompatibles et un faible nombre d'années effectives peuvent
   empêcher l'atteinte des probabilités demandées. Inspecter les diagnostics.
4. **Dépendance spatiale.** Les noyaux stationnaires et la copule gaussienne par
   approximation spectrale ne garantissent pas la reproduction des extrêmes
   conjoints, des structures convectives ou d'une non-stationnarité régionale.
5. **GLM/BayGEN.** La couche GP réalise un lissage empirique bayésien de
   coefficients; elle ne constitue pas une inférence hiérarchique bayésienne
   complète avec dépendance entre tous les coefficients.
6. **NHMM/NHSMM.** La mise à jour du risque NHSMM est une approximation EM;
   la monotonie stricte de vraisemblance n'est pas garantie. Les fréquences LOCI
   restent approximatives lorsque les données présentent des ex æquo.
7. **Réseaux.** Architecture, scores et conservation de masque sont vérifiés,
   mais aucune méthode neuronale n'est déclarée calibrée ou supérieure aux
   générateurs statistiques sans hindcasts comparatifs.
8. **Calendriers et plateformes.** Calendrier grégorien avec omission du
   29 février; pas de traitement natif 360_day. Pixi verrouille Linux x86-64,
   Windows x64 et macOS Intel/Apple Silicon avec Python 3.12. Les tests ont été
   exécutés localement sous Linux; les autres plateformes attendent les jobs
   GitHub Actions. La voie parallèle LOYO utilise `fork` sous Linux et calcule
   séquentiellement les plis sous Windows/macOS lorsque `n_jobs>1`.

## Validation réalisée et suite conseillée

La suite se relance avec `pixi run release-check`. Sont vérifiés : régressions
par famille, tuilage/lots,
lecture/écriture NetCDF, import optionnel, CPU PyTorch, wheel/sdist, métadonnées
Twine et commandes installées.

Avant une utilisation opérationnelle, conduire des hindcasts indépendants sur
tes propres données, avec masque terre et cellules sèches clairement définis.
Comparer climatologie, paramétrique, analogues et modèles plus complexes sur
cumuls/terciles, CRPS/RPSS et fiabilité, fréquence humide, longueurs de séquences,
quantiles extrêmes, corrélations temporelles/spatiales et cohérence
TMIN≤TMAX/HUMIN≤HUMAX. Une comparaison du mode `mean`, du mélange et des méthodes
de rééchantillonnage permettra de quantifier la dispersion saisonnière obtenue.
