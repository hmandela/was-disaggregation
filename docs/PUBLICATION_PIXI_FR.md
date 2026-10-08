# Dépôt GitHub, release et PyPI : guide pas à pas

Ce guide concerne **WAS Disaggregation 0.8.0**, nom PyPI `was-disaggregation`,
import Python `was_disaggregation`, licence GPL-3.0-only. Le dépôt visé est
`hmandela/was-disaggregation`.

Le kit contient deux dossiers voisins :

```text
was-disaggregation/          code, tests, documentation, .git, commit, tag v0.8.0
release-assets/v0.8.0/       wheel, sdist, notes, sommes SHA-256
```

Le dépôt est **déjà initialisé** : ne refais pas `git init`, le commit ou le
tag. Les archives construites restent hors du dépôt. Les environnements Pixi,
sorties et caches sont exclus de Git.

Le même `pixi.lock` prépare quatre plateformes : Linux x86-64 (`linux-64`),
Windows x64 (`win-64`), macOS Intel (`osx-64`) et macOS Apple Silicon
(`osx-arm64`). Le code est testé localement sous Linux; la matrice GitHub
Actions vérifie les quatre systèmes après la première poussée. Attends que
les quatre jobs soient verts avant la publication publique.

## 1. Extraire et inspecter

Sous Linux ou macOS, dans un terminal :

```bash
unzip was-disaggregation-github-ready.zip -d was-release
cd was-release/was-disaggregation
git status --short --branch
git log -1 --oneline
git tag --list
git ls-files | wc -l
```

Sous Windows, dans PowerShell :

```powershell
Expand-Archive .\was-disaggregation-github-ready.zip -DestinationPath .\was-release
Set-Location .\was-release\was-disaggregation
git status --short --branch
git log -1 --oneline
git tag --list
git ls-files | Measure-Object -Line
```

Tu dois voir `main` propre, le commit « Prepare WAS Disaggregation 0.8.0 for
release », le tag `v0.8.0` et **35 fichiers suivis**. Regarde `.gitignore` :
`.pixi/`, `dist/`, données NetCDF/GRIB, sorties et jetons n'appartiennent pas
au dépôt. N'utilise pas `git add -f` pour y forcer ces fichiers.

## 2. Installer les outils et vérifier le package

```bash
git --version
pixi --version
gh --version
```

Si Pixi est absent, l'installation officielle sur Linux ou macOS est :

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Sous Windows PowerShell, suis [l'installation officielle](https://pixi.sh/latest/installation/)
ou utilise :

```powershell
powershell -ExecutionPolicy Bypass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

Rouvre le terminal. Si `gh` est absent, suis les instructions officielles de
[GitHub CLI](https://cli.github.com/). Depuis `was-release/was-disaggregation` :

```bash
pixi install
pixi run release-check
git status --short
```

Ces commandes Pixi et Git fonctionnent aussi dans PowerShell. `pixi install`
recrée `.pixi/` depuis `pixi.lock` pour ton système. Ne lance pas `pixi init` :
le manifeste est déjà dans `pyproject.toml`. `release-check` lance les tests
(116 réussis sous Linux, sept tests PyTorch optionnels ignorés dans l'environnement
standard), le contrôle Python, construit les deux distributions, lance
`twine check --strict` puis vérifie le wheel installé hors des sources :

```text
dist/was_disaggregation-0.8.0-py3-none-any.whl
dist/was_disaggregation-0.8.0.tar.gz
```

`git status --short` doit rester vide. Les fichiers de
`../release-assets/v0.8.0/` ont été construits depuis le même commit; leurs
empreintes se trouvent dans `SHA256SUMS.txt`.

Sous Windows et macOS, `HindcastExperiment.run(n_jobs>1)` calcule les plis
séquentiellement avec un avertissement. Sous Linux, les plis peuvent utiliser
`fork`. Le calcul reste disponible sur les trois systèmes.

## 3. Connecter GitHub CLI

```bash
gh auth login
gh auth status
```

Choisis GitHub.com, HTTPS et la connexion par navigateur. Sur un serveur sans
navigateur, `gh` peut afficher un code et une page de validation à ouvrir sur
ton propre appareil. Vérifie que le compte connecté est **hmandela**. Ne colle
aucun mot de passe, code temporaire ou jeton dans le dépôt ou dans un message.

## 4. Créer le dépôt public et pousser la branche et le tag

Dans le dossier du projet :

```bash
gh repo create hmandela/was-disaggregation \
  --public --description "Forecast-conditioned stochastic daily weather generation" \
  --source=. --remote=origin --push
git push origin v0.8.0
git status --short --branch
gh repo view hmandela/was-disaggregation
```

La première commande crée le dépôt public, ajoute `origin` et pousse `main`.
La deuxième pousse le tag annoté existant. `gh repo view` vérifie le résultat.
Sous PowerShell, exécute la commande `gh repo create` sur une seule ligne
(sans les barres obliques de continuation Bash).

Ouvre ensuite **Actions → Package checks** sur GitHub et attends la réussite
des quatre jobs : Linux, Windows, macOS Intel et macOS Apple Silicon. Une
installation ou un test échoué sur une plateforme doit être corrigé avant
de publier `v0.8.0` sur GitHub et PyPI. Le tag existant peut être révisé tant
qu'il n'a pas été publié; consulte le guide GitHub si tu dois le remplacer.

**Alternative par le site GitHub** : crée « New repository », propriétaire
`hmandela`, nom `was-disaggregation`, visibilité publique. Ne coche pas
l'initialisation avec README, `.gitignore` ou licence : ces fichiers existent
déjà localement. Puis exécute, depuis le projet :

```bash
git remote add origin https://github.com/hmandela/was-disaggregation.git
git push -u origin main
git push origin v0.8.0
```

Utilise une seule des deux voies. Si `origin` existe, regarde `git remote -v`
avant de l'ajouter de nouveau.

## 5. Créer puis publier la release GitHub

Après avoir poussé le tag, crée un **brouillon** avec les deux fichiers du kit :

```bash
gh release create v0.8.0 \
  ../release-assets/v0.8.0/was_disaggregation-0.8.0-py3-none-any.whl \
  ../release-assets/v0.8.0/was_disaggregation-0.8.0.tar.gz \
  --verify-tag --draft --title "WAS Disaggregation v0.8.0" \
  --notes-file ../release-assets/v0.8.0/RELEASE_NOTES.md
gh release view v0.8.0
```

Vérifie sur GitHub le texte et la présence des deux fichiers, puis publie :

```bash
gh release edit v0.8.0 --draft=false
gh release view v0.8.0
```

Sous PowerShell, mets la longue commande `gh release create` sur une seule
ligne et utilise les chemins `..\release-assets\v0.8.0\...`.

La release GitHub et la publication PyPI sont deux opérations distinctes.

## 6. Faire un essai sur TestPyPI avec Pixi

Crée ou ouvre ton compte sur <https://test.pypi.org/>, vérifie l'e-mail et
crée un jeton API. Pour un projet inexistant, le premier jeton doit permettre
sa création (portée « Entire account »). Dans le dossier du projet :

```bash
pixi run publish-testpypi
```

La tâche refait les contrôles. Twine demande le jeton **TestPyPI**, avec
`__token__` comme nom d'utilisateur; la saisie est masquée. Ne mets pas le
jeton dans `pyproject.toml`, `pixi.lock`, Git ou l'historique du shell.

Teste ensuite l'installation dans un environnement vierge. Les dépendances
viennent de PyPI, puis le package seul de TestPyPI. L'option `-I` vérifie
l'import installé sans prendre le code source du dossier courant :

```bash
pixi run python -m venv .venv-testpypi
.venv-testpypi/bin/python -m pip install numpy scipy pandas xarray netCDF4 'dask[array]'
.venv-testpypi/bin/python -m pip install \
  --index-url https://test.pypi.org/simple/ --no-deps 'was-disaggregation==0.8.0'
.venv-testpypi/bin/python -I -c "import was_disaggregation as w; print(w.__version__)"
.venv-testpypi/bin/was-disaggregation --version
```

Sous Windows PowerShell, remplace ces commandes par :

```powershell
pixi run python -m venv .venv-testpypi
& .\.venv-testpypi\Scripts\python.exe -m pip install numpy scipy pandas xarray netCDF4 'dask[array]'
& .\.venv-testpypi\Scripts\python.exe -m pip install --index-url https://test.pypi.org/simple/ --no-deps 'was-disaggregation==0.8.0'
& .\.venv-testpypi\Scripts\python.exe -I -c "import was_disaggregation as w; print(w.__version__)"
& .\.venv-testpypi\Scripts\was-disaggregation.exe --version
```

Les comptes et jetons TestPyPI et PyPI sont distincts.

## 7. Publier sur le vrai PyPI et vérifier

Crée ou ouvre ton compte sur <https://pypi.org/>, vérifie l'e-mail et crée
un jeton API. Le premier jeton doit pouvoir créer ce projet. Puis :

```bash
pixi run publish-pypi
pixi run python -m venv .venv-pypi
.venv-pypi/bin/python -m pip install 'was-disaggregation==0.8.0'
.venv-pypi/bin/python -I -c "import was_disaggregation as w; print(w.__version__)"
.venv-pypi/bin/was-disaggregation --help
```

Sous Windows PowerShell :

```powershell
pixi run publish-pypi
pixi run python -m venv .venv-pypi
& .\.venv-pypi\Scripts\python.exe -m pip install 'was-disaggregation==0.8.0'
& .\.venv-pypi\Scripts\python.exe -I -c "import was_disaggregation as w; print(w.__version__)"
& .\.venv-pypi\Scripts\was-disaggregation.exe --help
```

Saisis ici le jeton **PyPI**, différent de celui de TestPyPI. Une fois le projet
créé, les versions suivantes peuvent utiliser un jeton limité à ce projet.
PyPI n'autorise pas le remplacement d'un fichier sous le même nom et la même
version.

## 8. Faire évoluer le projet

Garde l'historique Git et `pixi.lock`. Modifie le code et les tests; incrémente
`__version__` dans `was_disaggregation/_version.py` (par exemple `0.8.1`);
mets à jour `CHANGELOG.md`; relance `pixi run release-check`; puis commite,
crée un **nouveau** tag et répète la publication GitHub, TestPyPI et PyPI.
Ne republie pas 0.8.0 sous le même nom.

Références officielles : [Pixi](https://pixi.sh/latest/installation/),
[connexion GitHub CLI](https://cli.github.com/manual/gh_auth_login),
[création du dépôt](https://cli.github.com/manual/gh_repo_create),
[création de la release](https://cli.github.com/manual/gh_release_create) et
[guide PyPA](https://packaging.python.org/en/latest/tutorials/packaging-projects/).
