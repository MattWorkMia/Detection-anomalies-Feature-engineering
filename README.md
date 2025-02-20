## 1. Interprétabilité et Explicabilité du Modèle

Dans cette première partie de l'assignation, nous travaillerons sur l'interprétabilité du modèle et son explicabilité. Nous utiliserons un ensemble de données sur les réadmissions à l'hôpital (`hospital.csv`), qui nous aidera à comprendre l'importance de l'interprétation de l'interaction de notre modèle avec les caractéristiques présentes dans notre ensemble de données. Il nous aidera également à apprendre comment expliquer les prédictions de nos modèles.

### 1.1 Chargement des Données

- Compléter `hw5.py:encode_target_column()`
- Compléter les exécutions de code dans `hw5.ipynb`

Nous commencerons par charger nos données et y jeter un coup d'œil rapide. Dans cette section, vous n'aurez pas grand-chose à faire, vous devrez seulement compléter la méthode `encode_target_column()`. Utilisez le `LabelEncoder` de scikit-learn pour compléter cette méthode.

### 1.2 Interprétabilité du Modèle

- Compléter `hw5.py:train_random_forest()`
- Compléter `hw5.py:evaluate_model()`
- Compléter les exécutions de code dans `hw5.ipynb`

Dans cette section, nous commencerons par entraîner un simple classificateur de forêt aléatoire et évaluer sa performance sur l'ensemble de validation. Vous devrez créer deux fonctions `train_random_forest()` et `evaluate_model()`. La première fonction nécessitera d'utiliser le `RandomForestClassifier` de scikit-learn et d'implémenter le processus d'entraînement dans une seule méthode. La méthode `evaluate_model()` devra renvoyer l'exactitude de notre modèle et le rapport de classification. Ici, vous devrez utiliser deux méthodes de `sklearn.metrics` qui ont déjà été importées dans `hw5.py`.

#### 1.2.1 Importance des Caractéristiques

- Compléter `hw5.py:calculate_permutation_importance()`
- Compléter les exécutions de code dans `hw5.ipynb`
- Répondre à la question dans `hw5.ipynb`

Dans cette section, vous apprendrez à utiliser une technique appelée `Importance de Permutation`, qui nous aidera à comprendre l'importance des caractéristiques de notre ensemble de données lorsqu'elles interagissent avec un modèle spécifique. Cette technique bénéficie d'être indépendante du modèle et peut être calculée de nombreuses fois avec différentes permutations de la caractéristique (Avertissement : les caractéristiques considérées comme ayant peu d'importance pour un mauvais modèle pourraient être très importantes pour un bon modèle). Vous trouverez plus d'informations sur cette technique [ici](https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html). Ici, vous compléterez une méthode appelée `calculate_permutation_importance()` et nous utiliserons la méthode `PermutationImportance` de la bibliothèque `eli5` pour le faire.

Vous devrez répondre à la question présentée dans cette section.

#### 1.2.2 Graphique de Dépendance Partielle

- Compléter `hw5.py:plot_partial_dependence()`
- Compléter `hw5.py:plot_mean_readmission_vs_time()`
- Compléter les exécutions de code dans `hw5.ipynb`
- Compléter les graphiques dans `hw5.ipynb`

Nous utiliserons également un type de visualisation appelé `Graphique de Dépendance Partielle`. Le Graphique de Dépendance Partielle (GDP) est une visualisation assez intuitive et facile à comprendre de l'impact des caractéristiques sur la variable de résultat prédite. Si les hypothèses du GDP sont remplies, il peut montrer comment une caractéristique influence une variable de résultat. Vous trouverez plus d'informations sur les graphiques PD [ici](https://slds-lmu.github.io/iml_methods_limitations/pdp.html). Ici, vous compléterez une méthode appelée `plot_partial_dependence()` et pour la compléter, vous utiliserez `PartialDependenceDisplay` de scikit-learn pour créer cette visualisation.

Enfin, dans le scénario hypothétique proposé, nos parties prenantes vous demanderont de vérifier que les données d'une caractéristique spécifique sont correctes. Par conséquent, elles nous demanderont de produire un graphique des "moyennes des réadmissions par rapport au temps". Ici, vous devrez compléter la méthode `plot_mean_readmission_vs_time()`, qui nous aidera à obtenir cette visualisation.

### 1.3 Explicabilité du Modèle (Bonus +5)

#### 1.3.1 Valeurs SHAP

- Compléter `hw5.py:main_factors()`
- Compléter les exécutions de code dans `hw5.ipynb`
- Compléter le graphique dans `hw5.ipynb`
- Répondre à la question dans `hw5.ipynb`

Maintenant que nous avons appris un peu sur l'interprétabilité du modèle, nous allons apprendre l'explicabilité du modèle. Dans cette section, nous apprendrons comment expliquer la prédiction d'un modèle sur un exemple unique. Pour ce faire, nous utiliserons les valeurs SHAP (un acronyme de SHapley Additive exPlanations) pour décomposer une prédiction et montrer l'impact de chaque caractéristique. C'est plutôt pratique, car elles nous permettent d'identifier visuellement quelles caractéristiques (et dans quelle mesure) soutiennent la prédiction et lesquelles la diminuent. Pour utiliser les valeurs SHAP, vous implémenterez une méthode appelée `main_factors()` avec l'aide de la bibliothèque `shap`. Vous devrez lire la documentation de la bibliothèque pour apprendre comment produire la visualisation souhaitée pour un exemple unique.

Vous devrez répondre à la question présentée dans cette section.

## Partie 2. Détection et Suppression des Valeurs Aberrantes + Sélection et Ingénierie des Caractéristiques

### 2.1 Chargement des Données

- Compléter les exécutions de code dans `hw5.ipynb`

Dans cette section, vous n'avez rien à compléter. Nous allons simplement charger l'ensemble de données de prédiction des tarifs de taxi de New York (`ny_taxi.csv`). Cet ensemble de données nous aidera à relier la technique d'importance de permutation à la sélection et à l'ingénierie des caractéristiques. Vous devez simplement exécuter les cellules de code dans `hw5.ipynb`.

### 2.2 Gestion des Valeurs Aberrantes et Sélection des Caractéristiques

- Compléter `hw5.py:remove_outliers_iqr()`
- Compléter les exécutions de code dans `hw5.ipynb`
- Répondre aux questions dans `hw5.ipynb`

Dans cette section, nous explorerons une méthode de détection des valeurs aberrantes connue sous le nom de méthode IQR (Interquartile Range). La méthode IQR définit les valeurs aberrantes comme des points de données qui se trouvent en dessous de Q1 - 1,5 * IQR ou au-dessus de Q3 + 1,5 * IQR, où Q1 et Q3 sont les 25e et 75e percentiles, respectivement. Votre tâche est de compléter la méthode `remove_outliers_iqr()` qui se trouve dans `hw5.py`. Considérations :

- Votre fonction doit mettre en œuvre la méthode IQR pour détecter et supprimer les lignes avec des valeurs aberrantes.
- Étant donnée une liste de caractéristiques présélectionnées, votre fonction doit vérifier chaque colonne à la recherche de valeurs aberrantes et renvoyer un nouvel ensemble de données exempt de valeurs aberrantes.

Après cela, nous formerons un régresseur de forêt aléatoire et utiliserons notre `calculate_permutation_importance()` pour identifier les caractéristiques les plus pertinentes. Cela nous aidera à sélectionner les caractéristiques que nous utiliserons pour l'exercice d'ingénierie des caractéristiques de la prochaine section.

Vous devrez répondre aux questions présentées dans cette section.

### 2.3 Ingénierie des Caractéristiques

- Compléter `hw5.py:add_absolute_coordinate_changes()`
- Compléter les exécutions de code dans `hw5.ipynb`
- Répondre aux questions dans `hw5.ipynb`

Nous arrivons à la fin, mais nous devons encore en apprendre un peu sur l'ingénierie des caractéristiques. Dans cet exercice, nous explorerons la création de deux nouvelles caractéristiques appelées `abs_lon_change` et `abs_lat_change`, qui représenteront la `distance longitudinale absolue` et la `distance latitudinale absolue`. Pour créer ces nouvelles caractéristiques, vous devrez compléter la fonction `add_absolute_coordinate_changes()`.

Vous devrez répondre aux questions présentées dans cette section.

**C'est la fin de cette assignation. J'espère que vous l'avez trouvé utile.**

# Références

- **Cette assignation** est basée sur le [cours](https://www.kaggle.com/learn/machine-learning-explainability) de Kaggle.
 de la Méthode d'Explicabilité de l'Apprentissage Automatique.
