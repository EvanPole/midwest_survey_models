# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Midwest Survey — Model Comparison
#
# Predict the census region from survey responses using three pipelines:
# Logistic Regression, Random Forest, and Gradient Boosting.
# Each uses `skrub.TableVectorizer` for automatic feature encoding.
# Training is done on a shuffled sample of 1,000 rows.

# %%
import joblib
import skops.io as sio
import skrub
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

from midwest_survey_models.transformers import NumericalStabilizer

# %% [markdown]
# ## Data

# %%
bunch = skrub.datasets.fetch_midwest_survey()
X_full, y_full = bunch.X, bunch.y

sample_idx = X_full.sample(n=1000, random_state=1).index
X = X_full.loc[sample_idx].reset_index(drop=True)
y = y_full.loc[sample_idx].reset_index(drop=True)

print(f"Training set: {X.shape}, target classes: {y.nunique()}")

# %%
y_simplified = y.apply(lambda x: "North Central" if x in ["East North Central", "West North Central"] else "other")

# %%
y_simplified.value_counts()

# %% [markdown]
# ## Logistic Regression

# %%
lr = make_pipeline(
    skrub.TableVectorizer(numeric=SimpleImputer()),
    LogisticRegression(),
)
lr

# %%
lr.fit(X, y_simplified)
sio.dump(lr, "model_logistic_regression.skops")

# %% [markdown]
# ## Random Forest

# %%
rf = make_pipeline(
    skrub.TableVectorizer(),
    NumericalStabilizer(),
    RandomForestClassifier(n_estimators=200, random_state=42),
)
rf

# %%
rf.fit(X, y_simplified)
sio.dump(rf, "model_random_forest.skops")

# %% [markdown]
# ## Gradient Boosting

# %%
gb = make_pipeline(
    skrub.TableVectorizer(),
    HistGradientBoostingClassifier(max_iter=200, random_state=42),
)
gb

# %%
gb.fit(X, y_simplified)
sio.dump(gb, "model_gradient_boosting.skops")

# %% [markdown]
# ## Chargement securise avec skops
#
# Avec skops.io, le chargement d'un modele necessite de specifier explicitement
# les types de confiance. Si le fichier contient des classes inconnues ou
# malveillantes, le chargement echoue au lieu d'executer du code arbitraire
# (contrairement a pickle/joblib).

# %%
unknown_types = sio.get_untrusted_types(file="model_logistic_regression.skops")
print("Types a valider avant chargement :", unknown_types)

lr_loaded = sio.load("model_logistic_regression.skops", trusted=unknown_types)
print("Modele charge en securite :", type(lr_loaded))
