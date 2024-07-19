# %%
import os
import sys
from functools import partial
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.feature_selection
import sklearn.tree

_ROOT_DIRNAME = Path(__file__).parents[1]
sys.path.insert(0, _ROOT_DIRNAME.as_posix())


import utils.img_loader as img_loader_utils
import utils.process_dataset as process_dataset_utils
import utils.process_funcs as process_funcs_utils
import utils.random as random_utils
import utils.similarity_dataset as similarity_dataset_utils
import utils.similarity_funcs as similarity_funcs_utils
from utils.algos import (
    describe_properties,
    get_pca_res,
    split_df,
    train_and_validate_decision_tree,
    u_test,
)
from utils.config import get_categories_config, get_features_config

# %%
_FEATURES_CONFIG = get_features_config()
_CATEGORIES_CONFIG = get_categories_config()
# %%
roi_img_df = img_loader_utils.load_roi_images(
    dirname=_ROOT_DIRNAME.joinpath("data").as_posix()
)
# %%
roi_random_img_df = random_utils.get_random_img_df(
    img_df=roi_img_df, random_seed=42
)
# %%
img_df = process_dataset_utils.concat_dfs([roi_img_df, roi_random_img_df])
del roi_img_df, roi_random_img_df
# %%
process_dict = {
    "x gradient": partial(process_funcs_utils.compute_gradient, axis=0),
    "y gradient": partial(process_funcs_utils.compute_gradient, axis=1),
    "z gradient": partial(process_funcs_utils.compute_gradient, axis=2),
    "fft real": lambda x: process_funcs_utils.fft(x).real,
    "spatial average": partial(
        process_funcs_utils.spatial_average, kernel_size=3, sigma=1
    ),
}
for process_name, process_func in process_dict.items():
    img_df = process_dataset_utils.process_original_imgs(
        img_df=img_df, process_method=process_name, process_func=process_func
    )
del process_name, process_func
# %%
similarity_dict = {
    "chebyshev": similarity_funcs_utils.chebyshev,
    "cityblock": similarity_funcs_utils.cityblock,
    "cosine": similarity_funcs_utils.cosine,
    "euclidean": similarity_funcs_utils.euclidean,
    "minkowski (5)": similarity_funcs_utils.minkowski_5,
    "minkowski (10)": similarity_funcs_utils.minkowski_10,
    "minkowski (50)": similarity_funcs_utils.minkowski_50,
    "pearson": similarity_funcs_utils.pearson,
    "spearman": similarity_funcs_utils.spearman,
}
for similarity_name, similarity_func in similarity_dict.items():
    for process_name in list(process_dict.keys()) + [None]:
        img_df = process_dataset_utils.add_similarity(
            img_df=img_df,
            process_method=process_name,
            similarity_method=similarity_name,
            similarity_func=similarity_func,
        )
del similarity_name, similarity_func, process_name


# %%
def get_barplot_similarity_df():
    series_lst = []
    for img_idx, img_row in img_df.iterrows():
        similarity_df: pd.DataFrame = img_row[
            _FEATURES_CONFIG.SIMILARITY
        ].copy()
        similarity_df = similarity_df[
            similarity_df[_FEATURES_CONFIG.PROCESS_METHOD]
            == _CATEGORIES_CONFIG.PROCESS_METHOD.ORIGINAL
        ].copy()
        for _, similarity_row in similarity_df.iterrows():
            if similarity_row[_FEATURES_CONFIG.SUBJECT_ID] == img_idx:
                continue
            fields_dict = {}
            fields_dict[_FEATURES_CONFIG.IS_SPECIFIC] = img_row[
                _FEATURES_CONFIG.IS_SPECIFIC
            ]
            fields_dict[_FEATURES_CONFIG.SIMILARITY_METHOD] = similarity_row[
                _FEATURES_CONFIG.SIMILARITY_METHOD
            ]
            if bool(similarity_row[_FEATURES_CONFIG.IS_SPECIFIC]):
                data_type = (
                    f"Specific ({similarity_row[_FEATURES_CONFIG.DATA_TYPE]})"
                )
            else:
                data_type = f"Non-specific ({similarity_row[_FEATURES_CONFIG.DATA_TYPE]})"
            fields_dict[_FEATURES_CONFIG.DATA_TYPE] = data_type
            fields_dict[_FEATURES_CONFIG.SIMILARITY] = similarity_row[
                _FEATURES_CONFIG.SIMILARITY
            ]
            series_lst.append(pd.Series(fields_dict))
    return pd.DataFrame(series_lst)


barplot_similarity_df = get_barplot_similarity_df()


# %%
def iter_plot_barplot():
    for similarity_method in barplot_similarity_df[
        _FEATURES_CONFIG.SIMILARITY_METHOD
    ].unique():
        fig, ax = plt.subplots()
        method_df = barplot_similarity_df[
            barplot_similarity_df[_FEATURES_CONFIG.SIMILARITY_METHOD]
            == similarity_method
        ].copy()
        sns.barplot(
            data=method_df,
            x=_FEATURES_CONFIG.IS_SPECIFIC,
            y=_FEATURES_CONFIG.SIMILARITY,
            hue=_FEATURES_CONFIG.DATA_TYPE,
            n_boot=10000,
            ax=ax,
        )
        sns.move_legend(obj=ax, loc="upper right")
        ax.set_ylabel(similarity_method)
        plt.show(fig)


iter_plot_barplot()

# %%
avg_similarity_df = similarity_dataset_utils.get_avg_similarity_df(
    img_df=img_df,
    similarity_process_pairs=list(
        product(similarity_dict.keys(), list(process_dict.keys()) + [None])
    ),
    only_real=True,
    n_subjects=-1,
    include_self=False,
    similairty_type="specific",
)
# %%
avg_similarity_df = split_df(
    df=avg_similarity_df,
    train_val_test=(6, 2, 2),
    shuffle=True,
    random_state=42,
)
# %%
specific_props, non_specific_props = describe_properties(
    avg_similarity_df=avg_similarity_df
)
# %%
statistics, pvalue = u_test(avg_similarity_df=avg_similarity_df)

# %%

pca_res = get_pca_res(avg_similarity_df=avg_similarity_df, random_seed=42)
# plt.scatter(
#     x=pca_res["1st principal component"],
#     y=pca_res["2nd principal component"],
#     s=0.2,
#     c=pca_res[_FEATURES_CONFIG.IS_SPECIFIC],
#     cmap="plasma"
# )
sns.scatterplot(
    data=pca_res,
    x="1st principal component",
    y="2nd principal component",
    s=10,
    hue=_FEATURES_CONFIG.IS_SPECIFIC,
)
# %%
# Machine Learning Part
subset_dict = {
    subset: avg_similarity_df[
        avg_similarity_df[_FEATURES_CONFIG.SUBSET_TYPE] == subset
    ]
    .copy()
    .drop(_FEATURES_CONFIG.SUBSET_TYPE, axis=1)
    for subset in [
        _CATEGORIES_CONFIG.SUBSET_TYPE.TRAIN,
        _CATEGORIES_CONFIG.SUBSET_TYPE.VALIDATION,
        _CATEGORIES_CONFIG.SUBSET_TYPE.TEST,
    ]
}


def get_xy(dataset: pd.DataFrame):
    x = dataset.drop(_FEATURES_CONFIG.IS_SPECIFIC, axis=1).copy()
    y = dataset.loc[:, _FEATURES_CONFIG.IS_SPECIFIC].copy()
    return x, y


# %%


def search_tree(seed: int):
    from sklearn.model_selection import GridSearchCV
    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier(
        criterion="gini",
        splitter="random",
        random_state=seed,
    )
    params = {
        "max_depth": range(5, 15),
        "min_samples_split": range(2, 12),
        "min_samples_leaf": range(1, 6),
    }
    gs = GridSearchCV(
        tree,
        params,
        cv=5,
        scoring="f1",
        verbose=1,
        n_jobs=-1,
    )

    train_set: pd.DataFrame = pd.concat(
        [
            subset_dict[subset]
            for subset in [
                _CATEGORIES_CONFIG.SUBSET_TYPE.TRAIN,
                _CATEGORIES_CONFIG.SUBSET_TYPE.VALIDATION,
            ]
        ]
    ).copy()
    x, y = get_xy(train_set)
    gs.fit(x, y)
    print(f"score: {gs.best_score_}")
    print(f"params: {gs.best_params_}")
    return gs.best_params_


search_tree(42)
del search_tree


# %%
def describe_feature_importance(seed: int) -> pd.Series:
    from sklearn.ensemble import ExtraTreesClassifier

    train_set: pd.DataFrame = pd.concat(
        [
            subset_dict[subset]
            for subset in [
                _CATEGORIES_CONFIG.SUBSET_TYPE.TRAIN,
            ]
        ]
    )
    x, y = get_xy(train_set)
    clf = ExtraTreesClassifier(
        n_estimators=100000, n_jobs=-1, verbose=1, random_state=seed
    )
    clf = clf.fit(x, y)
    importance_dict = {
        name: importance
        for name, importance in zip(x.columns, clf.feature_importances_)
    }
    importance_s = pd.Series(importance_dict)
    importance_s = importance_s.sort_values(ascending=True)
    return importance_s


importance_s = describe_feature_importance(42)
# %%


def plot_tree(seed: int):
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    clf = train_and_validate_decision_tree(
        avg_similarity_df=avg_similarity_df,
        splitter="random",
        max_depth=8,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=seed,
        max_features=None,
        max_leaf_nodes=None,
        ccp_alpha=0,
    )
    x, y = get_xy(subset_dict["validation"])
    y_pred = clf.predict(x)
    scores_dict = {
        "Accuracy": accuracy_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "F1": f1_score(y, y_pred),
        "ROC AUC": roc_auc_score(y, y_pred),
    }
    from sklearn.tree import plot_tree

    plot_tree(clf, filled=True)
    return scores_dict

    # from sklearn import tree

    # fig, ax = plt.subplots(figsize=(20, 20))
    # tree.plot_tree(clf, filled=True, ax=ax, class_names=["Non-specific", "Specific"], feature_names=x.columns)
    # for name, importance in zip(
    #     x.columns, clf.feature_importances_
    # ):
    #     print(f"{name}, {importance:.2%}")


dict_lst = []
# for seed in range(10000):
# dict_lst.append(plot_tree(seed))
plot_tree(42)
pd.DataFrame(dict_lst).describe()

# %%

path = tree.cost_complexity_pruning_path(x_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(
        random_state=42,
        max_depth=14,
        min_samples_leaf=1,
        min_samples_split=4,
        ccp_alpha=ccp_alpha,
    )
    clf.fit(x_train, y_train)
    clfs.append(clf)
print(
    "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]
    )
)

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()

train_scores = [clf.score(x_train, y_train) for clf in clfs]
test_scores = [clf.score(x_val, y_val) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(
    ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post"
)
ax.plot(
    ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post"
)
ax.legend()
plt.show()
# %%

# sns.barplot(
#     data=avg_similarity_df,
#     x="is specific",
#     y="spearman (original, specific similarity)",
# )

# %%
# for region in img_df[_FEATURES_CONFIG.REGION].unique():
#     for similarity_name in similarity_dict.keys():
#         for process_name in list(process_dict.keys()) + [None]:
#             sim_mat = similarity_dataset_utils.get_similarity_mat(
#                 img_df=img_df,
#                 region=region,
#                 similarity_name=similarity_name,
#                 process_name=process_name,
#             )
#             sns.clustermap(sim_mat)
#             plt.title(f"{region} - {similarity_name}")
#             plt.show()

# # %%
# same_stimulation_similarity_df_lst = []
# for similarity_name in similarity_dict.keys():
#     same_stimulation_similarity_df = (
#         similarity_dataset_utils.get_same_stimulation_similarity_df(
#             img_df=img_df, similarity_name=similarity_name, include_self=False
#         )
#     )
#     same_stimulation_similarity_df_lst.append(same_stimulation_similarity_df)

# same_stimulation_similarity_df = pd.concat(
#     same_stimulation_similarity_df_lst, ignore_index=True
# )
# %%
