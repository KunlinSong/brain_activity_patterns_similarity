from collections import namedtuple
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle as skshuffle

from .config import get_categories_config, get_features_config


def split_df(
    df: pd.DataFrame,
    train_val_test: tuple[int, int, int] = (6, 2, 2),
    shuffle: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    _FEATURES_CONFIG = get_features_config()
    _CATEGORIES_CONFIG = get_categories_config()
    df = df.copy()
    if shuffle:
        df = skshuffle(df, random_state=random_state)
    df.loc[:, _FEATURES_CONFIG.SUBSET_TYPE] = None
    total = sum(train_val_test)
    labels = []
    for subset_label, subset_factor in zip(
        [
            _CATEGORIES_CONFIG.SUBSET_TYPE.TRAIN,
            _CATEGORIES_CONFIG.SUBSET_TYPE.VALIDATION,
            _CATEGORIES_CONFIG.SUBSET_TYPE.TEST,
        ],
        train_val_test,
    ):
        subset_len = len(df) * subset_factor // total
        if len(labels) + subset_len > len(df):
            subset_len = len(df) - len(labels)
        labels += [subset_label] * subset_len
    if (rest_len := (len(df) - len(labels))) > 0:
        labels += [_CATEGORIES_CONFIG.SUBSET_TYPE.TEST] * rest_len
    df[_FEATURES_CONFIG.SUBSET_TYPE] = labels
    return df


def _split_target_df(avg_similarity_df) -> tuple[pd.DataFrame, pd.DataFrame]:
    _FEATURES_CONFIG = get_features_config()
    avg_similarity_df, _ = pop_column(
        df=avg_similarity_df,
        column_name=_FEATURES_CONFIG.SUBSET_TYPE,
    )
    specific_df = avg_similarity_df[
        avg_similarity_df[_FEATURES_CONFIG.IS_SPECIFIC].astype(bool)
    ].copy()
    non_specific_df = avg_similarity_df[
        ~avg_similarity_df[_FEATURES_CONFIG.IS_SPECIFIC].astype(bool)
    ].copy()
    specific_df, _ = pop_column(
        df=specific_df, column_name=_FEATURES_CONFIG.IS_SPECIFIC
    )
    non_specific_df, _ = pop_column(
        df=non_specific_df,
        column_name=_FEATURES_CONFIG.IS_SPECIFIC,
    )
    return specific_df, non_specific_df


def describe_properties(
    avg_similarity_df,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def _describe_df(df: pd.DataFrame) -> pd.DataFrame:
        prop_dict = {}
        for column in df.columns:
            prop_dict[column] = {
                "N": len(df),
                "Mean": np.mean(df.loc[:, column]),
                "Std": np.std(df.loc[:, column]),
                "0%": np.percentile(df.loc[:, column], 0),
                "25%": np.percentile(df.loc[:, column], 25),
                "50%": np.percentile(df.loc[:, column], 50),
                "75%": np.percentile(df.loc[:, column], 75),
                "100%": np.percentile(df.loc[:, column], 100),
            }

        return pd.DataFrame(prop_dict)

    avg_similarity_df = avg_similarity_df.copy()
    specific_df, non_specific_df = _split_target_df(
        avg_similarity_df=avg_similarity_df
    )
    specific_props = _describe_df(specific_df)
    non_specific_props = _describe_df(non_specific_df)
    return specific_props, non_specific_props


def pop_column(
    df: pd.DataFrame, column_name: str
) -> tuple[pd.DataFrame, pd.Series | None]:
    if column_name not in df.columns:
        return None, df
    else:
        column = df[column_name].copy()
        df = df.drop(column_name, axis=1).copy()
        return df, column


def _get_pandas_scaler():
    scaler = StandardScaler()
    scaler = scaler.set_output(transform="pandas")
    return scaler


UTestRes = namedtuple("UTestRes", ("statistics", "pvalues"))


def u_test(
    avg_similarity_df: pd.DataFrame,
) -> UTestRes:
    specific_df, non_specific_df = _split_target_df(
        avg_similarity_df=avg_similarity_df
    )
    statistic_dict = {}
    pvalue_dict = {}
    for feat in specific_df.columns:
        statistic, pvalue = mannwhitneyu(
            specific_df.loc[:, feat], non_specific_df.loc[:, feat]
        )
        statistic_dict[feat] = statistic
        pvalue_dict[feat] = pvalue
    return (
        pd.Series(statistic_dict),
        pd.Series(pvalue_dict),
    )


def get_pca_res(
    avg_similarity_df: pd.DataFrame, random_seed: int = 42
) -> pd.DataFrame:
    _FEATURES_CONFIG = get_features_config()
    feat_series_dict = {}
    for feat in [
        _FEATURES_CONFIG.IS_SPECIFIC,
        _FEATURES_CONFIG.SUBSET_TYPE,
    ]:
        avg_similarity_df, feat_series_dict[feat] = pop_column(
            df=avg_similarity_df, column_name=feat
        )
    scaler = _get_pandas_scaler()
    scaled_df = scaler.fit_transform(avg_similarity_df)
    pca = PCA(n_components=2, random_state=random_seed)
    scaled_pca = pca.fit(scaled_df)
    trained_res = scaled_pca.transform(scaled_df)
    pca_res_dict = {
        "1st principal component": trained_res[:, 0],
        "2nd principal component": trained_res[:, 1],
    }
    pca_res = pd.DataFrame(pca_res_dict)
    for feat, cont in feat_series_dict.items():
        pca_res.loc[:, feat] = cont
    return pca_res


def get_kmeans_res(
    avg_similarity_df: pd.DataFrame,
) -> tuple[KMeans, StandardScaler]:
    _FEATURES_CONFIG = get_features_config()
    avg_similarity_df = avg_similarity_df.copy()
    for feat in [
        _FEATURES_CONFIG.IS_SPECIFIC,
        _FEATURES_CONFIG.SUBSET_TYPE,
    ]:
        avg_similarity_df, _ = pop_column(
            df=avg_similarity_df, column_name=feat
        )

    scaler = _get_pandas_scaler()
    scaled_df = scaler.fit_transform(avg_similarity_df)
    kmeans = KMeans(
        n_clusters=2,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=42,
        algorithm="lloyd",
    )
    kmeans.fit(scaled_df)
    return kmeans, scaler


def _get_subset_df(
    avg_similarity_df: pd.DataFrame,
    subset_type: Literal["train", "validation", "test"],
) -> tuple[pd.DataFrame, pd.Series]:
    avg_similarity_df = avg_similarity_df.copy()
    _FEATURES_CONFIG = get_features_config()
    _CATEGORIES_CONFIG = get_categories_config()
    match subset_type:
        case "train":
            type_name = _CATEGORIES_CONFIG.SUBSET_TYPE.TRAIN
        case "validation":
            type_name = _CATEGORIES_CONFIG.SUBSET_TYPE.VALIDATION
        case "train":
            type_name = _CATEGORIES_CONFIG.SUBSET_TYPE.TEST
        case _:
            raise ValueError(f"Invalid subset type: {subset_type}")
    df = avg_similarity_df[
        avg_similarity_df[_FEATURES_CONFIG.SUBSET_TYPE] == type_name
    ].copy()
    df = df.drop(_FEATURES_CONFIG.SUBSET_TYPE, axis=1).copy()
    x, y = pop_column(df=df, column_name=_FEATURES_CONFIG.IS_SPECIFIC)
    return x, y


def _print_scores(y_true, y_pred) -> None:
    print(f"Accuracy: {accuracy_score(y_true=y_true, y_pred=y_pred)}")
    print(f"Recall: {recall_score(y_true=y_true, y_pred=y_pred)}")
    print(f"Precision: {precision_score(y_true=y_true, y_pred=y_pred)}")
    print(f"F1: {f1_score(y_true=y_true, y_pred=y_pred)}")
    print(f"ROC AUC: {roc_auc_score(y_true=y_true, y_score=y_pred)}")


def train_and_validate_decision_tree(
    avg_similarity_df: pd.DataFrame,
    splitter: str,
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int,
    max_features: str | None,
    random_state: int,
    max_leaf_nodes: int,
    ccp_alpha: float,
) -> DecisionTreeClassifier:
    # ) -> tuple[DecisionTreeClassifier, StandardScaler]:
    avg_similarity_df = avg_similarity_df.copy()
    x_train, y_train = _get_subset_df(
        avg_similarity_df=avg_similarity_df, subset_type="train"
    )
    x_val, y_val = _get_subset_df(
        avg_similarity_df=avg_similarity_df, subset_type="validation"
    )

    # scaler: StandardScaler = _get_pandas_scaler()
    # x_train = scaler.fit_transform(x_train)
    # x_val = scaler.transform(x_val)

    clf = DecisionTreeClassifier(
        criterion="gini",
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        max_leaf_nodes=max_leaf_nodes,
        ccp_alpha=ccp_alpha,
    )
    clf.fit(x_train, y_train)
    y_train_pred = clf.predict(x_train)
    y_val_pred = clf.predict(x_val)
    print(f"{'=':<10}")
    print("Train scores:")
    _print_scores(y_true=y_train, y_pred=y_train_pred)
    print(f"{'-':<10}")
    print("Validation scores:")
    _print_scores(y_true=y_val, y_pred=y_val_pred)
    print(f"{'-':<10}")
    print("Params: ")
    print(
        f"{splitter=}, {max_depth=}, {min_samples_split=}, {min_samples_leaf=}, {max_features=}, {max_leaf_nodes=}"
    )
    print(f"{'=':<10}")
    # return clf, scaler
    return clf


def train_and_validate_svm(
    avg_similarity_df: pd.DataFrame,
    c: float,
) -> tuple[DecisionTreeClassifier, StandardScaler]:
    avg_similarity_df = avg_similarity_df.copy()
    x_train, y_train = _get_subset_df(
        avg_similarity_df=avg_similarity_df, subset_type="train"
    )
    x_val, y_val = _get_subset_df(
        avg_similarity_df=avg_similarity_df, subset_type="validation"
    )

    scaler: StandardScaler = _get_pandas_scaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    clf = SVC(C=c, kernel="rbf", gamma="scale", random_state=42)
    clf.fit(x_train, y_train)
    y_train_pred = clf.predict(x_train)
    y_val_pred = clf.predict(x_val)
    print(f"{'=':<10}")
    print("Train scores:")
    _print_scores(y_true=y_train, y_pred=y_train_pred)
    print(f"{'-':<10}")
    print("Validation scores:")
    _print_scores(y_true=y_val, y_pred=y_val_pred)
    print(f"{'-':<10}")
    print("Params: ")
    print(f"{c=}")
    print(f"{'=':<10}")
    return clf, scaler


def train_and_validate_knn(
    avg_similarity_df: pd.DataFrame,
    n_neighbors: float,
) -> tuple[DecisionTreeClassifier, StandardScaler]:
    avg_similarity_df = avg_similarity_df.copy()
    x_train, y_train = _get_subset_df(
        avg_similarity_df=avg_similarity_df, subset_type="train"
    )
    x_val, y_val = _get_subset_df(
        avg_similarity_df=avg_similarity_df, subset_type="validation"
    )

    scaler: StandardScaler = _get_pandas_scaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    clf = KNeighborsClassifier(n_neighbors=n_neighbors, random_state=42)
    clf.fit(x_train, y_train)
    y_train_pred = clf.predict(x_train)
    y_val_pred = clf.predict(x_val)
    print(f"{'=':<10}")
    print("Train scores:")
    _print_scores(y_true=y_train, y_pred=y_train_pred)
    print(f"{'-':<10}")
    print("Validation scores:")
    _print_scores(y_true=y_val, y_pred=y_val_pred)
    print(f"{'-':<10}")
    print("Params: ")
    print(f"{n_neighbors=}")
    print(f"{'=':<10}")
    return clf, scaler


def test_model(
    model,
    avg_similarity_df: pd.DataFrame,
    scaler: StandardScaler | None = None,
) -> None:
    avg_similarity_df = avg_similarity_df.copy()

    x_test, y_test = _get_subset_df(
        avg_similarity_df=avg_similarity_df, subset_type="test"
    )
    if scaler is not None:
        x_test = scaler.transform(x_test)
    y_pred = model.predict(x_test)
    print(f"{'=':<10}")
    print("Test scores")
    _print_scores(y_true=y_test, y_pred=y_pred)
    print(f"{'=':<10}")
