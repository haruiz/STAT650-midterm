from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_numeric_dtype
import math
from geopy.geocoders import Nominatim
import pycountry
from collections import namedtuple
import pycountry_convert as pc

plt.style.use("seaborn-darkgrid")


def load_data(uri: str) -> pd.DataFrame:
    """Load data from a URI
    @param uri: URI of the data
    @return: dataframe
    """
    return pd.read_csv(uri)


def filter_data_based_on_missing_values(
    df: pd.DataFrame, threshold: float = 0.5
) -> pd.DataFrame:
    """Filter data based on missing values
    @param df: dataframe
    @param threshold: threshold of missing values
    @return: filtered dataframe
    """
    return df[df.columns[df.isnull().mean() < threshold]]


def get_cols_description(cols_df: pd.DataFrame) -> dict:
    """Get the description of the columns of a dataframe
    @param df: dataframe
    @return: description of the columns
    """
    return dict(zip(cols_df["column"], cols_df["description"]))


def get_numeric_columns(df):
    """
    return the subset of numeric columns within a df
    :param df:
    :return:
    """
    return df.select_dtypes(include=["number"]).columns.tolist()


def get_categorical_columns(df):
    """
    return the subset of categorical columns within a df
    :param df:
    :return:
    """
    return df.select_dtypes(include="category").columns.tolist()


def get_cols_with_nan_values(df):
    """
    Return the name of the columns with missing values
    @param df: input dataframe
    """
    missing_values_df = df.isna().sum()
    missing_values_df = missing_values_df[missing_values_df > 0]
    column_names = list(missing_values_df.index)
    return column_names


def plot_numerical_col(df, col_name, fig=None):
    """
    Plot the histogram for the given numerical variable in the dataframe
    :param df: input dataframe
    :param col_name: target column
    :return: None
    """
    assert col_name in df.columns, f"Column {col_name} not found"
    assert is_numeric_dtype(df[col_name]), "The provide column must be numeric"
    if fig:
        (ax_box, ax_hist) = fig.subplots(
            2, sharex=True, gridspec_kw={"height_ratios": (0.10, 0.90)}
        )
    else:
        fig, (ax_box, ax_hist) = plt.subplots(
            2, gridspec_kw={"height_ratios": (0.10, 0.90)}
        )
    # assigning a graph to each ax
    sns.boxplot(data=df, x=col_name, ax=ax_box)
    sns.histplot(data=df, x=col_name, ax=ax_hist, kde=True, stat="density")

    # Remove x-axis name for the boxplot
    ax_box.set(xlabel="")


def plot_categorical_col(df, col_name, fig=None):
    """
    Plot the histogram for the given numerical variable in the dataframe
    :param df: input daframe
    :param col_name: target column
    :return: None
    """
    assert col_name in df.columns, f"Column {col_name} not found"
    assert is_categorical_dtype(df[col_name]), "The provide column must be categorical"
    if fig:
        ax_hist = fig.add_subplot(111)
    else:
        ax_hist = plt.subplot()
    p = sns.countplot(y=df[col_name], ax=ax_hist)
    p.set_ylabel(col_name, fontsize=15)
    p.set_yticklabels(p.get_yticklabels(), size=10)


def plot_df_col(df, col_name, fig=None):
    """
    pick the right plot according to the column type
    :param df: input dataframe
    :param col_name: column name
    :param fig:
    :return:
    """
    if is_categorical_dtype(df[col_name]):
        plot_categorical_col(df, col_name, fig)
    elif is_numeric_dtype(df[col_name]):
        plot_numerical_col(df, col_name, fig)


def plot_df(df, ncols=3):
    """
    plot the columns of a df
    :param df: input dataframe
    :return: None
    """
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    num_plots = len(df.columns)
    nrows = math.ceil(num_plots / ncols)
    subfigs = fig.subfigures(nrows, ncols)
    i = 0
    for col in range(ncols):
        for row in range(nrows):
            plot_df_col(df, df.columns[i], fig=subfigs[row][col])
            i += 1


def plot_corr_heatmap(df: pd.DataFrame, title: str = None, ax=None) -> plt.Axes:
    """Plot the correlation heatmap of a dataframe
    @param df: dataframe
    @param title: title of the plot
    @param ax: matplotlib axes
    @return: matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    corr = df.corr()
    sns.heatmap(corr, ax=ax, cmap="YlGnBu", annot=False, fmt=".2f")
    ax.set_title(title)
    return ax


def get_location_from_country_name(country: str) -> str:
    """Get the location of a country
    @param country: country
    @return: location
    """
    return pycountry.countries.get(name=country)


def get_continent_from_country_name(country: str) -> Any | None:
    """Get the continent of a country
    @param country: country
    @return: continent
    """
    try:
        country_alpha2 = pc.country_name_to_country_alpha2(country)
        country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        country_continent_name = pc.convert_continent_code_to_continent_name(
            country_continent_code
        )
    except:
        return None
    return country_continent_name


def get_lat_and_long_from_country_name(
    country: str, application: str, raise_exception=True
):
    """Get the latitude and longitude of a location
    @param country: country
    @param application: application
    @param raise_exception: if raise exception when the location is not found
    @return: latitude and longitude
    """
    Location = namedtuple("Location", ["latitude", "longitude"])
    geolocator = Nominatim(user_agent=application)
    location = geolocator.geocode(country)
    if location is None:
        if raise_exception:
            raise ValueError(f"Cannot find the location of {country}")
        return Location(np.nan, np.nan)
    return Location(location.latitude, location.longitude)


def add_lat_and_long_to_df(
    df: pd.DataFrame, application: str = "STAT650-midterm"
) -> pd.DataFrame:
    """Add latitude and longitude to a dataframe
    @param df: dataframe
    @param application: application
    @return: dataframe with latitude and longitude
    """
    lookup_dict = {}
    for country in df["country"].unique():
        lookup_dict[country] = get_lat_and_long_from_country_name(
            country, application, raise_exception=False
        )

    df = df.copy()
    df["lat"] = df["country"].map(lambda x: lookup_dict[x].latitude)
    df["long"] = df["country"].map(lambda x: lookup_dict[x].longitude)
    return df


def add_continent_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add continent to a dataframe
    @param df: dataframe
    @return: dataframe with continent
    """
    lookup_dict = {}
    for country in df["country"].unique():
        lookup_dict[country] = get_continent_from_country_name(country)
    df = df.copy()
    df["continent"] = df["country"].map(lambda x: lookup_dict[x])
    return df


def mean_imputation(in_df):
    """
    Replace the missing values in a dataframe by the corresponding column mean
    @param in_df: input dataframe
    @return: a copy of the input dataset after having replacing the missing values
    """
    cols_with_nan = get_cols_with_nan_values(in_df)
    assert cols_with_nan, "Not missing values found"

    out_df = in_df.copy()
    for col_name in cols_with_nan:
        if not is_numeric_dtype(out_df[col_name]):
            continue
        col_mean = out_df[col_name].mean()
        # replace missing values
        out_df.loc[out_df[col_name].isna(), col_name] = col_mean
        out_df[col_name] = pd.to_numeric(out_df[col_name])
    return out_df


def row_imputation(in_df):
    """
    Replace the missing values in a dataframe by the corresponding column mean
    @param in_df: input dataframe
    @return: a copy of the input dataset after having replacing the missing values
    """
    cols_with_nan = get_cols_with_nan_values(in_df)
    assert cols_with_nan, "Not missing values found"

    out_df = in_df.copy()
    for col_name in cols_with_nan:
        if not is_numeric_dtype(out_df[col_name]):
            continue
        # obtain the indices of the rows with missing values in column `col_name`
        rows_indices = out_df[col_name].isna()
        # select and get the mean of the obtained indices
        rows = out_df.loc[rows_indices]
        rows_mean = rows.mean(axis=1)
        # replace the missing value at column `col_name` by the rows means
        out_df.loc[rows_indices, col_name] = rows_mean
        out_df[col_name] = pd.to_numeric(out_df[col_name])
    return out_df


def remove_rows_with_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing values
    @param df: dataframe
    @return: dataframe without missing values
    """
    return df.dropna()


def remove_missing_values(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Remove missing values
    @param df: dataframe
    @param method: method to remove missing values
    @return: dataframe without missing values
    """
    if method == "mean":
        return mean_imputation(df)
    elif method == "row":
        return row_imputation(df)
    elif method == "drop":
        return remove_rows_with_missing_values(df)
    else:
        raise ValueError(f"Method {method} not supported")


def get_observed_values_by_country(df: pd.DataFrame, country: str) -> pd.Series:
    """Get the observed values by country
    @param df: dataframe
    @param country: country
    @return: observed values
    """
    return df[df["country"] == country]


def get_observed_values_by_continent(df: pd.DataFrame, continent: str) -> pd.Series:
    """Get the observed values by continent
    @param df: dataframe
    @param continent: continent
    @return: observed values
    """
    return df[df["continent"] == continent]


def plot_co2_emissions_by_year(df: pd.DataFrame, country: str):
    """Plot the CO2 emissions by year
    @param df: dataframe
    @param country: country
    """
    country_data = get_observed_values_by_country(df, country)
    sns.lineplot(data=country_data, x="year", y="co2")
    plt.title(f"CO2 emissions by year - {country}")


def plot_co2_emissions_by_continent(df: pd.DataFrame, continent: str, top: int = None):
    """Plot the CO2 emissions by year
    @param df: dataframe
    @param continent: continent
    @param top: top countries to plot
    """
    continent_data = get_observed_values_by_continent(df, continent)
    continent_data = continent_data.sort_values("co2", ascending=False)
    if top is not None:
        top_n_countries = continent_data["country"].unique().tolist()[:top]
        continent_data = continent_data[continent_data["country"].isin(top_n_countries)]
    sns.lineplot(data=continent_data, x="year", y="co2", hue="country")
    plt.title(f"CO2 emissions by year - {continent}")
