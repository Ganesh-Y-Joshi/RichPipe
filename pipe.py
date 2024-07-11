import enum
import json
import logging
import queue
import time
import uuid
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import threading
from sklearn.impute import KNNImputer
from sklearn.preprocessing import (LabelEncoder,
                                   RobustScaler,
                                   StandardScaler,
                                   MinMaxScaler)

from scipy.stats import mstats
from sklearn.model_selection import train_test_split


def _generate_suffixes(merged_columns):
    """
            Generates suffixes to differentiate overlapping column names during joins.

            Args:
                merged_columns (set): A set of column names present in both DataFrames.

            Returns:
                list: A list of suffixes to append to column names.
    """
    suffixes = []
    for col in merged_columns:
        suffixes.append(f"_{col}")
    return suffixes


class RichPipe:
    def __init__(self, df: pd.DataFrame, _id=uuid.uuid4()):
        """
                Initializes a RichPipe object with a Pandas DataFrame.

                Args:
                    df (pd.DataFrame): The DataFrame to operate on.
        """
        self.df = df
        self._want_outlier_indices = False
        self._id = _id

    @property
    def want_outlier_indices(self):
        """
                A simple property to depict whether we want to have outlier indices in data

                Returns:
                     bool: whether we want to have the outliers
                     indices in the outliers removal
        """
        return self._want_outlier_indices

    @want_outlier_indices.setter
    def want_outlier_indices(self, val: bool):
        """
                Args:
                     bool: represnting wether we want outliers indices during removal
        """
        self._want_outlier_indices = val

    @property
    def _id_(self):
        return self._id

    def project(self, field):
        """
                Selects a specific column from the DataFrame.

                Args:
                    field (str): The name of the column to project.

                Returns:
                    RichPipe: A RichPipe object with the projected DataFrame.
        """
        try:
            return self.df[field]
        except KeyError as err:
            raise Exception(err)

    def discard(self, fields, axis):
        """
                Drops columns from the DataFrame.

                Args:
                    fields (list): A list of column names to discard.
                    axis (int, optional): The axis to drop columns from. Defaults to 1 (columns).

                Returns:
                    RichPipe: A RichPipe object with the DataFrame with columns discarded.
        """
        try:
            self.df = self.df.drop(fields, axis=axis)
            return self
        except KeyError as e:
            missing_columns = [col for col in fields if col not in self.df.columns]
            if missing_columns:
                raise ValueError(f"Columns not found in the DataFrame: {', '.join(missing_columns)}")
            else:
                raise e

    def limit(self, n):
        """
                Limits the number of rows returned by the DataFrame.

                Args:
                    n (int): The number of rows to return.

                Returns:
                    RichPipe: A RichPipe object with the first n rows of the DataFrame.
        """
        return self.df.head(n)

    def map(self, from_field, to_field, fn):
        """
                Applies a function to a column and stores the result in a new column.

                Args:
                    from_field (str): The name of the column to apply the function to.
                    to_field (str): The name of the new column to store the results.
                    fn (callable): The function to apply to each element in the column.

                Returns:
                    RichPipe: A RichPipe object with the transformed DataFrame.
        """
        try:
            self.df[to_field] = self.df[from_field].apply(fn)
            return self
        except KeyError:
            raise AttributeError(f"Column '{from_field}' not found in the DataFrame")

    def map_to(self, from_field, to_field, fn):
        """
                Applies a function to a column and stores the result in a new column
                (alternative implementation).

                Args:
                    from_field (str): The name of the column to apply the function to.
                    to_field (str): The name of the new column to store the results.
                    fn (callable): The function to apply to each element in the column.

                Returns:
                    pd.DataFrame: The transformed DataFrame.
        """
        try:
            df = self.df.assign(**{to_field: self.df[from_field].apply(fn)})
            return df
        except KeyError:
            raise AttributeError(f"Column '{from_field}' not found in the DataFrame")

    def flat_map(self, from_field, to_field, fn):
        """
                Explodes a list-valued column and applies a function to each element,
                storing the results in a new column.

                Args:
                    from_field (str): The name of the list-valued column to explode.
                    to_field (str): The name of the new column to store the results.
                    fn (callable): The function to apply to each element in the exploded list.

                Returns:
                    RichPipe: A RichPipe object with the transformed DataFrame.
        """
        try:
            self.df = self.df.explode(from_field)
            self.df[to_field] = self.df[from_field].apply(fn)
            return self
        except KeyError:
            raise AttributeError(f"Column '{from_field}' not found in the DataFrame")

    def flat_map_to(self, from_field, to_field, fn):
        """
                Explodes a list-valued column and applies a function to each element,
                storing the results in a new column (alternative implementation).

                Args:
                    from_field (str): The name of the list-valued column to explode.
                    to_field (str): The name of the new column to store the results.
                    fn (callable): The function to apply to each element in the exploded list.

                Returns:
                    pd.DataFrame: The transformed DataFrame.
        """
        try:
            exploded_df = self.df.explode(from_field).reset_index(drop=True)
            transformed_column = exploded_df[from_field].apply(fn)
            df = exploded_df.assign(**{to_field: transformed_column})
            return df
        except KeyError:
            raise AttributeError(f"Column '{from_field}' not found in the DataFrame")

    def then(self, fn):
        """
                Passes the RichPipe object to a function for further processing.

                Args:
                    fn (callable): The function to call with the RichPipe object as an argument.

                Returns:
                    Any: The return value of the function.
        """
        return fn(self)

    def normalize(self, field):
        """
                Normalizes a column by dividing each element by the sum of the column.

                Args:
                    field (str): The name of the column to normalize.

                Returns:
                    RichPipe: A RichPipe object with the normalized DataFrame.
        """
        total = self.df[field].sum()
        self.df[field] = self.df[field] / total
        return self.df

    def join_with_smaller(self, l_field, r_field, other, how='inner', suffixes=("_left", "_right")):
        """
                Joins the current DataFrame with another DataFrame on specified columns.

                Args:
                    l_field (str): The name of the left join column in the current DataFrame.
                    r_field (str): The name of the right join column in the other DataFrame.
                    other (RichPipe): The RichPipe object containing the DataFrame to join with.
                    how (str, optional): The type of join to perform. Defaults to "inner".
                        Valid options include "inner", "left", "right", and "outer".
                    suffixes (list, optional): A list of suffixes to add to column names in case of overlap.
                        Defaults to ["_left", "_right"].

                Returns:
                    RichPipe: A RichPipe object with the joined DataFrame.
        """
        try:
            self.df = self.df.merge(other.df, left_on=l_field, right_on=r_field, how=how, suffixes=suffixes)
            return self
        except KeyError as e:
            missing_cols = [col for col in [l_field, r_field] if
                            col not in self.df.columns and col not in other.df.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in DataFrames: {', '.join(missing_cols)}")
            else:
                raise e

    def left_join(self, l_field, r_field, other):
        """
                Performs a left join with another DataFrame.

                Args:
                    l_field (str): The name of the left join column in the current DataFrame.
                    r_field (str): The name of the right join column in the other DataFrame.
                    other (RichPipe): The RichPipe object containing the DataFrame to join with.

                Returns:
                    RichPipe: A RichPipe object with the left-joined DataFrame.
        """
        return self.join_with_smaller(l_field, r_field, other, 'left')

    def right_join(self, l_field, r_field, other):
        """
                Performs a right join with another DataFrame.

                Args:
                    l_field (str): The name of the left join column in the current DataFrame.
                    r_field (str): The name of the right join column in the other DataFrame.
                    other (RichPipe): The RichPipe object containing the DataFrame to join with.

                Returns:
                    RichPipe: A RichPipe object with the right-joined DataFrame.
        """
        return self.join_with_smaller(l_field, r_field, other, 'right')

    def inner_join(self, l_field, r_field, other):
        """
                Performs an inner join with another DataFrame.

                Args:
                    l_field (str): The name of the left join column in the current DataFrame.
                    r_field (str): The name of the right join column in the other DataFrame.
                    other (RichPipe): The RichPipe object containing the DataFrame to join with.

                Returns:
                    RichPipe: A RichPipe object with the inner-joined DataFrame.
        """
        return self.join_with_smaller(l_field, r_field, other, 'inner')

    def outer_join(self, l_field, r_field, other):
        """
                Performs an outer join with another DataFrame.

                Args:
                    l_field (str): The name of the left join column in the current DataFrame.
                    r_field (str): The name of the right join column in the other DataFrame.
                    other (RichPipe): The RichPipe object containing the DataFrame to join with.

                Returns:
                    RichPipe: A RichPipe object with the outer-joined DataFrame.
        """
        merged_columns = set(self.df.columns).intersection(other.df.columns)
        suffixes = _generate_suffixes(merged_columns)
        return self.join_with_smaller(l_field, r_field, other, 'outer', suffixes=suffixes)

    def __add__(self, other):
        """
                Concatenates two DataFrames (not recommended for RichPipe operations).

                Args:
                    other (RichPipe): The RichPipe object containing the DataFrame to concatenate with.

                Returns:
                    pd.DataFrame: The concatenated DataFrame (consider using RichPipe.concat instead).
        """
        pd.concat([self.df, other.df])

    def filter(self, condition_fn):
        """
                Filters the DataFrame based on a condition function.

                Args:
                    condition_fn (callable): A function that takes a DataFrame and returns a boolean mask.

                Returns:
                    RichPipe: A RichPipe object with the filtered DataFrame.
        """
        self.df = self.df[condition_fn(self.df)]
        return self

    def group_by(self, by, agg_fn):
        """
                Groups the DataFrame by specified columns and applies aggregation functions.

                Args:
                    by (list): A list of column names to group by.
                    agg_fn (dict): A dictionary mapping column names to aggregation functions.

                Returns:
                    RichPipe: A RichPipe object with the grouped and aggregated DataFrame.
        """
        grouped_df = self.df.groupby(by).agg(agg_fn).reset_index()
        return RichPipe(grouped_df, self._id)

    def pivot(self, index, columns, values, aggfunc='mean'):
        """
                 pivots the DataFrame.

                Args:
                    index (str): The column to use as the row index in the pivoted table.
                    columns (str): The column to use as the column labels in the pivoted table.
                    values (str): The column to use as the values in the pivoted table.
                    aggfunc (str, optional): The aggregation function to apply to the values. Defaults to "mean".

                Returns:
                    RichPipe: A RichPipe object with the pivoted DataFrame, or self if pivoting fails.
        """
        try:
            aggregated_df = self.df.groupby([index, columns]).agg({values: aggfunc}).reset_index()
            pivoted_df = aggregated_df.pivot(index=index, columns=columns, values=values)
            return RichPipe(pivoted_df, self._id)
        except Exception as e:
            warnings.warn(f"Warning: Pivoting failed. DataFrame remains unchanged. Error: {str(e)}")
            return self

    def melt(self, id_vars, value_vars, var_name='variable', value_name='value'):
        """
                Melts the DataFrame from wide to long format.

                Args:
                    id_vars (list): A list of column names to use as identifier variables.
                    value_vars (list): A list of column names to melt into separate rows.
                    var_name (str, optional): The name of the variable name column in the melted format.
                                                Defaults to "variable".
                    value_name (str, optional): The name of the value column in the melted format. Defaults to "value".

                Returns:
                    RichPipe: A RichPipe object with the melted DataFrame.
        """
        try:
            self.df = self.df.melt(id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=value_name)
            return self
        except Exception as e:
            warnings.warn(f"Warning: Pivoting failed. DataFrame remains unchanged. Error: {str(e)}")
            return self

    def applymap(self, fn):
        """
                Applies a function element-wise to the DataFrame.

                Args:
                    fn (callable): The function to apply to each element in the DataFrame.

                Returns:
                    RichPipe: A RichPipe object with the element-wise transformed DataFrame.
        """
        new_df = self.df.applymap(fn)
        return RichPipe(new_df, self._id)

    def to_csv(self, filepath, **kwargs):
        """
                Saves the DataFrame to a CSV file.

                Args:
                    filepath (str): The path to the CSV file to save.
                    **kwargs: Additional keyword arguments to be passed to the `to_csv` method of pandas.DataFrame.

                Returns:
                    RichPipe: A RichPipe object (the DataFrame is saved to the specified file).
        """
        self.df.to_csv(filepath, **kwargs)
        return self

    @staticmethod
    def from_csv(filepath, **kwargs):
        """
                Reads a DataFrame from a CSV file.

                Args:
                    filepath (str): The path to the CSV file to read from.
                    **kwargs: Additional keyword arguments to be passed to the `read_csv` method of pandas.DataFrame.

                Returns:
                    RichPipe: A RichPipe object containing the loaded DataFrame.
        """
        df = pd.read_csv(filepath, **kwargs)
        return RichPipe(df, uuid.uuid4())

    def to_dict(self):
        """
                Converts the dataframe to python dictionary

                Returns:
                    Dict: A python dictionary
        """
        return self.df.to_dict()

    def sort(self, by, ascending=True):
        """
                Sorts the DataFrame by the specified columns.

                Args:
                    by (str or list of str): The column(s) to sort by.
                    ascending (bool or list of bool, optional): Sort ascending vs. descending. Defaults to True.

                Returns:
                    RichPipe: A RichPipe object with the sorted DataFrame.
        """
        self.df = self.df.sort_values(by=by, ascending=ascending)
        return self

    def drop_duplicates(self, subset=None, keep='first'):
        """
                Drops duplicate rows from the DataFrame.

                Args:
                    subset (str or list of str, optional): Only consider certain columns for identifying duplicates.
                    keep (str, optional): Which duplicates to keep. Defaults to 'first'.

                Returns:
                    RichPipe: A RichPipe object with duplicates dropped.
        """
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        return self

    def fill_na(self, value, method=None, axis=None):
        """
                Fills missing values in the DataFrame.

                Args:
                    value (scalar, dict, Series, or DataFrame): Value to use to fill holes.
                    method (str, optional): Method to use for filling holes.
                    axis (int, optional): Axis along which to fill missing values.

                Returns:
                    RichPipe: A RichPipe object with missing values filled.
        """
        self.df = self.df.fillna(value=value, method=method, axis=axis)
        return self

    def rename_columns(self, columns):
        """
                Renames the columns of the DataFrame.

                Args:
                    columns (dict): A dictionary mapping old column names to new column names.

                Returns:
                    RichPipe: A RichPipe object with renamed columns.
        """
        try:
            self.df = self.df.rename(columns=columns)
            return self
        except KeyError as err:
            raise Exception(err)

    def reset_index(self, drop=False):
        """
                Resets the index of the DataFrame.

                Args:
                    drop (bool, optional): Do not try to insert index into the DataFrame columns. Defaults to False.

                Returns:
                    RichPipe: A new RichPipe object with the index reset.
        """
        self.df = self.df.reset_index(drop=drop)
        return self

    def add_column(self, name, value):
        """
                Adds a new column to the DataFrame.

                Args:
                    name (str): The name of the new column.
                    value (any): The value to populate the new column with.

                Returns:
                    RichPipe: A RichPipe object with the added column.
        """
        self.df[name] = value
        return self

    def drop_column_by_index(self, index):
        """
                Drops a column by its index.

                Args:
                    index (int): The index of the column to drop.

                Returns:
                    RichPipe: A RichPipe object with the column dropped.
        """
        col_name = self.df.columns[index]
        self.df.drop(columns=[col_name], inplace=True)
        return self

    def filter_rows_by_index(self, indices):
        """
                Filters rows by their index values.

                Args:
                    indices (list): A list of indices to keep.

                Returns:
                    RichPipe: A RichPipe object with filtered rows.
        """
        self.df = self.df.loc[indices]
        return self

    def describe(self):
        """
                Generates descriptive statistics of the DataFrame.

                Returns:
                    pd.DataFrame: A DataFrame with descriptive statistics.
        """
        return self.df.describe()

    def print_df(self):
        """
                Prints the DataFrame to the console.

                Returns:
                    None
        """
        print(self.df)

    def to_excel(self, filepath, **kwargs):
        """
                Saves the DataFrame to an Excel file.

                Args:
                    filepath (str): The path to save the Excel file.
                    **kwargs: Additional keyword arguments passed to pandas.DataFrame.to_excel.

                Returns:
                    RichPipe: A RichPipe object (the DataFrame is saved to the specified file).
        """
        self.df.to_excel(filepath, **kwargs)
        return self

    def export_to_json(self, filepath, **kwargs):
        """
                Exports the DataFrame to a JSON file.

                Args:
                    filepath (str): The path to save the JSON file.
                    **kwargs: Additional keyword arguments passed to pandas.DataFrame.to_json.

                Returns:
                    RichPipe: A RichPipe object (the DataFrame is saved to the specified file).
        """
        self.df.to_json(filepath, **kwargs)
        return self

    @staticmethod
    def _df(data: dict):
        """
                Builds a RichPipeline from a python Dictionary

                Args:
                    data (dict): A python dictionary for building RichPipe ops

                Returns:
                     RichPipe: A RichPipe object
        """
        return RichPipe(pd.DataFrame(data), uuid.uuid4())

    def one_hot_encode(self, columns):
        """
                Performs one-hot encoding on specified columns.

                Args:
                    columns (list): List of column names to encode.

                Returns:
                    RichPipe: A RichPipe object with one-hot encoded columns.
        """
        self.df = pd.get_dummies(self.df, columns=columns)
        return self

    def label_encode(self, column):
        """
                Performs label encoding on a specified column.

                Args:
                    column (str): Name of the column to encode.

                Returns:
                    RichPipe: A RichPipe object with label encoded column.
        """
        le = LabelEncoder()
        self.df[column] = le.fit_transform(self.df[column])
        return self

    def standard_scale(self, columns):
        """
                Performs standard scaling on specified columns.

                Args:
                    columns (list): List of column names to scale.

                Returns:
                    RichPipe: A RichPipe object with scaled columns.
        """
        scaler = StandardScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])
        return self

    def min_max_scale(self, columns):
        """
                Performs min-max scaling on specified columns.

                Args:
                    columns (list): List of column names to scale.

                Returns:
                    RichPipe: A RichPipe object with scaled columns.
        """
        scaler = MinMaxScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])
        return self

    def winsorize(self, columns, limits=(0.05, 0.05)):
        """
                Performs winsorization on specified columns.

                Args:
                    columns (list): List of column names to winsorize.
                    limits (tuple, optional): Tuple of lower and upper limits for winsorization.
                    Defaults to (0.05, 0.05).

                Returns:
                    RichPipe: A RichPipe object with winsorized columns.
        """
        self.df[columns] = self.df[columns].apply(lambda x: mstats.winsorize(x, limits=limits))
        return self

    def robust_scale(self, columns):
        """
                Performs robust scaling on specified columns.

                Args:
                    columns (list): List of column names to scale.

                Returns:
                    RichPipe: A RichPipe object with scaled columns.
        """
        scaler = RobustScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])
        return self

    def corr(self, method='pearson'):
        """
                Computes pairwise correlation of columns.

                Args:
                    method (str, optional): Correlation method (pearson, kendall, spearman). Defaults to 'pearson'.

                Returns:
                    pd.DataFrame: A DataFrame of pairwise correlation.
        """
        return self.df.corr(method=method)

    def set_index(self, keys, drop=True, append=False, inplace=False):
        """
                Sets the DataFrame index (row labels) using one or more existing columns.

                Args:
                    keys (str or list of str): Column(s) to use as index.
                    drop (bool, optional): Whether to drop columns used as index. Defaults to True.
                    append (bool, optional): Whether to append columns to existing index. Defaults to False.
                    inplace (bool, optional): Whether to modify the DataFrame in place. Defaults to False.

                Returns:
                    RichPipe: A RichPipe object with updated index.
        """
        self.df.set_index(keys=keys, drop=drop, append=append, inplace=inplace)
        return self

    def drop_index(self):
        """
                Drops the current index from the DataFrame.

                Returns:
                    RichPipe: A new RichPipe object with the index dropped.
        """
        self.df = self.df.reset_index(drop=True)
        return self

    def datetime_features(self, column):
        """
                Extracts datetime features from a datetime column.

                Args:
                    column (str): Name of the datetime column.

                Returns:
                    RichPipe: A new RichPipe object with extracted datetime features.
        """
        self.df[column + '_year'] = self.df[column].dt.year
        self.df[column + '_month'] = self.df[column].dt.month
        self.df[column + '_day'] = self.df[column].dt.day
        self.df[column + '_hour'] = self.df[column].dt.hour
        self.df[column + '_minute'] = self.df[column].dt.minute
        self.df[column + '_second'] = self.df[column].dt.second
        return self

    def select_dtypes(self, include=None, exclude=None):
        """
                Selects columns based on their dtype.

                Args:
                    include (str or list, optional): Dtypes to include (e.g., 'number', 'object').
                    exclude (str or list, optional): Dtypes to exclude (e.g., 'number', 'object').

                Returns:
                    RichPipe: A new RichPipe object with selected columns.
        """
        self.df = self.df.select_dtypes(include=include, exclude=exclude)
        return self

    def to_datetime(self, column, _format=None):
        """
                Converts a column to datetime format.

                Args:
                    column (str): Name of the column to convert.
                    _format (str, optional): Format string to parse datetime.

                Returns:
                    RichPipe: A new RichPipe object with the converted datetime column.
        """
        self.df[column] = pd.to_datetime(self.df[column], format=_format)
        return self

    def get_df(self):
        """
                This returns a data frame object of the pipe

                Returns:
                     A pandas DataFrame object representing the pipe
        """
        df = self.df
        return df

    def __str__(self):
        """
                Returns:
                     a json string for the data representation
        """
        return json.dumps(self.to_dict())

    def __iter__(self):
        """
                Generator to yield rows of the DataFrame one by one.
        """
        for _, row in self.df.iterrows():
            yield row

    def from_json_str(self, j_str):
        """
                Args:
                    j_str: A json string object

                Returns:
                     RichPipe: object from reading data from a json-string
        """
        try:
            return self._df(json.loads(j_str))
        except json.JSONDecodeError:
            raise Exception("Please pass a correct json string object")

    def to_generator(self):
        """
                Converts the RichPipe object to a generator that yields each row as a dictionary.

                Returns:
                    generator: A generator yielding each row of the DataFrame as a dictionary.
        """
        return iter(self)

    def train_test_split(self, test_size=0.2, random_state=None, stratify=None):
        """
                Splits the DataFrame into random train and test subsets.

                Args:
                    test_size (float or int, optional): If float, should be between 0.0 and
                    1.0 and represent the proportion of the dataset to include in the test split.
                    If int, represents the absolute number of test samples. Defaults to 0.2.
                    random_state (int, RandomState instance or None, optional):
                    Controls the shuffling applied to the data before applying the split.
                     Pass an int for reproducible output across multiple function calls. Defaults to None.
                    stratify (array-like, optional): If not None, data is split in a stratified fashion,
                    using this as the class labels.

                Returns:
                    tuple: A tuple containing RichPipe objects for train and test subsets.
        """
        train_df, test_df = train_test_split(self.df,
                                             test_size=test_size,
                                             random_state=random_state,
                                             stratify=stratify)
        return RichPipe(train_df, uuid.uuid4()), RichPipe(test_df, uuid.uuid4())

    def knn_impute(self, n_neighbors=5, weights='uniform', missing_values=np.nan):
        """
                Imputes missing values in the DataFrame using KNN imputation from scikit-learn.

                Args:
                    n_neighbors (int, optional): The number of nearest neighbors to consider for imputation.
                     Defaults to 5.
                    weights (str, optional): The weight function used in neighbor averaging
                    ('uniform', 'distance', or a callable). Defaults to 'uniform'.
                    missing_values (any, optional): The value(s) to consider as missing. Defaults to np.nan.

                Returns:
                    pd.DataFrame: The DataFrame with imputed values.
        """

        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        self.df = pd.DataFrame(imputer.fit_transform(self.df), columns=self.df.columns)
        return self

    def iqr_outliers(self, column_name, threshold=1.5):
        """
                Detects outliers in a column using Interquartile Range (IQR).

                Args:
                    column_name (str): The name of the column to detect outliers in.
                    threshold (float, optional): The number of standard deviations above/below the
                    IQR to consider outliers. Defaults to 1.5.

                Returns:
                    pd.DataFrame or pd.Series:
                        - DataFrame containing only non-outlier rows (default).
                        - Series containing outlier indices if `want_outlier_indices` is True.
        """

        q1 = self.df[column_name].quantile(0.25)
        q3 = self.df[column_name].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outliers = (self.df[column_name] < lower_bound) | (
                    self.df[column_name] > upper_bound)  # Element-wise comparison
        return self.df[~outliers] if not self.want_outlier_indices else self.df.index[outliers]

    def zscore_outliers(self, column_name, threshold=3):
        """
        Detects outliers in a column using Z-scores from scikit-learn.

        Args:
            column_name (str): The name of the column to detect outliers in.
            threshold (float, optional): The number of standard deviations to consider outliers. Defaults to 3.

        Returns:
            pd.DataFrame or pd.Series:
                - DataFrame containing only non-outlier rows.
                - Series containing outlier indices.
        """

        scaler = StandardScaler()
        # Exclude rows with missing values for accurate scaling
        scaler.fit(self.df[column_name].dropna().values.reshape(-1, 1))
        z_scores = scaler.transform(self.df[[column_name]])[:, 0]
        outliers = np.abs(z_scores) > threshold
        return self.df[~outliers] if not self._want_outlier_indices else self.df.index[outliers]

    def remove_iqr_outliers(self, column_name, threshold=1.5):
        """
                Removes outliers based on IQR detection.

                Args:
                    column_name (str): The name of the column to detect outliers in.
                    threshold (float, optional): The number of standard deviations
                    above/below the IQR to consider outliers. Defaults to 1.5.

                Returns:
                    pd.DataFrame: The DataFrame with outliers removed.
        """

        self.df = self.iqr_outliers(column_name, threshold)
        return self

    def remove_zscore_outliers(self, column_name, threshold=3):
        """
                Removes outliers based on Z-score detection.

                Args:
                    column_name (str): The name of the column to detect outliers in.
                    threshold (float, optional): The number of standard deviations
                     to consider outliers. Defaults to 3.

                Returns:
                    pd.DataFrame: The DataFrame with outliers removed.
        """

        self.df = self.zscore_outliers(column_name, threshold)
        return self

    def plot(self, kind='line', x=None, y=None, **kwargs):
        """
                Creates a plot of the DataFrame using matplotlib.

                Args:
                  kind (str, optional): Type of plot to create. Defaults to 'line'.
                  x (str, optional): Column name for the x-axis. Defaults to None (uses index).
                  y (str or list, optional): Column(s) to plot on the y-axis.
                  Defaults to None (plots all numerical columns).
                  **kwargs: Additional keyword arguments to be passed to the plotting function.

                Returns:
                  RichPipe: The RichPipe object for chaining.
        """

        if x is None:
            x = self.df.index

        if y is None:
            y = self.df.select_dtypes(include=[np.number])

        self.df.plot(kind=kind, x=x, y=y, **kwargs)
        plt.show()
        return self

    def to_parquet(self, filepath, **kwargs):
        """
                Saves the DataFrame to a Parquet file.

                Args:
                    filepath (str): The path to save the Parquet file.
                    **kwargs: Additional keyword arguments passed to pandas.DataFrame.to_parquet.

                Returns:
                    RichPipe: A RichPipe object (the DataFrame is saved to the specified file).
        """
        self.df.to_parquet(filepath, **kwargs)
        return self

    @staticmethod
    def from_parquet(filepath, **kwargs):
        """
                Reads a DataFrame from a Parquet file.

                Args:
                    filepath (str): The path to the Parquet file to read from.
                    **kwargs: Additional keyword arguments passed to pandas.read_parquet.

                Returns:
                    RichPipe: A RichPipe object containing the loaded DataFrame.
        """
        df = pd.read_parquet(filepath, **kwargs)
        return RichPipe(df, uuid.uuid4())

    def to_numpy(self):
        """
                Converts the DataFrame to a NumPy array.

                Returns:
                    np.ndarray: A NumPy array representation of the DataFrame.
        """
        return self.df.to_numpy()

    def clip(self, lower=None, upper=None):
        """
                Clips values in the DataFrame to a specified range.

                Args:
                    lower (float, optional): Minimum threshold value.
                    upper (float, optional): Maximum threshold value.

                Returns:
                    RichPipe: A RichPipe object with values clipped.
        """
        self.df = self.df.clip(lower=lower, upper=upper)
        return self

    def log_transform(self, column):
        """
                Performs logarithmic transformation on a specified column.

                Args:
                    column (str): Name of the column to transform.

                Returns:
                    RichPipe: A RichPipe object with the transformed column.
        """
        self.df[column] = np.log1p(self.df[column])  # log1p handles log(0) cases
        return self

    def sqrt_transform(self, column):
        """
                Performs square root transformation on a specified column.

                Args:
                    column (str): Name of the column to transform.

                Returns:
                    RichPipe: A RichPipe object with the transformed column.
        """
        self.df[column] = np.sqrt(self.df[column])
        return self

    def power_transform(self, column, power):
        """
                Performs power transformation on a specified column.

                Args:
                    column (str): Name of the column to transform.
                    power (float): The power to which each value should be raised.

                Returns:
                    RichPipe: A RichPipe object with the transformed column.
        """
        self.df[column] = np.power(self.df[column], power)
        return self

    def inverse_transform(self, column):
        """
                Performs inverse transformation on a specified column.

                Args:
                    column (str): Name of the column to transform.

                Returns:
                    RichPipe: A RichPipe object with the transformed column.
        """
        self.df[column] = 1 / self.df[column]
        return self

    def sample(self, n=None, frac=None, random_state=None):
        """
                Samples rows from the DataFrame.

                Args:
                    n (int, optional): Number of rows to return.
                    frac (float, optional): Fraction of rows to return.
                    random_state (int, optional): Seed for the random number generator.

                Returns:
                    RichPipe: A RichPipe object with the sampled rows.
        """
        self.df = self.df.sample(n=n, frac=frac, random_state=random_state)
        return self

    def date_range(self, start=None, end=None, periods=None, freq=None, tz=None, normalize=False, name=None,
                   closed=None, **kwargs):
        """
        Create a range of dates.

        Parameters:
        - start, end, periods, freq, tz, normalize, name, closed: Arguments to pandas.date_range().

        Returns:
        - RichPipe object with the date range added as a new column.
        """
        date_range = pd.date_range(start=start, end=end, periods=periods, freq=freq, tz=tz, normalize=normalize,
                                   name=name, closed=closed, **kwargs)
        self.df[name or 'date_range'] = date_range
        return self

    def to_period(self, freq='D', axis=0):
        """
        Convert DataFrame to specified frequency period.

        Parameters:
        - freq: Frequency alias. Default is 'D' (daily).
        - axis: Axis to convert. Default is 0 (index).

        Returns:
        - RichPipe object with the DataFrame converted to period.
        """
        self.df = self.df.to_period(freq=freq, axis=axis)
        return self

    def to_timestamp(self, freq=None, how='start', axis=0):
        """
        Convert DataFrame to specified timestamp.

        Parameters:
        - freq: Frequency alias or Offset object. Default is None.
        - how: {'start', 'end', 'nearest', 'previous', 'next'}, default 'start'.
        - axis: Axis to convert. Default is 0 (index).

        Returns:
        - RichPipe object with the DataFrame converted to timestamp.
        """
        self.df = self.df.to_timestamp(freq=freq, how=how, axis=axis)
        return self

    def memory_usage(self, deep=False):
        """
        Display memory usage of the DataFrame.

        Parameters:
        - deep: If True, introspect the data deeply for memory consumption. Default is False.

        Returns:
        - Memory usage of the DataFrame.
        """
        return self.df.memory_usage(deep=deep)

    def query(self, expr):
        """
        Query the DataFrame with a boolean expression.

        Parameters:
        - expr: The boolean expression to query the DataFrame.

        Returns:
        - RichPipe object with the DataFrame filtered by the boolean expression.
        """
        self.df = self.df.query(expr)
        return self

    def to_sparse(self, fill_value=0):
        """
        Convert DataFrame to SparseDataFrame.

        Parameters:
        - fill_value: Fill value for the sparse representation. Default is 0.

        Returns:
        - RichPipe object with the DataFrame converted to SparseDataFrame.
        """
        self.df = self.df.to_sparse(fill_value=fill_value)
        return self

    def cumsum(self, axis=0, skipna=True):
        """
        Compute cumulative sum of the DataFrame.

        Parameters:
        - axis: Axis to perform operation on. Default is 0 (index).
        - skipna: Exclude NA/null values. Default is True.

        Returns:
        - RichPipe object with the DataFrame computed cumulative sum.
        """
        self.df = self.df.cumsum(axis=axis, skipna=skipna)
        return self

    def cumprod(self, axis=0, skipna=True):
        """
        Compute cumulative product of the DataFrame.

        Parameters:
        - axis: Axis to perform operation on. Default is 0 (index).
        - skipna: Exclude NA/null values. Default is True.

        Returns:
        - RichPipe object with the DataFrame computed cumulative product.
        """
        self.df = self.df.cumprod(axis=axis, skipna=skipna)
        return self

    def cummax(self, axis=0, skipna=True):
        """
        Compute cumulative maximum of the DataFrame.

        Parameters:
        - axis: Axis to perform operation on. Default is 0 (index).
        - skipna: Exclude NA/null values. Default is True.

        Returns:
        - RichPipe object with the DataFrame computed cumulative maximum.
        """
        self.df = self.df.cummax(axis=axis, skipna=skipna)
        return self

    def cummin(self, axis=0, skipna=True):
        """
        Compute cumulative minimum of the DataFrame.

        Parameters:
        - axis: Axis to perform operation on. Default is 0 (index).
        - skipna: Exclude NA/null values. Default is True.

        Returns:
        - RichPipe object with the DataFrame computed cumulative minimum.
        """
        self.df = self.df.cummin(axis=axis, skipna=skipna)
        return self

    def diff(self, periods=1, axis=0):
        """
        Compute the difference between consecutive rows of the DataFrame.

        Parameters:
        - periods: Number of periods to shift for calculating difference. Default is 1.
        - axis: Axis along which the difference is computed. Default is 0 (index).

        Returns:
        - RichPipe object with the DataFrame computed difference.
        """
        self.df = self.df.diff(periods=periods, axis=axis)
        return self

    def stack(self, level=-1, dropna=True):
        """
        Stack the prescribed level(s) from columns to index.

        Parameters:
        - level: Level(s) to stack from column axis onto index axis. Default is -1 (last level).
        - dropna: Whether to drop rows in the resulting DataFrame with all NA values. Default is True.

        Returns:
        - RichPipe object with the DataFrame stacked.
        """
        self.df = self.df.stack(level=level, dropna=dropna)
        return self

    def unstack(self, level=-1, fill_value=None):
        """
        Unstack the prescribed level(s) from index to column(s).

        Parameters:
        - level: Level(s) to unstack from index axis onto column axis. Default is -1 (last level).
        - fill_value: Replace NaN with this value if the unstack produces missing values. Default is None.

        Returns:
        - RichPipe object with the DataFrame unstacked.
        """
        self.df = self.df.unstack(level=level, fill_value=fill_value)
        return self

    def execute_task_async(self, executor, method_name, *args, **kwargs):
        """
        Executes a RichPipe method asynchronously using TaskExecutor.
        Args:
            executor (TaskExecutor): The TaskExecutor instance.
            method_name (str): The name of the method to execute.
            *args: Positional arguments for the method.
            **kwargs: Keyword arguments for the method.
        """
        task_tag = str(uuid.uuid4())  # Unique identifier for the task
        task = (self, method_name, args, kwargs)
        executor.execute_async(task_tag, task)
        return task_tag


class State(enum.Enum):
    SUCCESS = 0
    FAILED = 1


class TaskWorker(threading.Thread):
    def __init__(self, task_queue: queue.Queue, result_queue: queue.Queue):
        threading.Thread.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        while True:
            key, task = self.task_queue.get()
            if key is None:
                self.task_queue.task_done()
                break
            try:
                if isinstance(task, tuple) and len(task) == 4:
                    richpipe, method_name, args, kwargs = task
                    method = getattr(richpipe, method_name)
                    result = method(*args, **kwargs)
                    state = (State.SUCCESS, result)
                else:
                    raise ValueError("Invalid task format")
            except Exception as e:
                state = (State.FAILED, str(e))
                logging.error(str(e))
            self.result_queue.put((key, state))
            self.task_queue.task_done()


class TaskExecutor:
    def __init__(self, p):
        self.wq = queue.Queue()
        self.rq = queue.Queue()
        self.commands = {}
        self.event_buffer = {}
        self.workers = [TaskWorker(self.wq, self.rq) for _ in range(p)]

    def start(self):
        for w in self.workers:
            w.start()

    def execute_async(self, tag, task):
        self.wq.put_nowait((tag, task))

    def change_state(self, key, state):
        self.commands[key] = state
        self.event_buffer[key] = state

    def heartbeat(self):
        while not self.rq.empty():
            key, (state, result) = self.rq.get()
            self.change_state(key, (state, result))

    def end(self):
        [self.wq.put((None, None)) for w in self.workers]
        self.wq.join()


# def main():
#     logging.basicConfig(level=logging.INFO)
#
#     # Initialize the commands dictionary to track command states
#     commands = {}
#
#     try:
#         # Example commands
#         command1_id = str(uuid.uuid4())
#         command2_id = str(uuid.uuid4())
#         command3_id = str(uuid.uuid4())
#
#         # Create RichPipe instances with sample data
#         data1 = pd.DataFrame({"id": [1, 2, 3], "value1": [10, 20, 30]})
#         data2 = pd.DataFrame({"id": [2, 3, 4], "value2": [200, 300, 400]})
#
#         pipe1 = RichPipe(data1)
#         pipe2 = RichPipe(data2)
#
#         # Execute inner join
#         joined_pipe = pipe1.inner_join("id", "id", pipe2)
#         result = joined_pipe.df.to_dict()
#
#         # Track command states
#         commands[command1_id] = (State.SUCCESS, pipe1)
#         commands[command2_id] = (State.SUCCESS, pipe2)
#         commands[command3_id] = (State.SUCCESS, joined_pipe)
#
#         # Log current state of commands
#         logging.info(f"Current state of commands: {commands}")
#
#         # Validate results
#         expected_result = {
#             "id": {0: 2, 1: 3},
#             "value1": {0: 20, 1: 30},
#             "value2": {0: 200, 1: 300}
#         }
#         logging.info(result)
#         assert result == expected_result, f"Result does not match expected: {result} != {expected_result}"
#
#         logging.info("All results validated successfully.")
#
#     except Exception as e:
#         logging.error(f"An error occurred: {e}")
#         raise
#
#     # Final state of commands
#     logging.info(f"Final state of commands: {commands}")
#
#
# if __name__ == "__main__":
#     main()
#
