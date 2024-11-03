from sklearn.model_selection import train_test_split

class DataUtils:
    """
    DataUtils provides common data manipulation and utility functions.

    Methods
    -------
    handle_missing_values(df, method="mean"):
        Handle missing values in a dataframe.
    split_data(X, y, test_size=0.2, random_state=42):
        Split data into training and test sets.
    """

    @staticmethod
    def handle_missing_values(df, method="mean"):
        """
        Handle missing values in a dataframe.

        Parameters
        ----------
        df : DataFrame
            Dataframe with missing values.
        method : str, optional
            Method to handle missing values ('mean', 'median', 'mode'). Default is 'mean'.

        Returns
        -------
        DataFrame
            Dataframe with missing values handled.
        """
        if method == "mean":
            return df.fillna(df.mean())
        elif method == "median":
            return df.fillna(df.median())
        elif method == "mode":
            return df.fillna(df.mode().iloc[0])
        else:
            raise ValueError("Method must be 'mean', 'median', or 'mode'.")

    @staticmethod
    def split_data(X, y, test_size=0.2, random_state=42):
        """
        Split data into training and test sets.

        Parameters
        ----------
        X : array-like
            Features data.
        y : array-like
            Target data.
        test_size : float, optional
            Proportion of the dataset to include in the test split. Default is 0.2.
        random_state : int, optional
            Controls the shuffling applied to the data before splitting. Default is 42.

        Returns
        -------
        tuple
            Training and test splits of X and y.
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
