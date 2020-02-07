# -*- coding: utf-8 -*-
"""Example Google style docstrings.

This module demonstrates documentation as specified by the `Google Python
Style Guide`_. Docstrings may extend over multiple lines. Sections are created
with a section header and a colon followed by a block of indented text.

Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python example_google.py

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.

Attributes:
    module_level_variable1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""
import pandas as pd
import numpy as np
pd.set_option("display.precision", 2)
pd.options.display.float_format = '{:.2f}'.format


class Basic:
    """Exceptions are documented in the same way as classes.

    The __init__ method may be documented in either the class level
    docstring, or as a docstring on the __init__ method itself.

    Either form is acceptable, but the two should not be mixed. Choose one
    convention to document the __init__ method and be consistent with it.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        msg (str): Human readable string describing the exception.
        code (:obj:`int`, optional): Error code.

    Attributes:
        msg (str): Human readable string describing the exception.
        code (int): Exception error code.

    """

    def __init__(self, data):
        """Example of docstring on the __init__ method.

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1 (str): Description of `param1`.
            param2 (:obj:`int`, optional): Description of `param2`. Multiple
                lines are supported.
            param3 (:obj:`list` of :obj:`str`): Description of `param3`.

        """
        self.data = data

    def show(self):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        columns = []
        datatypes = []
        length = []
        unique = []
        nas = []
        pmv = []
        mean = []
        median = []
        mode = []
        drange = []
        for i in self.data.columns:
            columns.append(i)
            datatypes.append(self.data[i].dtypes)
            length.append(len(self.data[i]))
            unique.append(len(self.data[i].unique()))
            nas.append(self.data[i].isna().sum())
            pmv.append((self.data[i].isna().sum()/len(self.data[i]))*100)
            if self.data[i].dtypes == 'int64' or self.data[i].dtypes == 'float64':
                mean.append(self.data[i].mean())
                median.append(self.data[i].median())
                mode.append('NA')
                drange.append(str(self.data[i].min())+'-'+str(self.data[i].max()))
            else:
                mean.append('NA')
                median.append('NA')
                mode.append(self.data[i].mode()[0])
                drange.append('NA')

        print("Dimensions of Dataframe-Rows:{},Columns:{}".format(self.data.shape[0], self.data.shape[1]))

        df = pd.DataFrame({'Column Name': columns,
                            'Data Type': datatypes,
                            'Count': length,
                            'No. of Unique': unique,
                            'Missing Values': nas,
                            '% Missing Values': pmv,
                            'Mean': mean,
                            'Median': median,
                            'Mode': mode,
                            'Range': drange})
        display(df)

    def Distributions(self):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        cols = 3
        rows = int(np.ceil(len(numeric_cols)/3))
        fig, axes = plt.subplots(ncols=cols, nrows=rows,figsize=(20,12+rows))

        for i, ax in zip(numeric_cols, axes.flat):
            sns.distplot(self.data[i].dropna(), hist=True, ax=ax);
        plt.show()

        catogorical_cols = []
        higher_categories = []
        for i in self.data.select_dtypes(exclude=[np.number]).columns.tolist():
            if len(self.data[i].unique())<15:
                catogorical_cols.append(i)
            else:
                higher_categories.append(i)

        cols = 3
        rows = int(np.ceil(len(catogorical_cols)/3))
        if rows <=2:
            fig, axes = plt.subplots(ncols=cols, nrows=1,figsize=(20,12+rows))
        else:
            fig, axes = plt.subplots(ncols=cols, nrows=rows,figsize=(20,8))

        for i, ax in zip(catogorical_cols, axes.flat):
            sns.countplot(self.data[i].dropna(), ax=ax);
        plt.show()
        print('Categorical columns with more than 15 categories : {}'.format(higher_categories))
        for i in higher_categories:
            if self.data[i].dropna().shape[0] != data.shape[0]:
                plt.figure(figsize=(20,8))
                sns_plot = sns.countplot(self.data[i].dropna());
                sns_plot.set_xticklabels(sns_plot.get_xticklabels(), rotation=40, ha="right")
                plt.tight_layout()
                plt.show()
