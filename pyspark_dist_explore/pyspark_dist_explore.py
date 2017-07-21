from scipy.interpolate import spline
from pyspark.sql.types import NumericType

import pyspark.sql.functions as F
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def hist(axis, x, **kwargs):
    histogram = create_histogram_object(kwargs)
    histogram.add_data(x)
    return histogram.plot_hist(axis, **kwargs)


def distplot(axis, x, **kwargs):
    histogram = create_histogram_object(kwargs)
    histogram.add_data(x)
    _, _, patches = histogram.plot_hist(axis, normed=True, **kwargs)
    colors = [patch[0].get_facecolor() for patch in patches]
    histogram.plot_density(axis, color=colors)


def pandas_histogram(x, bins=10, range=None):
    histogram = Histogram(bins=bins, range=range)
    histogram.add_data(x)
    return histogram.to_pandas()


def create_histogram_object(kwargs):
    bins = 10
    range = None

    if 'bins' in kwargs:
        bins = kwargs['bins']
        del kwargs['bins']

    if 'range' in kwargs:
        range = kwargs['range']
        del kwargs[range]

    return Histogram(bins=bins, range=range)


class Histogram(object):
    """The Histogram object leverages Spark to calculate histograms, and matplotlib to visualize these.

    Args:
        range: (:obj: `tuple`, optional): The lower and upper range of the bins.
            Lower and upper outliers are ignored. If not provided, range is (min(x), max(x)). Range has no
            effect if bins is a sequence. If bins is a sequence or range is specified, autoscaling is
            based on the specified bin range instead of the range of x.
        bins ((:obj:int or `list` of :obj:`str` or `list of :obj:`int`, optional):
            If an integer is given: Number of bins in the histogram. Defaults to 10.
            If a list is given: Predefined list of bin boundaries.
            The bins are all open to the right except for the last which is closed. e.g. [1,10,20,50] means
            the buckets are [1,10) [10,20) [20,50], which means 1<=x<10, 10<=x<20, 20<=x<=50.

    """
    def __init__(self, bins=10, range=None, use_log10=False):
        # todo: fix use_log10
        self.col_list = []
        self.bin_list = []
        self.hist_dict = {}
        self.nr_bins = None
        self.min_value = None
        self.max_value = None
        self.useLog10 = use_log10
        self.is_build = False

        if isinstance(bins, list):
            self.bin_list = [float(bin_border) for bin_border in bins]
        else:
            self.nr_bins = bins

        if range is not None:
            self.min_value = range[0]
            self.max_value = range[1]

    def add_column(self, table):
        """Add single column DataFrame to the histogram object.

        If multiple columns share the same name, a (n) will be appended to the name, where n is
        the next available number.

        Args:
            table (:obj:`dataframe`): A pyspark dataframe with a single column

        """
        if len(table.columns) > 1:
            raise ValueError('More then one column is being added, use add_data() to add multi-column DataFrames')

        column_name = table.columns[0]

        if not isinstance(table.schema.fields[0].dataType, NumericType):
            raise ValueError('Column %s has a non-numeric type (%s), only numeric types are supported'
                             % (column_name, str(table.schema.fields[0].dataType)))

        self.col_list.append((table, column_name))

    def _get_bin_centers(self):
        result = []
        for i in range(len(self.bin_list)-1):
            result.append(((self.bin_list[i + 1] - self.bin_list[i]) / 2) + self.bin_list[i])
        return result

    def _get_col_names(self):
        new_col_names = []
        for i in range(len(self.bin_list) - 1):
            if self.useLog10:
                new_col_names.append('%.2f - %.2f' % (pow(10, self.bin_list[i]), (pow(10, self.bin_list[i + 1]))))
            else:
                new_col_names.append('%.2f - %.2f' % (self.bin_list[i], self.bin_list[i + 1]))
        return new_col_names

    def _check_col_name(self, column_name):
        n = 0
        col_name_new = column_name
        while col_name_new in self.hist_dict.keys():
            n += 1
            col_name_new = '%s (%d)' % (column_name, n)
        return col_name_new

    def _get_min_value(self):
        if self.min_value is not None:
            return self.min_value
        return min([table.select(F.min(F.col(col_name))).collect()[0][0]
                    for table, col_name in self.col_list])

    def _get_max_value(self):
        if self.max_value is not None:
            return self.max_value
        return max([table.select(F.max(F.col(col_name))).collect()[0][0]
                    for table, col_name in self.col_list])

    def _calculate_bins(self):
        if len(self.bin_list) > 0:
            return self.bin_list

        if len(self.bin_list) == 0 and len(self.col_list) == 1 \
                and self.min_value is None and self.max_value is None:
            # Only use the amount of bins as input For the histogram function
            return self.nr_bins

        min_value = self._get_min_value()
        max_value = self._get_max_value()
        step = (float(max_value) - float(min_value)) / self.nr_bins
        return [min_value + (step * float(bn_nr)) for bn_nr in range(self.nr_bins + 1)]

    def _add_hist(self, table, column_name):
        # Uses spark to calculate the hist values
        hist = table.select(column_name).rdd.flatMap(lambda x: x).histogram(self.bin_list)
        self.hist_dict[self._check_col_name(column_name)] = hist[1]

        if isinstance(self.bin_list, int):
            self.bin_list = hist[0]

    @staticmethod
    def _convert_number_bmk(axis_value, _):
        """Converts the values on axes to Billions, Millions or Thousands"""
        if axis_value >= 1e9:
            return '{:1.1f}B'.format(axis_value * 1e-9)
        if axis_value >= 1e6:
            return '{:1.1f}M'.format(axis_value * 1e-6)
        if axis_value >= 1e3:
            return '{:1.1f}K'.format(axis_value * 1e-3)
        if axis_value >= 1 or axis_value == 0:
            return '{:1.0f}'.format(axis_value)
        return axis_value

    def build(self):
        """Calculates the histogram values for each of the columns.

        If the Histogram has already been build, it doesn't build it again.
        """
        if not self.is_build:
            self.bin_list = self._calculate_bins()
            for table, column_name in self.col_list:
                self._add_hist(table, column_name)
            self.is_build = True

    # def _scale_list(self, list):
    #     hist_sum = sum(list)
    #     if hist_sum > 0:
    #         return [float(bin) / float(hist_sum) for bin in list]
    #     else:
    #         return list

    def to_pandas(self, kind='hist'):
        """Returns a pandas dataframe from the Histogram object.

        This function calculates the Histogram function in Spark if it was not done yet.

        Args:
            kind (:obj:`str`, optional): 'hist' or 'density'. When using hist this returns the histogram object
            as pandas dataframe. When using density the index contains the bin centers, and the values in the
            dataframe are the scaled values. Defaults to 'hist'

        Returns:
            A pandas DataFrame from the Histogram object.
        """
        self.build()
        if kind == 'hist':
            return pd.DataFrame(self.hist_dict).set_index([self._get_col_names()])
        elif kind == 'density':
            result = pd.DataFrame(self.hist_dict).set_index([self._get_bin_centers()])
            return result.apply(lambda x: x / x.max(), axis=0)

    def plot_hist(self, ax, overlapping=False, formatted_yaxis=True, **kwargs):
        """Returns a matplotlib style histogram (matplotlib.pyplot.hist)

        Uses the matplotlib object oriented interface to add a Histogram to an matplotlib Axes object.
        All named arguments from pyplot.hist can be used. A new argument called "type" makes it possible to
        make overlapping histogram plots.

        Args:
            ax (:obj:`Axes`): An matplotlib Axes object on which the histogram will be plot
            overlapping (:obj:`bool`, optional): If set to true, this will generate an overlapping plot.
            When set to False it will generate a normal grouped histogram. Defaults to False.
            formatted_yaxis (:obj:`bool`, optional). If set to true, the numbers on the yaxis will be formatted
            for better readability. E.g. 1500000 will become 1.5M. Defaults to True
            **kwargs: The keyword arguments as used in matplotlib.pyplot.hist
        """
        self.build()

        if formatted_yaxis:
            # Round the y-axis value to nearest thousand, million, or billion for readable y-axis
            formatter = plt.FuncFormatter(Histogram._convert_number_bmk)
            ax.yaxis.set_major_formatter(formatter)

        if overlapping:
            for colname in self.hist_dict:
                ax.hist(self._get_bin_centers(),
                        bins=self.bin_list,
                        alpha=0.5,
                        label=self.hist_dict.keys(),
                        weights=self.hist_dict[colname],
                        **kwargs
                        )
        else:
            weights_multi = [self.hist_dict[colname] for colname in self.hist_dict]
            return ax.hist([self._get_bin_centers()] * len(self.hist_dict),
                           bins=self.bin_list,
                           weights=weights_multi,
                           label=self.hist_dict.keys(),
                           **kwargs)
    
    def plot_density(self, ax, num=300, **kwargs):
        """Returns a density plot on an Pyplot Axes object.

        Args:
            ax (:obj:`Axes`): An matplotlib Axes object on which the histogram will be plot
            num (:obj:`int`): The number of x values the line is plotted on. Default: 300
            **kwargs: Keyword arguments that are passed on to the pyplot.plot function.
        """
        colors = []

        self.build()
        bin_centers = np.asarray(self._get_bin_centers())
        x_new = np.linspace(bin_centers.min(), bin_centers.max(), num)

        if 'color' in kwargs:
            colors = kwargs['color']
            del kwargs['color']

        power_smooth = []

        for (colname, bin_values) in self.hist_dict.items():
            normed_values, ble = np.histogram(self._get_bin_centers(),
                                              bins=self.bin_list,
                                              weights=bin_values,
                                              normed=True
                                              )

            power_smooth.append(x_new)
            power_smooth.append(spline(bin_centers, normed_values, x_new))

        lines = ax.plot(*power_smooth, **kwargs)

        for i, line in enumerate(lines):
            if len(colors) > 0:
                plt.setp(line, color=colors[i], label=list(self.hist_dict.keys())[i])
            else:
                plt.setp(line, label=list(self.hist_dict.keys())[i])

        return lines

    def add_data(self, data):
        """Ads 1 or more columns to a histogram

        Multiple options are available:
        * Add a single column dataframe
        * Add a list of single column dataframes
        * Add a dataframe with multiple columns

        Args:
            (:obj:`Data`): A single column Spark dataframe, a list of single column Spark
            dataframes, or a multi column Spark dataframe.
        """
        if isinstance(data, list):
            for df_column in data:
                self.add_column(df_column)

        elif len(data.columns) > 1:
            for col_name in data.columns:
                self.add_column(data.select(col_name))

        else:
            self.add_column(data)
