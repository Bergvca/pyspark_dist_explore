from scipy.interpolate import spline

try:
    from pyspark.sql.types import NumericType

    import pyspark.sql.functions as F
except:
    pass

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def hist(axis, x, overlapping=False, formatted_yaxis=True, **kwargs):
    """Plots a histogram on an Axis object

    Args:
        :axis: (`Axes`)
            An matplotlib Axes object on which the histogram will be plot.
        :x: (`DataFrame` or `list` of `DataFrame`)
            A DataFrame with one or more numerical columns, or a list of single numerical column DataFrames
        :overlapping: (`bool`, optional)
            Generate overlapping histograms.

            If set to true, this will generate an overlapping plot.
            When set to False it will generate a normal grouped histogram. Defaults to False.
        :formatted_yaxis: (`bool`, optional)
            If set to true, the numbers on the yaxis will be formatted
            for better readability. E.g. 1500000 will become 1.5M. Defaults to True

        :\*\*kwargs:
            The keyword arguments as used in matplotlib.pyplot.hist

    Returns:
        :n: (`array` or `list` of `arrays`)
            The values of the histogram bins. See normed and weights for a description of the possible semantics.
            If input x is an array, then this is an array of length nbins. If input is a sequence arrays
            [data1, data2,..], then this is a list of arrays with the values of the histograms for each of the
            arrays in the same order.
        :bins: (`array`)
            The edges of the bins.
            Length nbins + 1 (nbins left edges and right edge of last bin). Always a single array even
            when multiple data sets are passed in.
        :patches: (`list` or `list` of `lists`)
            Silent list of individual patches used to create the histogram or list of such lists if multiple
            input datasets.

    """
    histogram = create_histogram_object(kwargs)
    histogram.add_data(x)
    return histogram.plot_hist(axis, overlapping, formatted_yaxis, **kwargs)


def distplot(axis, x, **kwargs):
    """Plots a normalised histogram and a density plot on an Axes object

    Args:
        :axis: (`Axes`)
            An matplotlib Axes object on which the histogram will be plot.
        :x: (`DataFrame` or `list` of `DataFrame`)
            A DataFrame with one or more numerical columns, or a list of single numerical column DataFrames
        :\*\*kwargs:
            The keyword arguments as used in matplotlib.pyplot.hist. Normed is set to True

    Returns:
        :n: (`array` or `list` of `arrays`)
            The values of the histogram bins. See normed and weights for a description of the possible semantics.
            If input x is an array, then this is an array of length nbins. If input is a sequence arrays
            [data1, data2,..], then this is a list of arrays with the values of the histograms for each of the
            arrays in the same order.
        :bins: (`array`)
            The edges of the bins.
            Length nbins + 1 (nbins left edges and right edge of last bin). Always a single array even
            when multiple data sets are passed in.
        :patches: (`list` or `list` of `lists`)
            Silent list of individual patches used to create the histogram or list of such lists if multiple
            input datasets.
    """
    histogram = create_histogram_object(kwargs)
    histogram.add_data(x)
    n, bins, patches = histogram.plot_hist(axis, density=True, **kwargs)

    # If working with a list of DataFrames as input, patches will be a list of lists with Rectangle objects
    # We will get the color of the first Rectangle object. If there is only one DataFrame patches is a single list
    # Of Rectangle objects
    if type(x) == list and len(x) > 1:
        colors = [patch[0].get_facecolor() for patch in patches]
    elif type(patches[0]) is Rectangle:
        colors = [patches[0].get_facecolor()]
    else:
        raise TypeError("Unexpected Patch Type. Expected Rectangle")

    histogram.plot_density(axis, color=colors)
    return n, bins, patches


def pandas_histogram(x, bins=10, range=None):
    """Returns a pandas DataFrame with histograms of the Spark DataFrame

    Bin ranges are formatted as text an put on the Index.

    Args:
        :x: (`DataFrame` or `list` of `DataFrame`)
            A DataFrame with one or more numerical columns, or a list of single numerical column DataFrames
        :bins: (`integer` or `array_like`, optional)
            If an integer is given, bins + 1 bin edges are returned, consistently with numpy.histogram() for
            numpy version >= 1.3.

            Unequally spaced bins are supported if bins is a sequence.

            Default is 10
        :range: (tuple or None, optional)
            The lower and upper range of the bins. Lower and upper outliers are ignored.
            If not provided, range is (x.min(), x.max()). Range has no effect if bins is a sequence.

            If bins is a sequence or range is specified, autoscaling is based on the specified bin range instead
            of the range of x.

            Default is None
    """
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
        del kwargs['range']

    return Histogram(bins=bins, range=range)


class Histogram(object):
    """The Histogram object leverages Spark to calculate histograms, and matplotlib to visualize these.

    Args:
        :range: (`tuple`, optional)
            The lower and upper range of the bins.

            Lower and upper outliers are ignored. If not provided, range is (min(x), max(x)). Range has no
            effect if bins is a sequence. If bins is a sequence or range is specified, autoscaling is
            based on the specified bin range instead of the range of x.
        :bins: (`int` or `list` of `str` or `list of `int`, optional)
            If an integer is given: Number of bins in the histogram.

            Defaults to 10.

            If a list is given: Predefined list of bin boundaries.

            The bins are all open to the right except for the last which is closed. e.g. [1,10,20,50] means
            the buckets are [1,10) [10,20) [20,50], which means 1<=x<10, 10<=x<20, 20<=x<=50.

    """
    def __init__(self, bins=10, range=None):
        self.col_list = []
        self.bin_boundaries = []
        self.hist_dict = {}  # column names: bin weight lists pairs
        self.nr_bins = None
        self.min_value = None
        self.max_value = None
        self.is_build = False

        if isinstance(bins, list):
            self.bin_boundaries = [float(bin_border) for bin_border in bins]
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
            :table: (:obj:`dataframe`)
                A PySpark DataFrame with a single column

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
        for i in range(len(self.bin_boundaries) - 1):
            result.append(((self.bin_boundaries[i + 1] - self.bin_boundaries[i]) / 2) + self.bin_boundaries[i])
        return result

    def _get_col_names(self):
        new_col_names = []
        for i in range(len(self.bin_boundaries) - 1):
            new_col_names.append('%.2f - %.2f' % (self.bin_boundaries[i], self.bin_boundaries[i + 1]))
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
        if len(self.bin_boundaries) > 0:
            return self.bin_boundaries

        if len(self.bin_boundaries) == 0 and len(self.col_list) == 1 \
                and self.min_value is None and self.max_value is None:
            # Only use the amount of bins as input For the histogram function
            return self.nr_bins

        min_value = self._get_min_value()
        max_value = self._get_max_value()

        # expand empty range to avoid empty graph
        return Histogram._calc_n_bins_between(min_value, max_value, self.nr_bins)

    def _add_hist(self, table, column_name):
        """Uses spark to calculate the hist values: for each column a list of weights, and if the bin_list is not set
           a set of bin boundaries"""
        bin_boundaries, bin_weights = table.select(column_name).rdd.flatMap(lambda x: x).histogram(self.bin_boundaries)
        self.hist_dict[self._check_col_name(column_name)] = bin_weights

        if isinstance(self.bin_boundaries, int): # the bin_list is not set
            if len(bin_boundaries) == 2 and bin_boundaries[0] == bin_boundaries[1]:
                # In case of a column with 1 unique value we need to calculate the histogram ourselves.
                min_value = bin_boundaries[0]
                max_value = bin_boundaries[1]
                self.bin_boundaries = self._calc_n_bins_between(min_value, max_value, self.nr_bins)
                self.hist_dict[column_name] = Histogram._calc_weights(self.bin_boundaries, min_value, bin_weights)
            else:
                self.bin_boundaries = bin_boundaries

    @staticmethod
    def _calc_n_bins_between(min_value, max_value, nr_bins):
        """Returns a list of bin borders between min_value and max_value"""
        if min_value == max_value:
            min_value = min_value - 0.5
            max_value = max_value + 0.5
        step = (float(max_value) - float(min_value)) / nr_bins
        return [min_value + (step * float(bn_nr)) for bn_nr in range(nr_bins + 1)]

    @staticmethod
    def _calc_weights(bins, value, value_count):
        """Calculate weights given a bin list, value within that bin list and a count"""
        # first we get a list of bin boundary tuples
        weights = list()
        bin_boundary_idx = [(idx, idx+2) for idx in range(len(bins)-1)]
        bin_boundaries = [tuple(bins[left_idx:right_idx]) for (left_idx, right_idx) in bin_boundary_idx]
        for left_boundary, right_boundary in bin_boundaries:
            if left_boundary <= value < right_boundary:
                weights.append(value_count[0])
            else:
                weights.append(0)
        return weights


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
            self.bin_boundaries = self._calculate_bins()
            for table, column_name in self.col_list:
                self._add_hist(table, column_name)
            self.is_build = True

    def to_pandas(self, kind='hist'):
        """Returns a pandas dataframe from the Histogram object.

        This function calculates the Histogram function in Spark if it was not done yet.

        Args:
            :kind: (:obj:`str`, optional):
                'hist' or 'density'. When using hist this returns the histogram object
                as pandas dataframe. When using density the index contains the bin centers, and the values in the
                DataFrame are the scaled values. Defaults to 'hist'

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
            :ax: (`Axes`)
                An matplotlib Axes object on which the histogram will be plot
            :overlapping (`bool`, optional):
                If set to true, this will generate an overlapping plot.
                When set to False it will generate a normal grouped histogram. Defaults to False.
            :formatted_yaxis: (`bool`, optional).
                If set to true, the numbers on the yaxis will be formatted
                for better readability. E.g. 1500000 will become 1.5M. Defaults to True
            :**kwargs:
                The keyword arguments as used in matplotlib.pyplot.hist
        """
        self.build()

        if formatted_yaxis:
            # Round the y-axis value to nearest thousand, million, or billion for readable y-axis
            formatter = plt.FuncFormatter(Histogram._convert_number_bmk)
            ax.yaxis.set_major_formatter(formatter)

        if overlapping:
            for colname in self.hist_dict:
                ax.hist(self._get_bin_centers(),
                        bins=self.bin_boundaries,
                        alpha=0.5,
                        label=self.hist_dict.keys(),
                        weights=self.hist_dict[colname],
                        **kwargs
                        )
        else:
            weights_multi = [self.hist_dict[colname] for colname in self.hist_dict]
            return ax.hist([self._get_bin_centers()] * len(self.hist_dict),
                           bins=self.bin_boundaries,
                           weights=weights_multi,
                           label=self.hist_dict.keys(),
                           **kwargs)
    
    def plot_density(self, ax, num=300, **kwargs):
        """Returns a density plot on an Pyplot Axes object.

        Args:
            :ax: (`Axes`)
                An matplotlib Axes object on which the histogram will be plot
            :num: (`int`)
                The number of x values the line is plotted on. Default: 300
            :**kwargs:
                Keyword arguments that are passed on to the pyplot.plot function.
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
                                              bins=self.bin_boundaries,
                                              weights=bin_values,
                                              density=True
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
        """Ads 1 or more columns to a histogram.

        Multiple options are available:
            * Add a single column dataframe
            * Add a list of single column dataframes
            * Add a dataframe with multiple columns

        Args:
            :data:
                A single column Spark dataframe, a list of single column Spark
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
