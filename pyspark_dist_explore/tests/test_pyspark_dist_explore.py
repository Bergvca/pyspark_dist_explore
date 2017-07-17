import findspark
findspark.init('/usr/lib/spark')

import pyspark.sql.functions as F
import sparktestingbase.sqltestcase
import pandas as pd

from pyspark.sql import Row

from unittest import mock
import sys
sys.path.append('../' )
from pyspark_dist_explore import Histogram


class HistogramTest(sparktestingbase.sqltestcase.SQLTestCase):
    def test_init_default(self):
        """Should set default settings when no arguments are given"""
        hist = Histogram()
        self.assertIsNone(hist.min_value)
        self.assertIsNone(hist.max_value)
        self.assertEqual(10, hist.nr_bins)
        self.assertEqual(0, len(hist.bin_list))
        self.assertEqual(0, len(hist.hist_dict))
        self.assertEqual(0, len(hist.col_list))
        self.assertFalse(hist.is_build)

    def test_init_non_default(self):
        """"Should set min bin, max bin, and number of bins"""
        hist = Histogram(bins=10, range=(5, 8))
        self.assertEqual(10, hist.nr_bins)
        self.assertEqual(5, hist.min_value)
        self.assertEqual(8, hist.max_value)
        self.assertEqual(0, len(hist.bin_list))

    def test_init_bins_given(self):
        """"Should set the list of bins when given in the constructor,
        bins are converted to float"""
        hist = Histogram(bins=[1, 2, '3'])
        self.assertListEqual([1, 2, 3], hist.bin_list)

    def create_test_df(self):
        test_list = [(1, 2), (2, 3), (3, 4)]
        rdd = self.sc.parallelize(test_list)
        rdd_f = rdd.map(lambda x: Row(value=x[0], value2=x[1]))
        return self.sqlCtx.createDataFrame(rdd_f)

    def test_add_column(self):
        """"Should add a column name, column tuple to the col_list when a single column data frame is given"""
        hist = Histogram(bins=10)
        test_df = self.create_test_df()
        hist.add_column(test_df.select(F.col('value')))
        self.assertEqual(1, len(hist.col_list))
        self.assertEqual('value', hist.col_list[0][1])
        self.assertDataFrameEqual(test_df.select(F.col('value')), hist.col_list[0][0])

    def test_add_column_more_then_1_column_in_dataframe(self):
        """"Should throw an error when the input data frame contains more then one column"""
        hist = Histogram(bins=10)
        test_df = self.create_test_df()
        with self.assertRaises(ValueError):
            hist.add_column(test_df)

    def test_add_column_non_numeric(self):
        """Should raise an ValueError if a non-numeric column is added"""
        test_list = ['a', 'b']
        rdd = self.sc.parallelize(test_list)
        rdd_f = rdd.map(lambda x: Row(value=x))
        spark_df = self.sqlCtx.createDataFrame(rdd_f)
        hist = Histogram()
        with self.assertRaises(ValueError):
            hist.add_column(spark_df)

    def test_add_multiple_columns(self):
        """Adds new items to the col_list when new items are added"""
        hist = Histogram(bins=10)
        test_df = self.create_test_df()
        hist.add_column(test_df.select(F.col('value')))
        hist.add_column(test_df.select(F.col('value2')))
        self.assertEqual(2, len(hist.col_list))
        self.assertEqual('value', hist.col_list[0][1])
        self.assertDataFrameEqual(test_df.select(F.col('value')), hist.col_list[0][0])
        self.assertEqual('value2', hist.col_list[1][1])
        self.assertDataFrameEqual(test_df.select(F.col('value2')), hist.col_list[1][0])

    def test_get_min_value(self):
        """Should return the minimum value over all columns in a Histogram"""
        hist = Histogram(bins=10)
        test_df = self.create_test_df()
        hist.add_column(test_df.select(F.col('value')))
        hist.add_column(test_df.select(F.col('value2')))
        self.assertEqual(1, hist._get_min_value())

    def test_get_max_value(self):
        """Should return the maximum value over all columns in a Histogram"""
        hist = Histogram(bins=10)
        test_df = self.create_test_df()
        hist.add_column(test_df.select(F.col('value')))
        hist.add_column(test_df.select(F.col('value2')))
        self.assertEqual(4, hist._get_max_value())

    def test_calculate_bins(self):
        """Should return a list of evenly spaced bins between min and max bin if they are set"""
        hist = Histogram(range=(5, 10), bins=2)
        self.assertListEqual([5, 7.5, 10], hist._calculate_bins())

    def test_calculate_bins_bins_set(self):
        """Should just return the list of bins edges when this was set in the constructor"""
        hist = Histogram(bins=[1, 2, 3])
        self.assertListEqual([1, 2, 3], hist._calculate_bins())

    def test_calculate_bins_single_column(self):
        """Should return the number of bins when there is only a single column, and no min and max is set"""
        hist = Histogram(bins=5)
        test_df = self.create_test_df()
        hist.add_column(test_df.select(F.col('value')))
        self.assertEqual(5, hist._calculate_bins())

    def test_calculate_bins_multiple_columns(self):
        """Should return a list of evenly spaced bins between the smallest and highest value over all columns"""
        hist = Histogram(bins=3)
        test_df = self.create_test_df()   # The lowest value in this DF is 1, the highest is 4
        hist.add_column(test_df.select(F.col('value')))
        hist.add_column(test_df.select(F.col('value2')))
        self.assertListEqual([1, 2, 3, 4], hist._calculate_bins())

    def test_add_hist_single_column(self):
        """Should add a list of bin values (e.g. the number of values that fall in a bin) to the hist_dict, where
        the key is the column name. If multiple columns have the same name a number is appended"""
        hist = Histogram(bins=2)
        test_df = self.create_test_df()
        column_to_ad = test_df.select(F.col('value'))
        hist.add_column(column_to_ad)
        hist.bin_list = hist._calculate_bins()
        hist._add_hist(column_to_ad, 'value')
        self.assertEqual(1, len(hist.hist_dict))
        self.assertListEqual([1, 2], hist.hist_dict['value'])

    def test_add_hist_single_column_sets_bin_list(self):
        """Should set the bin list if this is a single number"""
        hist = Histogram(bins=2)
        test_df = self.create_test_df()
        column_to_ad = test_df.select(F.col('value'))
        hist.add_column(column_to_ad)
        hist.bin_list = hist._calculate_bins()
        hist._add_hist(column_to_ad, 'value')
        self.assertEqual(3, len(hist.bin_list))

    def test_add_hist_multiple_column(self):
        """Should add a second list of bin values to the hist_dict"""
        hist = Histogram(bins=2)
        test_df = self.create_test_df()
        column_to_ad = test_df.select(F.col('value'))
        column_to_ad_2 = test_df.select(F.col('value2'))
        hist.add_column(column_to_ad)
        hist.add_column(column_to_ad_2)
        hist.bin_list = hist._calculate_bins()
        hist._add_hist(column_to_ad, 'value')
        hist._add_hist(column_to_ad_2, 'value2')
        self.assertEqual(2, len(hist.hist_dict))
        self.assertListEqual([1, 2], hist.hist_dict['value2'])

    def test_add_hist_multiple_column_rename_column(self):
        """Should rename the column name if the same column name is added"""
        hist = Histogram(bins=2)
        test_df = self.create_test_df()
        column_to_ad = test_df.select(F.col('value'))
        column_to_ad_2 = test_df.select(F.col('value'))
        hist.add_column(column_to_ad)
        hist.add_column(column_to_ad_2)
        hist.bin_list = hist._calculate_bins()
        hist._add_hist(column_to_ad, 'value')
        hist._add_hist(column_to_ad_2, 'value')
        self.assertEqual(2, len(hist.hist_dict))
        self.assertTrue('value (1)' in hist.hist_dict)

    def test_build(self):
        """Should calculate the bin list, and hist values for each column in the Histogram, if the
        histogram hasn't been build before"""
        hist = Histogram(bins=2)
        test_df = self.create_test_df()
        column_to_ad = test_df.select(F.col('value'))
        column_to_ad_2 = test_df.select(F.col('value2'))
        hist.add_column(column_to_ad)
        hist.add_column(column_to_ad_2)
        hist.build()
        self.assertEqual(3, len(hist.bin_list))
        self.assertEqual(2, len(hist.hist_dict))
        self.assertTrue(hist.is_build)

    @mock.patch('pyspark_dist_explore.Histogram._add_hist')
    @mock.patch('pyspark_dist_explore.Histogram._calculate_bins')
    def test_build_already_build(self, calculate_bins_func, add_hist_func):
        """Should not rebuild if Histogram was already build before"""
        hist = Histogram()
        hist.is_build = True
        hist.build()
        self.assertFalse(add_hist_func.called)
        self.assertFalse(calculate_bins_func.called)

    def test_to_pandas_default(self):
        """Should create a pandas dataframe from the Histogram object"""
        hist = Histogram(bins=2)
        test_df = self.create_test_df()
        column_to_ad = test_df.select(F.col('value'))
        column_to_ad_2 = test_df.select(F.col('value2'))
        hist.add_column(column_to_ad)
        hist.add_column(column_to_ad_2)
        expected_df = pd.DataFrame({'value': [2, 1],
                                    'value2': [1, 2]}).set_index([['1.00 - 2.50', '2.50 - 4.00']])
        self.assertTrue(expected_df.equals(hist.to_pandas()))

    def test_to_pandas_density(self):
        """Should create a pandas dataframe of a denisty plot of the histogram"""
        hist = Histogram(bins=2)
        test_df = self.create_test_df()
        column_to_ad = test_df.select(F.col('value'))
        column_to_ad_2 = test_df.select(F.col('value2'))
        hist.add_column(column_to_ad)
        hist.add_column(column_to_ad_2)
        expected_df = pd.DataFrame({'value': [1.0, 0.5], 'value2': [0.5, 1.0]}).set_index([[1.75, 3.25]])
        self.assertTrue(expected_df.equals(hist.to_pandas('density')))

    def test_add_data_single_column(self):
        """Should add a single column of data to the Histogram"""
        hist = Histogram()
        test_df = self.create_test_df()
        column_to_ad = test_df.select(F.col('value'))
        hist.add_data(column_to_ad)
        self.assertEqual(1, len(hist.col_list))

    def test_add_data_list_of_columns(self):
        """Should add all columns from the list of columns to the Histogram"""
        test_df = self.create_test_df()
        column_to_ad = test_df.select(F.col('value'))
        column_to_ad_2 = test_df.select(F.col('value2'))
        hist = Histogram()
        hist.add_data([column_to_ad, column_to_ad_2])
        self.assertEqual(2, len(hist.col_list))

    def test_add_data_entire_dataframe(self):
        """Should add all columns of a dataframe to the histogram"""
        test_df = self.create_test_df()
        hist = Histogram()
        hist.add_data(test_df)
        self.assertEqual(2, len(hist.col_list))
