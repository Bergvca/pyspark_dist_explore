from setuptools import setup

setup(
    name='pyspark_dist_explore',
    version='0.1.7',
    packages=['pyspark_dist_explore'],
    license='MIT License',
    description='Create histogram and density plots from PySpark Dataframes',
    author='Chris van den Berg',
    author_email='fake_email@gmail.com',
    zip_safe=False,
    install_requires=['pandas'
                      , 'numpy'
                      , 'scipy'
                      , 'matplotlib'
                      # , 'spark_testing_base' # Only required for testing
                      # , 'findspark' # Only required for testing
                      ]

)
