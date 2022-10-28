from setuptools import setup, find_packages

setup(name='weighted-ensembles',
      version='0.1.0',
      description='Library for combining probabilistic classifiers',
      author='Rene Fabricius',
      author_email='quadrupedans@gmail.com',
      packages=find_packages(include=['weensembles', 'weensembles.*']),
      install_requires=[
            "torch>=1.12.0",
            "torchvision>=0.13.0",
            "pandas>=1.1.2",
            "psutil",
            "scikit-learn>=0.23.2",
            "numpy>=1.21.6"
      ],
      python_requires=">=3.7"
     )