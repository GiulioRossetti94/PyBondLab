from setuptools import setup, find_packages
import os

setup(name = 'PyBondLab',
      version = '0.2.0',
      description = ('Performs portfolio sorting and strategy evaluation for corporate bonds'),
      long_description=open('README.md').read() if os.path.exists('README.md') else '',
      long_description_content_type='text/markdown',
      author = 'Giulio Rossetti, Alex Dickerson',
      author_email = 'Giulio.Rossetti.1@wbs.ac.uk, alexander.dickerson1@unsw.edu.au',
      license='MIT',
      classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.11',
          'Programming Language :: Python :: 3.12',
          'Programming Language :: Python :: 3.13',
          'Operating System :: OS Independent',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Financial and Insurance Industry',
          'Topic :: Office/Business :: Financial',
          'Topic :: Scientific/Engineering',
      ],
      keywords='corporate bonds, portfolio sorting, asset pricing',
      packages=find_packages(include=['PyBondLab', 'PyBondLab.*']),
      include_package_data=True,
      package_data={
          'PyBondLab': ['data/WRDS/*.csv'],
      },
      url='https://github.com/GiulioRossetti94/PyBondLab',
      project_urls={
        'Open Source Bond Asset Pricing': 'https://openbondassetpricing.com/',
        'Source Code': 'https://github.com/GiulioRossetti94/PyBondLab',
        'Bug Tracker': 'https://github.com/GiulioRossetti94/PyBondLab/issues',
    },
      python_requires='>=3.11',
      install_requires=[
          'numpy<2',
          'pandas>=1.5',
          'statsmodels>=0.14',
          'matplotlib>=3.5',
          'scipy>=1.10',
          'pyarrow>=10.0',
      ],
      extras_require={
          'wrds': ['wrds'],  # For WRDS data download
          'performance': ['numba>=0.57'],  # For performance optimization
          'all': ['wrds', 'numba>=0.57'],
      },
)
