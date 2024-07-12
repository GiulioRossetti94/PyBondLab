from setuptools import setup, find_packages

setup(name = 'PyBondLab',
      version = '0.0.1',
      description = ('Performs portfolio sorting and strategy evaluation for corporate bonds'),
      long_description=open('README.md').read(),
      author = 'Giulio Rossetti, Alex Dickerson',
      author_email = 'Giulio.Rossetti.1@wbs.ac.uk, alexander.dickerson1@unsw.edu.au',
      license='MIT',
      keywords='corporate bonds, portfolio sorting, data cleaning',
      # packages=['PyBondLab'],
      packages=find_packages(),
      include_package_data=True,
      package_data={
          '': ['data/wrds/*.csv'],
      },
      url='https://github.com/GiulioRossetti94/PyBondLab',
      project_urls={
        'Open Source Bond Asset Pricing project': 'https://openbondassetpricing.com/',
        'Source Code': 'https://github.com/GiulioRossetti94/PyBondLab',
        'Bug Tracker': 'https://github.com/GiulioRossetti94/PyBondLab/issues',
    },
      python_requires='>=3.11',
      install_requires=[
          'numpy','pandas','statsmodels','matplotlib'],
)