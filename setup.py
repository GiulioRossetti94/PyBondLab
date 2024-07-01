from setuptools import setup

setup(name = 'PyBondLab',
      version = '0.0.1',
      description = ('Performs portfolio sorting and strategy evaluation for corporate bonds'),
      author = 'Giulio Rossetti, Alex Dickerson',
      author_email = 'Giulio.Rossetti.1@wbs.ac.uk, alexander.dickerson1@unsw.edu.au',
      license='MIT',
      keywords='corporate bonds, portfolio sorting, data cleaning',
      packages=['PyBondLab'],
      url='https://github.com/GiulioRossetti94/PyBondLab',
      install_requires=[
          'numpy','pandas'],
)