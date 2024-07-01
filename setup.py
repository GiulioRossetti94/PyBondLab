from setuptools import setup

setup(name = 'PyBondLab',
      version = '0.0.1',
      description = ('Performs portfolio sorting and strategy evaluation for corporate bonds'),
      long_description=open('README.md').read(),
      author = 'Giulio Rossetti, Alex Dickerson',
      author_email = 'Giulio.Rossetti.1@wbs.ac.uk, alexander.dickerson1@unsw.edu.au',
      license='MIT',
      keywords='corporate bonds, portfolio sorting, data cleaning',
      packages=['PyBondLab'],
    #   packages=find_packages(),
      url='https://github.com/GiulioRossetti94/PyBondLab',
      install_requires=[
          'numpy','pandas'],
)