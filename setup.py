from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py
import os


class build_py(_build_py):
    """Keep internal development helpers out of the wheel."""

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [
            module for module in modules
            if not (module[0] == 'PyBondLab' and module[1] == 'pbl_test')
        ]

    def build_module(self, module, module_file, package):
        if package == 'PyBondLab' and module == 'pbl_test':
            return None
        return super().build_module(module, module_file, package)


setup(name = 'PyBondLab',
      version = '0.2.0',
      description = ('Performs portfolio sorting and strategy evaluation for corporate bonds'),
      long_description=open('README.md').read() if os.path.exists('README.md') else '',
      long_description_content_type='text/markdown',
      author = 'Giulio Rossetti, Alex Dickerson',
      author_email = 'Giulio.Rossetti.1@wbs.ac.uk, alexander.dickerson1@unsw.edu.au',
      license='MIT',
      classifiers=[
          'Development Status :: 5 - Production/Stable',
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
          'numba>=0.57',
      ],
      extras_require={
          'wrds': ['wrds'],  # For WRDS data download
          'performance': [],  # Backward-compatible alias; numba is required
          'all': ['wrds'],
      },
      cmdclass={'build_py': build_py},
)
