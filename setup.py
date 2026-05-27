from setuptools import setup, find_packages

setup(
    name='tabulairity',
    version='1.3.0',
    # Tell Python to look in the 'src' folder for packages
    package_dir={'': 'src'},
    # Find all packages inside that 'src' folder
    packages=find_packages(where='src'),
)
