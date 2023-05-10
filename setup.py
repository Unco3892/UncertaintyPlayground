from setuptools import setup, find_packages

setup(
    name='UncertaintyPlayground',
    version='0.1',
    packages=find_packages(),
    author='Ilia Azizi',
    author_email='ilia.azizi@unil.ch',
    description='A Python library for uncertainty estimation in supervised learning tasks',
    long_description=open('README.md').read(),
    install_requires=[
        'numpy',
        'sklearn',
        'torch',
        'gpytorch',
    ],
)
