from setuptools import setup, find_packages

# Load requirements from requirements.txt file
with open('uncertaintyplayground/requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='UncertaintyPlayground',
    version='0.1.0',
    packages=find_packages(),
    author='Ilia Azizi',
    author_email='ilia.azizi@unil.ch',
    description='A Python library for uncertainty estimation in supervised learning tasks',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Unco3892/UncertaintyPlayground', 
    install_requires=requirements,
    tests_require=[
        'unittest'
    ],
    test_suite='uncertaintyplayground.tests',
    classifiers=[
        'Development Status :: 3 - Alpha',
        # 'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
