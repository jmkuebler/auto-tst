from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = '1.0'

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs]

setup(
    name='autotst',
    version=__version__,
    description='AutoML Two-Sample Test',
    long_description=long_description,
    url='https://github.com/jmkuebler/auto-tst',
    download_url='https://github.com/jmkuebler/auto-tst/archive/refs/heads/master.zip',
    license='MIT',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Programming Language :: Python :: 3',
    ],
    keywords='',
    packages=find_packages(exclude=['docs', 'tests*']),
    include_package_data=True,
    author=['Jonas M. Kuebler', 'Vincent Stimper'],
    install_requires=install_requires,
    author_email=''
)
