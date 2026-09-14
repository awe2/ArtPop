import os, sys
import builtins
from setuptools import setup, find_packages
sys.path.append('src')


def readme():
    with open("README.rst") as f:
        return f.read()


# HACK: fetch version
builtins.__ARTPOP_SETUP__ = True
import artpop
version = artpop.__version__


# Publish the library to PyPI.
if "publish" in sys.argv[-1]:
    os.system("python setup.py sdist bdist_wheel")
    os.system(f"python3 -m twine upload dist/*{version}*")
    sys.exit()


# Push a new tag to GitHub.
if "tag" in sys.argv:
    os.system("git tag -a v{0} -m 'version {0}'".format(version))
    os.system("git push --tags")
    sys.exit()


setup(
    name='artpop',
    version=version,
    description='Building artificial galaxies one star at a time',
    long_description=readme(),
    author='Johnny Greco & Shany Danieli',
    author_email='artpopcode@gmail.com',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    # filter_curves/ lives at the repo root, where tools/build_filter_data.py
    # walks it, and is NOT installed from there. A K-correction needs the
    # curves themselves at runtime, so artpop.filters.filter_curve_dir()
    # looks under data/filter_curves/ as well as at the repo root and
    # raises naming both rather than returning silently. Syncing the curves
    # into src/artpop/data/filter_curves/ before a build is what makes an
    # installed copy work; this line is what ships them once they are there.
    package_data={"artpop": ["data/*.pkl", "data/*.txt",
                             "data/filter_curves/*/*.csv"]},
    include_package_data=True,
    url='https://github.com/ArtificialStellarPopulations/ArtPop',
    install_requires=[
        'numpy>=1.17',
        'scipy>=1',
        'astropy>=4',
        'matplotlib>=3',
        'fast-histogram>=0.9',
        'requests',
     ],
     classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Astronomy",
      ],
    python_requires='>=3.6',
)
