import codecs
import glob
import os
import re
import shutil

from setuptools import setup, Command


class CompleteClean(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        shutil.rmtree('./build', ignore_errors=True)
        shutil.rmtree('./dist', ignore_errors=True)
        shutil.rmtree('./' + project + '.egg-info', ignore_errors=True)
        temporal = glob.glob('./' + project + '/*.pyc')
        for t in temporal:
            os.remove(t)


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

project = "continuous"
here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    """
    Args:
        *parts:
    """
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    """
    Args:
        *file_paths:
    """
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name=project,
    python_requires='>=3.6',
    version=find_version(project, '_version.py'),
    description=("Navigation project from Udacity"),
    long_description=read('README.md'),
    url=None,
    packages=[project, 'tests'],
    install_requires=requirements,
    cmdclass={'clean': CompleteClean},
    test_suite='nose.collector'
)
