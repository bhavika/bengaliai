"""
Setup
"""

from setuptools import setup, find_packages

# Parse the version from the module.
version = "0.0.1"

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

dependency_links = []
setup(
    name="bengaliai-kaggle",
    version=version,
    description=u"bengaliai",
    keywords="",
    author=u"Bhavika Tekwani",
    author_email="bhavicka@protonmail.com",
    url="https://github.com/bhavika/bengaliai",
    license="BSD",
    packages=find_packages(),
    exclude_package_data={'': ['data/*']},
    zip_safe=False,
    install_requires=requirements,
    extras_require={"test": ["pytest"]},
)