from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(f:str) -> List[str] :
    requirement = []
    requirement = open(f).readlines()
    requirement = [req.replace("\n","") for req in requirement]
    if HYPEN_E_DOT in requirement:
        requirement.remove(HYPEN_E_DOT)
    return requirement

setup (
    name='regression_project',
    version = '0.0.1',
    author = 'Prakash',
    author_email='Prakash.mewari@yahoo.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)