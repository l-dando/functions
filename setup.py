from setuptools import setup

setup(
    name='functions',
    url='https://github.com/l-dando/functions',
    author='Luke Dando',
    author_email='luke.dando@hotmail.com',
    packages=['functions'],
    install_requires=['numpy', 'pandas', 'scipy'],
    version='0.1',
    license='MIT',
    description='Functions for Data Science',
    long_description=open('README.md').read(),
)
