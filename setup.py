from setuptools import setup

setup(
    name='nearest_neighbour',
    url='https://github.com/l-dando/functions',
    author='Luke Dando',
    author_email='luke.dando@hotmail.com',
    packages=['nearest_neighbour'],
    install_requires=['numpy', 'pandas', 'scipy'],
    version='0.1',
    license='MIT',
    description='Functions using nearest neighbours',
    long_description=open('README.md').read(),
)
