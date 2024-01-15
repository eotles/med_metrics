from setuptools import setup, find_packages

setup(
    name='med_metrics',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn'
    ],
    author='Your Name',
    author_email='hi@eotles.com',
    description='Custom ML metrics for medical applications'
)
