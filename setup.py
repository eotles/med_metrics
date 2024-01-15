from setuptools import setup, find_packages

# Read the contents of your README file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='med_metrics',
    version='0.0.5',
    packages=find_packages(),
    author='Erkin Ötleş',
    author_email='hi@eotles.com',
    url='https://github.com/eotles/med_metrics',
    description='Custom ML metrics for medical applications',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy',
        'scikit-learn',
        'matplotlib'
    ],
)
