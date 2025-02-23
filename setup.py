from setuptools import setup, find_packages


def load_requirements(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [
            line.strip() for line in f
            if line.strip() and not line.startswith('#')
        ]


setup(
    name='x_mobility',
    version='0.1.0',
    packages=find_packages(),
    package_dir={"": "."},
    author='Wei Liu',
    author_email='liuw@nvidia.com',
    description='Python package for X-Mobility',
    url='https://github.com/NVlabs/X-MOBILITY',
    install_requires=load_requirements('requirements.txt'),
)
