from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='pdhcg', 
    version='0.0.1', 
    keywords='Optimizer',
    license='MIT', 
    author='hongpeili',
    author_email='ishongpeili@gmail.com',
    packages=find_packages(), 
    description = 'A python wrapper for PDHCG.jl',
    long_description=long_description,
    long_description_content_type='text/markdown',
    include_package_data = True,
    entry_points={
        'console_scripts':[
            'pdhcg=run:main' 
        ],
        },
    install_requires=[
        'juliacall', 'numpy', 'scipy'
    ],
    
)