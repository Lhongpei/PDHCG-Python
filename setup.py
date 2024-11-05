from setuptools import setup, find_packages

setup(
    name='pdhcg', 
    version='0.0.0', 
    keywords='Optimizer',
    license='MIT', 
    author='hongpeili',
    author_email='ishongpeili@gmail.com',
    packages=find_packages(), 
    description = 'A python wrapper for PDHCG.jl',
    #long_description_content_type=open('README.md').read(),
    include_package_data = True,
    entry_points={
        'console_scripts':[
            'pdhcg=run:main' 
        ],
        },
    install_requires=[
        'juliacall>=0.9.23', 'numpy', 'scipy'
    ],
    
)