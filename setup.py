from setuptools import setup, find_packages


setup(
    name='ModaFew',
    version='0.1',
    description='The package for laoding Large Model to do Few-shot/Zero-shot Inference',
    author='YingzhePeng',
    author_email='yingzhepeng@foxmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    packages=find_packages()
)