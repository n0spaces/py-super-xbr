import numpy as np
import setuptools

extensions = [
    setuptools.Extension('superxbr.superxbr', ['superxbr/superxbr.c'])
]

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='py-super-xbr',
    version='0.0.1',
    long_description=long_description,
    long_description_content_type='text/markdown',
    ext_modules=extensions,
    install_requires=['numpy', 'Pillow>=7.1.2'],
    include_dirs=[np.get_include()],
    packages=['superxbr'],
    entry_points={'console_scripts': ['superxbr = superxbr.cli:main']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
