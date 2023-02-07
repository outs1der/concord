import setuptools
import glob

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="concord",
    version="1.1.0",
    author="Duncan Galloway",
    author_email="Duncan.Galloway@monash.edu",
    description="Package for multi-epoch comparisons of model thermonuclear bursts to observations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/outs1der/concord",
    packages=setuptools.find_packages(),
#    packages=['concord'],
#    package_dir={'concord': "concord"},
# The following line was intended to force importing of the diskmodel
# module, but it doesn't seem to work
    py_modules = ["diskmodel"],
#
    package_data={'': ['data/*']},
    scripts=glob.glob('scripts/*'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
