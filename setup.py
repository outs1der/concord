import setuptools
import glob, re

with open("README.md", "r") as fh:
    long_description = fh.read()


def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
                       open(project + '/__init__.py').read())
    return result.group(1)


def get_version():
    """Get the version number"""

    import concord
    return concord.__version__


package_name = 'concord'

setuptools.setup(
    name="concord",
    author=get_property('__author__', package_name),
    author_email=get_property('__email__', package_name),
    version=get_property('__version__', package_name),
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
