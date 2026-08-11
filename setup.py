from distutils.core import setup
from pathlib import Path
import re

# This bit of code reads the current cstool __version__ from cstool/__init__.py
# Saves a bit of time having to update both version numbers.
init_text = Path("cstool/__init__.py").read_text(encoding="utf-8")
version_number = re.search(
    r'^__version__\s*=\s*["\']([^"\']+)["\']',
    init_text,
    re.MULTILINE,
).group(1)

# Actual setup
setup(
	name = 'cstool',
	version = version_number,
	description = 'Computes cross sections for the Nebula simulator',
	packages = ['cstool',
		'cstool.common', 'cstool.dielectric_function', 'cstool.endf',
		'cstool.input_data', 'cstool.mott', 'cstool.phonon',
		'cstool.apps'
	],
	package_dir = {
		'cstool': 'cstool',
		'cstool.apps': 'apps'
	},
	package_data = {'cstool': ['data/endf_sources.json',
		'data/endf_data/atomic_relax.zip',
		'data/endf_data/electrons.zip',
		'data/endf_data/photoat.zip']},
	install_requires = ['numpy', 'scipy', 'pyyaml', 'h5py', 'numba', 'pint>=0.11', 'setuptools'],
	entry_points = {'console_scripts': ['cstool = cstool.apps.cstool:main']},
)
