import os
import json
from urllib.request import urlopen
from hashlib import sha1
from importlib import resources
import atexit
from contextlib import ExitStack

def download_file(source):
	"""Download resource files, and verify the hash.

	Raises an exception if the download is not successful."""
	try:
		with urlopen(source['url']) as response:
			data = response.read()
			with open(source['filename'], 'wb') as f:
				f.write(data)
	except Exception as e:
		raise RuntimeError("Unable to download file. "
			"Please manually try to download {} to {}.".format(
			source['url'], source['filename']))

	if verify_source(source) != 0:
		raise RuntimeError("Received an incorrect download from the server. "
			"Please manually try to download {} to {}.".format(
			source['url'], source['filename']))


def verify_source(source):
	"""Verify that a file exists on disk, and that it has the correct hash.

	Returns 0 if file is OK, 1 if file does not exist, 2 if file has incorrect
	checksum.
	"""
	if not os.path.isfile(source['filename']):
		return 1

	with open(source['filename'], 'rb') as f:
		file_sha1 = sha1(f.read()).hexdigest()
	if file_sha1 != source['sha1']:
		return 2

	return 0


def obtain_endf_files():
	"""Get the endf files. Tries to download them if they don't exist or are
	corrupted.

	Returns a dict, with keys atomic_relax, electrons, and photoat for the
	respective endf files. Values are filenames, as strings.

	If the file is corrupted and cannot be downloaded, throws an exception.
	"""

	# Updated pkg_resources.resource_string() implementation using importlib
	reference = resources.files(__name__).joinpath('../data/endf_sources.json')
	contents = reference.read_bytes()
	sources = json.loads(contents.decode("utf-8"))

	# Updated pkg_resources.resource_filename() implementation using importlib, contextlib and atexit
	file_manager = ExitStack()
	atexit.register(file_manager.close)
	ref = resources.files(__name__) / '../data/endf_data'
	endf_dir = file_manager.enter_context(resources.as_file(ref))

	os.makedirs(endf_dir, exist_ok=True)
	for name, source in sources.items():
		source['filename'] = '{}/{}.zip'.format(endf_dir, name)

		source_status = verify_source(source)
		if source_status == 0:
			continue
		elif source_status == 1:
			print("Downloading {} file.".format(name))
		else:
			print("Checksum for {} file failed. Removing and re-downloading.".format(name))

		download_file(source)

	return { name: source['filename'] for name, source in sources.items() }
