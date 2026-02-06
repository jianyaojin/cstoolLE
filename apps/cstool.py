import cstool as cst
import numpy as np
import argparse
from cstool.input_data import param_file
from cstool.mott import mott_dimfp
from cstool.phonon import ac_phonon_dimfp, ac_phonon_loss
from cstool.common import units
from cstool.common.interpolate import interpolate_f
from cstool.common import datafile
from cstool.common.icdf_compile import compute_tcs, compute_tcs_icdf
from cstool.dielectric_function import (
	dimfp_kieft, compile_ashley_imfp_icdf,
	elf_full_penn, compile_full_imfp_icdf)
from cstool.endf import compile_electronionization_icdf, compile_photoionization_icdf

"""
Cross Section tool (cstool) calculates scattering cross sections for different materials from a 
.yaml file. It produces a .mat file that can be inserted into the nebula simulator for electron
matter simulations.

All physics models calculate their key scattering parameters here, to be written to .mat files.
"""

def compile_kieft_elastic(outfile, material_params, K, P):
	"""
	Calculates, tabulates and writes elastic cross sections, inverse cumulative distribution
	function and transport lengths into a .mat file. Function interpolates between two models: 
	Electron acoustic-phonon interaction model and Mott cross sections calculated via ELSEPA. 

	Parameters
	----------
	outfile : datafile class object 
		cstool unique class with h5py functionality + the addition of pint (a units package) units
		for quantities
	material_params : param_file class object
		file that stores data from .yaml material file
	K : array (should be geomspace)
		electron energies to calculate tables for. Nebula tends to complain if the energies
		aren't spaced logarithmically.
	P : array
		probabilities to evaluate the icdf at

	Returns
	-------
	None (imfp, icdf and transport length written directly into outfile) 
	"""
	print("# Computing Mott cross-sections using ELSEPA.")
	mott_fn = mott_dimfp(material_params, K[K>10*units.eV], threads=4)

	def elastic_cs_fn(E, costheta):
		return interpolate_f(
			lambda E: ac_phonon_dimfp(material_params)(E, costheta),
			lambda E: mott_fn(E, costheta),
			100*units.eV, 200*units.eV
		)(E)

	print("# Computing elastic total cross-sections and iCDFs.")
	imfp = np.zeros(K.shape) * units('nm^-1')
	icdf = np.zeros((K.shape[0], P.shape[0])) * units.dimensionless
	tl = np.zeros(K.shape) * units.nm
	for i, E in enumerate(K):
		_imfp, _icdf = compute_tcs_icdf(
			lambda costheta : elastic_cs_fn(E, costheta),
			P,
			np.linspace(-1, 1, 100000))
		imfp[i] = 2*np.pi * _imfp
		icdf[i,:] = _icdf

		# 2pi from integral over solid angle (dOmega = sintheta dtheta dphi)
		# integral from -1 to 1 because change of variables which also absorbs the sin term
		_itl = 2 * np.pi * compute_tcs(
			lambda costheta : elastic_cs_fn(E, costheta) * (1 - costheta),
			np.linspace(-1, 1, 100000))
		tl[i] = 1 / _itl

		print('.', end='', flush=True)
	print()

	group = outfile.create_group("/kieft/elastic")
	group.add_scale("energy", K, 'eV')
	group.add_dataset("imfp", imfp, ("energy",), 'nm^-1')
	group.add_dataset("costheta_icdf", icdf, ("energy", None), '')
	group.add_dataset("tl", tl, ("energy",), 'nm')


def compile_kieft_inelastic(outfile, material_params, K, P):
	"""
	Calculates, tabulates and writes elastic cross sections and inverse cumulative distribution
	function into a .mat file. This approach is based on that of Ashley (1988). This is not the
	default inelastic method for Nebula.

	Parameters
	----------
	outfile : datafile class object 
		cstool custom data storage class
	material_params : param_file class object
		file that stores data from .yaml material file
	K : array (should be geomspace)
		eelectron energies to calculate tables for
	P : array
		probabilities to evaluate the icdf at

	Returns
	-------
	None (imfp and icdf written directly into outfile) 
	"""
	print("# Computing inelastic-Kieft total cross-sections and iCDFs.")

	imfp, icdf = compile_ashley_imfp_icdf(
		lambda E,w : dimfp_kieft(E, w, material_params),
		K, P)

	group = outfile.create_group("/kieft/inelastic")
	group.add_scale("energy", K, 'eV')
	group.add_dataset("imfp", imfp, ("energy",), 'nm^-1')
	group.add_dataset("w0_icdf", icdf, ("energy", None), 'eV')

def compile_kieft_ionization(outfile, material_params, E, P):
	"""
	Compiles photoionization icdf dataset for the kieft inelastic model. Used for sampling the 
	shell involved in inelastic scattering. Based on Ashley's model.

	Parameters
	----------
	outfile : datafile class object 
		cstool custom data storage class
	material_params : param_file class object
		file that stores data from .yaml material file
	E : array (should be geomspace)
		electron energies to calculate tables for
	P : array
		probabilities to evaluate the icdf at

	Returns
	-------
	None (imfp and icdf written directly into outfile) 
	"""
	print("# Computing Kieft ionization energy probabilities")
	icdf = compile_photoionization_icdf(material_params, E, P)

	group = outfile.create_group("/kieft/ionization")
	group.add_scale("energy", E, 'eV')
	group.add_dataset("binding_icdf", icdf, ("energy", None), 'eV')

	group.add_dataset("outer_shells", material_params.optical.get_outer_shells(), None, 'eV')

def compile_full_penn(outfile, material_params,
	K, P_omega, n_omega_q, P_q):
	"""
	Compiles and writes imfp, stopping power, energy loss and momentum loss tables to file.

	Parameters
	----------
	outfile : datafile class object 
		cstool custom data storage class
	material_params : param_file class object
		file that stores data from .yaml material file
	K : array like
		electron energies to calculate tables for
	P_omega : array like
		probability grid to evaluate energy loss probabilities at (e.g. this amount of energy
		loss has this probability to occur)
	n_omega_q : float
		resolution of the omega-q lookup table. Omega is related to q, so each q_icdf table is 
		associated with a value for omega. The earlier sampled value for omega is used to choose
		which q_icdf to sample from to obtain q. This parameter controls the number of said
		q_icdf tables.
	P_q : array like
		probability grid to evaluate momentum loss probabilities at.

	Returns
	-------
	None (all quantities written to outfile)
	"""

	# Compute ELF
	omega, q, elf = elf_full_penn(material_params, K[-1], 1200, 1000)

	print("# Computing inelastic total cross-sections and iCDFs.")
	imfp, sp, omega_icdf, q_2dicdf = compile_full_imfp_icdf(
		omega, q, elf,
		K, P_omega, n_omega_q, P_q,
		material_params.band_structure.get_min_excitation())

	group = outfile.create_group("/full_penn")
	group.add_scale("energy", K, 'eV')
	group.add_dataset("imfp", imfp, ("energy",), 'nm^-1')
	group.add_dataset("sp", sp, ("energy",), 'eV/nm')
	group.add_dataset("omega_icdf", omega_icdf, ("energy", None), 'eV')
	group.add_dataset("q_icdf", q_2dicdf, ("energy", None, None), 'nm^-1')

def compile_ionization(outfile, material_params,
	K, E_frac, P):
	"""
	Calculates and writes icdf tables for electron shells. These later determine if the excitation
	is inner-shell or valence.

	Parameters
	----------
	outfile : datafile class object 
		cstool custom data storage class
	material_params : param_file class object
		file that stores data from .yaml material file
	K : array like
		Electron energies to calculate tables for.
	E_frac : array like
		Array of the fractions omega/K. Multiply each array value by K to obtain the omega table.
		Given a previously sampled omega this then can be used to choose a binding_icdf table, since
		there is one binding_icdf table per omega.
	P : array like
		Probabilities to evaluate icdfs at.
	Returns
	-------
	None (Written to outfile)
	"""
	print("# Computing ionization energy probabilities")

	icdf = compile_electronionization_icdf(material_params, K, E_frac, P)

	group = outfile.create_group("/ionization")
	group.add_scale("energy", K, 'eV')
	group.add_scale("loss_frac", E_frac, '')
	group.add_dataset("binding_icdf", icdf, ("energy", "loss_frac", None), 'eV')



def main():
	print("This is cstool version {}".format(cst.__version__))

	#
	# Parse arguments
	#
	parser = argparse.ArgumentParser(
		description='Create HDF5 file from material definition.')
	parser.add_argument(
		'material_file', type=str,
		help="Filename of material in YAML format.")
	parser.add_argument(
		'--max_energy', type=float, default=50,
		help="Upper energy limit to use, in keV. Default is 50 keV.")
	args = parser.parse_args()
	max_energy = args.max_energy * units.keV


	s = cst.input_data.param_file(args.material_file)
	print("Compiling material {}".format(s.name))


	#
	# Compute cross sections and write to file
	#
	with datafile("{}.mat".format(s.name), 'w') as outfile:
		outfile.set_property('cstool_version', cst.__version__)
		outfile.set_property('name', s.name)
		outfile.set_property('conductor_type', s.band_structure.model)
		outfile.set_property('fermi', s.band_structure.get_fermi(), 'eV')
		outfile.set_property('barrier', s.band_structure.get_barrier(), 'eV')
		outfile.set_property('phonon_loss', ac_phonon_loss(s), 'eV')
		outfile.set_property('effective_A',
				sum(e.M * e.count for e in s.elements) /
				(units.N_A*sum(e.count for e in s.elements)),
			'g')
		if s.band_structure.model == 'insulator' or s.band_structure.model == 'semiconductor':
			outfile.set_property('band_gap', s.band_structure.band_gap, 'eV')


		# Kieft elastic
		compile_kieft_elastic(outfile, s,
			np.geomspace(1, max_energy.to(units.eV).magnitude, 128) * units.eV,
			np.linspace(0.0, 1.0, 512))

		# Kieft inelastic
		compile_kieft_inelastic(outfile, s,
			np.geomspace(
				s.band_structure.get_fermi().to(units.eV).magnitude + .1,
				max_energy.to(units.eV).magnitude,
				128) * units.eV,
			np.linspace(0.0, 1.0, 512))

		# Kieft ionization
		compile_kieft_ionization(outfile, s,
			np.geomspace(1, max_energy.to(units.eV).magnitude, 1024) * units.eV,
			np.linspace(0.0, 1.0, 1024))

		# Full Penn
		compile_full_penn(outfile, s,
			np.geomspace(
				s.band_structure.get_min_excitation().to(units.eV).magnitude+0.1, \
				max_energy.to(units.eV).magnitude,
				128) * units.eV,
			np.linspace(0.0, 1.0, 512),
			512,
			np.linspace(0.0, 1.0, 512))

		# Ionization
		compile_ionization(outfile, s,
			np.geomspace(1, max_energy.to(units.eV).magnitude, 128) * units.eV,
			np.geomspace(1e-4, 1, 512) * units.dimensionless,
			np.linspace(0.0, 1.0, 512))

if __name__ == '__main__':
	main()
