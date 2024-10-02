# PhaseFunctionAnalysis
 A collection of python tools to create and analyze phase function LUTs for the scattering of visible light.
 This project is currently not complete. Its goals are to answer the following questions:
 
 How does the phase function change the rate at which the light becomes isotropic (diffuses) within a volume?
 
 How does this rate of diffusion or pattern of diffusion change when a low or high density section of the volume is present?
 
 How does this diffusion change based on the geometry of a nearby surface?
 
 Is there a way to approximate multiple scattering based off of how bounces become diffuse?
 
 Does an approximation exist for a multiple-scattering volume that also acts as a heterogenous medium (including emission, absorption, etc)?
 
 How can phase functions be modified to become more efficient to use with a Monte-Carlo path tracer?
 
 Is the index of refraction of an atmospheric medium important for the visual quality of a planetary atmosphere (using a rayleigh phase function accounting for this medium IOR)?
 

I will be attempting to answer these questions through a number of tools:
- generate_lut.py (alongside tools\data\) will generate look up tables of phase functions from Mie Theory using miepython.
- lut_visualzie.py will allow me to easily determine that generate_lut is successful within a reasonable amount of accuracy.
- phase_modifier.py will investigate various methods of decreasing noise, such as chopping the diffraction spike off of high-anistropic phase functions like those of water droplets in clouds or rain.
- bounce.py will analyze behavior as light diffuses, using a monte-carlo method and a render buffer that accumulates every individual bounce within a volume. Later I will include more tests related to densitiy changes and surface effects.
- render.py will allow the investigation of behaviors in 3 dimensions through an interactive monte-carlo method. This will mostly be used for analyzing approximations.

This program will also serve as a testbed for the application and evaluation of various multiple scattering approximations and rendering methods already published in numerous papers.
 
