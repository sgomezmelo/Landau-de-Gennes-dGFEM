# Alignment and Actuation of Liquid Crystals via 3D Confinement and Two-Photon Laser Printing
This repository contains the relevant scripts to the publication "Alignment and Actuation of Liquid Crystals via 3DConfinement and Two-Photon Laser Printing" (DOI: 10.26434/chemrxiv-2024-dp30s-v2). Please consider citing if you found this useful. 

These scripts calculate the preferred nematic director field of liquid crystals within a scaffold, using the Landau de Gennes free energy functional with strong anchoring boundary conditions, as well as the resulting birefringence patterns and mechanical actuation of the elastomer. Nematic orientation and mechanical response are computed using the Finite Elements method. Therefore, the following packages are required to run these scripts:

- Numpy
- Fenics Legacy (dolfin) and ufl
- Proplot
- Scipy
- Meshio

In all three scripts we assume that there is a directory "Geometries" with each geometry in xdmf or msh file contained inside a separate subdirectory. The name of the subdirectory is required as input in all cases. 
## Calculation of director orientation ##

To compute the preferred director orientation, the script "Landau_DeGennes_minimizer_Qvectorized.py" must be run. This script requires a Fenics suitable xdmf mesh file, whose physical surfaces on which strong anchoring is enforced must be labelled with a single tag. Such tag is also required as input for the program. The script then computes the Q tensor field minimizer to the Landau de Gennes and saves it as an xdmf file.  The user may then extract the Q tensor field at the mesh points and diagonalize it with standard packages (such as numpy.linalg). The director will be the vector with the largest eigenvalue.

## Simulation of optical experiments ##

The script "optical_intensity_jones.py" simulates the intensity field of an optical microscopy experiment with orthogonal polarizer and analyzer. It takes as input an xdmf file with the Q tensor solution (such as that produced by Landau_de_Gennes_Qvectorized.py), whose z-axis is aligned with the propagation director of the probing light. The calculation assumes an equal mixture of red (650nm), green (550nm) and blue (450nm). The program produces an image of the intensity field at the end of the sample. Additionally, intensity slices along the horizontal and diagonal are plotted, as well as a projection of the director field at the top of the sample onto the xy plane.


## Mechanical actuation ##

The mechanical actuation of the elastomer for an analytically specified director field is computed by the "LC_hyperelastic_minimizer.py" script. It takes as input the mesh of the undeformed configuration as an .msh file that is then internally converted into xdmf by meshio. The program returns two xdmf files; one with the director field projected into the mesh and the resulting displacement field, which may be read with suitable software (e.g. Paraview). The specified nematic field is currently cylindrically radial, but it can be changed within the code. 
