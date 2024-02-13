# Simulating Liquid Crystal Elastomer orientation and actuation #
These scripts calculate the preferred nematic director field of liquid crystals within a scaffold, using the Landau de Gennes free energy functional with strong anchoring boundary conditions, as well as the resulting birefringence patterns and mechanical actuation of the elastomer. Nematic orientation and mechanical response are computed using the Finite Elements method. Therefore, the following packages are required to run these scripts:

- Numpy
- Fenics Legacy (dolfin) and ufl
- Proplot
- Scipy
- Meshio

## Calculation of director orientation ##

To compute the preferred director orientation, the script "Landau_de_Gennes_Qvectorized.py" must be run. This script requires a Fenics suitable xdmf file, whose physical surfaces on which strong anchoring is enforced must be labelled with the same tag. Such tag is also required as input for the program. The script then computes the Q tensor field minimizer to the Landau de Gennes and saves it as an xdmf file.  The user may then extract the Q tensor field at the mesh points and diagonalize it with standard packages (such as numpy.linalg). The director will be the vector with the largest eigenvalue.

## Simulation of optical experiments ##

The script "optical_intensity_jones.py" simulates the intensity field of an optical microscopy experiment with orthogonal polarizer and analyzer. 
