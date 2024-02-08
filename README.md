Scripts to calculate the preferred nematic director field of liquid crystals elastomer within a scaffold, using the Landau de Gennes free energy functional with strong anchoring boundary conditions, as well as the resulting birefringence patterns and mechanical actuation. Nematic orientation and mechanical response are computed using the Finite Elements method. Therefore, the following packages are required to run these scripts:

- Numpy
- Fenics Legacy (dolfin)
- Proplot
- Scipy

## Calculation of director orientation ##

 To run the scripts, a .xdmf file suitable for fenics is required. The physical surfaces which enforce strong anchoring must labeled with the tag number "101". 
