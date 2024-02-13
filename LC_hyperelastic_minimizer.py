from fenics import *
import dolfin
import numpy as np
from ufl import nabla_div, VectorElement, FiniteElement, MixedElement, split
import math 
import meshio
import sys
import os

#Create result folder and read undeformed geometry 
save_dir = "./Results/"
geom_folder = "./Geometries/"
geom_subfolder = input("Type name of .msh subfolder path: ")
mesh_name = input("Type name of .msh file: ")
g_name = input("Choose director field name: ")
os.system("mkdir "+save_dir+mesh_name)
msh = meshio.read(geom_folder+geom_subfolder+"/"+mesh_name+'.msh')

line_data = msh.cell_data_dict["gmsh:physical"]["tetra"]
meshio.write(mesh_name+".xdmf",
    meshio.Mesh(points=msh.points,
        cells={"tetra": msh.cells_dict["tetra"]},
        cell_data={"bnd_marker": [line_data]}
    )
)

mesh = dolfin.cpp.mesh.Mesh()
mvc_subdomain = dolfin.MeshValueCollection("size_t", mesh, mesh.topology().dim())
mvc_boundaries = dolfin.MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)

with XDMFFile(MPI.comm_world, mesh_name+".xdmf") as xdmf_infile:
    xdmf_infile.read(mesh)
    xdmf_infile.read(mvc_subdomain, "")

domains = dolfin.cpp.mesh.MeshFunctionSizet(mesh, mvc_subdomain)
dx = Measure('dx', domain=mesh, subdomain_data=domains)
#Principal values of the growth tensor
alpha_1 = np.sqrt(1.0/2.0) 
alpha_2 = np.sqrt(2.0)
theta = Expression('atan2(x[1],x[0])', degree = 1) # Polar angular coordinate
n = Expression((('cos(theta)','sin(theta)',0.0)), theta = theta, degree = 1)
F = (alpha_1-alpha_2)*outer(n,n)+(alpha_2)*Identity(3)
director_field = project(n,VectorFunctionSpace(mesh, 'DG', 1)) 

with XDMFFile(dolfin.MPI.comm_world, save_dir+mesh_name+"/"+g_name+"director_field.xdmf") as xdmf_outfile:
        xdmf_outfile.write(director_field)

# Neo Hookean elastic energy W = tr(AA^T) with growth tensor G and decomposition GA = F 
def Energy_density(u,F):
    inv_F_n = inv(F) #Inverse growth tensor
    F = nabla_grad(u)+Identity(d) 
    A = dot(inv_F_n,F) 
    tensor_p = dot(A,A.T) 
    w = inner(Identity(d),tensor_p)/2.0 
    return w

#Define function space as Vector space (displacement) + scalar function (incompressibility) + 6 constants to eliminate rigid motions
V = VectorElement("CG", mesh.ufl_cell(), 1) 
Lagrange = FiniteElement("Real", mesh.ufl_cell(), 0)
L1 = FiniteElement("CG", mesh.ufl_cell(), 1) 
R6 = MixedElement([Lagrange for i in range(6)]) 
mixed_element = MixedElement([V,R6,L1]) 
W = FunctionSpace(mesh,mixed_element) 
q = Function(W) 
u, c, p = split(q) 
v, k, pt = TestFunctions(W)
d = u.geometric_dimension()   

q_degree = 5
dx = dx(metadata={'quadrature_degree': q_degree}) 

#Vectors accounting for (infinitesimal) rotations and translations
e1 = Constant((1.0,0.0,0.0))
e2 = Constant((0.0,1.0,0.0))
e3 = Constant((0.0,0.0,1.0))
e4 = Expression(('-x[1]','x[0]',0.0), degree = 1)
e5 = Expression(('-x[2]',0.0,'x[0]'), degree = 1)
e6 = Expression((0.0,'-x[2]','x[1]'), degree = 1)
e = [e1,e2,e3,e4,e5,e6]

parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}


# Energy per unit volume + rigid constraints + incompressibility constraint
Vol = assemble(Constant(1.0)*dx) 
Energy = (Energy_density(u, F)/Vol)*dx  
for i in range(6): 
    Energy += (c[i]*inner(u,e[i]))*dx
Energy += (p*(det(nabla_grad(u)+Identity(d))-1.0))*dx

#Solve Functional Derivative of E dE = 0
Dv_Energy = derivative(Energy,q) 
Jacobian = derivative(Dv_Energy,q)
solve(Dv_Energy == 0,q,[],J=Jacobian, form_compiler_parameters=ffc_options, 
      solver_parameters={"newton_solver":{"linear_solver":"mumps", 
                                            "relaxation_parameter":1.0,
                                            'maximum_iterations': 50}}) 

u,c,p = q.split() 
with XDMFFile(dolfin.MPI.comm_world, save_dir+mesh_name+"/"+g_name+"_displacement_mech_equilibrium.xdmf") as xdmf_outfile:
    xdmf_outfile.write(u)

#Remove .xdmf mesh because its no longer needed
os.system("rm "+ mesh_name+".xdmf")
os.system("rm "+ mesh_name+".h5")

