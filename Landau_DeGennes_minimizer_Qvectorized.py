from fenics import *
import dolfin
import numpy as np
from ufl import nabla_div, VectorElement, FiniteElement, MixedElement, split
import math 
import sys
import os


# Code to compute the director field n as a minimizer of the Landau de Gennes (LdG) energy functional
# using the discontinuous Galerkin finite element method. The code takes as input a Fenics/dolfin suitable geometry
# in xdmf format and ratio of constants L/c and a/c, while assuming b = c.

#### 1.Import files and define constants ####

#Create folder to save results
save_fold = "Results"
save_subdir = input("Type name of subdirectory to save: ")
save_dir = "./"+save_fold + "/" + save_subdir
os.system("mkdir "+" "+save_dir)
surf_marker = 101 #Label of physical surfaces to enforce strong anchoring
n_indep_comp = 5 #Number of independent components (3(3+1)/2 -1 = 5 for a symmetric traceless 3x3 tensor)
a_B = 1.0/10.0 #Ratio of LdG constants A/B = A/C (assuming B and C are equal)
L_c = 2e+6 # Coherence length of nematic LC L/B in um
S0 = (1.0/2.0)*((1.0/3.0)+np.sqrt((1.0/3.0)**2+8.0*a_B/3.0)) # Corresponding LdG S eigenvalue

#Import .xdmf geometry file into mesh + Mesh value collection for boundary conditions
subfolder = input("Type name of .xdmf geometry folder: ")
mesh_name = input("Type name of .xdmf geometry file: ")
geom_folder = "./Geometries/"
mesh = dolfin.cpp.mesh.Mesh()

mvc_subdomain = dolfin.MeshValueCollection("size_t", mesh, mesh.topology().dim())
mvc_boundaries = dolfin.MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)

with XDMFFile(MPI.comm_world, geom_folder+subfolder+"/"+mesh_name+".xdmf") as xdmf_infile:
    xdmf_infile.read(mesh)
    xdmf_infile.read(mvc_subdomain, "")

with XDMFFile(MPI.comm_world, geom_folder+subfolder+"/"+mesh_name+"_surf.xdmf") as xdmf_infile:
    xdmf_infile.read(mvc_boundaries, "")

domains = dolfin.cpp.mesh.MeshFunctionSizet(mesh, mvc_subdomain)
boundaries = dolfin.cpp.mesh.MeshFunctionSizet(mesh, mvc_boundaries)
dx = Measure('dx', domain=mesh, subdomain_data=domains)
ds = Measure('ds', domain=mesh, subdomain_data=domains)

print("Number of elements: ", mesh.num_cells())

def LG_energy(q,a_B,L_c):
    #Landau de Gennes energy functional with a single elastic constant L
    Q = as_tensor(((q[0],q[1],q[2]),(q[1],q[3],q[4]),(q[2],q[4],-(q[0]+q[3]))))
    dQ = nabla_grad(Q)
    tQ2 = inner(Identity(d),dot(Q,Q))
    tQ3 = inner(Identity(d),dot(Q,dot(Q,Q)))
    E = L_c*inner(dQ,dQ)/2.0 - a_B*tQ2/2.0 - tQ3/3.0 + tQ2**2/4.0
    return E

#Unnormalized one constant Frank energy grad(n)*grad(n)
def Frank_E(u):
   return inner(nabla_grad(u),nabla_grad(u))/2.0

#Define suitable vector, tensor and scalar spaces     
T = TensorElement("DG", mesh.ufl_cell(), 0) 
Tspace = FunctionSpace(mesh,T)
V = VectorElement("CG", mesh.ufl_cell(), 1) 
Lagrange = FiniteElement("DG", mesh.ufl_cell(), 0) 
Vspace = FunctionSpace(mesh,V) 
mixed_element = MixedElement([Lagrange for i in range(n_indep_comp)]) #Mixed element for q coefficients 
W = FunctionSpace(mesh,mixed_element) 
q = Function(W) 
d = q.geometric_dimension()  
q_degree = 5
dx = dx(metadata={'quadrature_degree': q_degree}) 

#Compute boundary normal for Boundary Conditions (or surface energy contribution)
normals = FacetNormal(mesh)
u = TrialFunction(Vspace)
v = TestFunction(Vspace)
a = inner(u,v)*ds
l = inner(normals, v)*ds
A = assemble(a, keep_diagonal=True)
L = assemble(l)

A.ident_zeros()
n = Function(Vspace)

solve(A, n.vector(), L)
print("Obtained normal")

Q_b = (S0/2.0)*(d*outer(n,n)-Identity(d)) # Specify Q tensor at boundary
q0 = [Q_b[0,0],Q_b[0,1],Q_b[0,2],Q_b[1,1],Q_b[1,2]] # Extract upper half of Q at boundary

#Strong anchoring as Dirichlet BC 
bc1 = [DirichletBC(W.sub(i), project(q0[i],W.sub(i).collapse(), solver_type = "gmres", preconditioner_type = "ilu"), boundaries, surf_marker) for i in range(n_indep_comp)]
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Minimize Frank functional to use as initial guess in LdG minimization
n0 = Function(Vspace)
FrankE = Frank_E(n0)*dx
dE_Frank  = derivative(FrankE,n0)
JacFrank = derivative(dE_Frank,n0)
bc_f = [DirichletBC(Vspace, n, boundaries, surf_marker)]

solve(dE_Frank == 0,n0,bc_f,J=JacFrank, form_compiler_parameters=ffc_options, solver_parameters={"newton_solver":
                                        {"relative_tolerance": 1e-6,"absolute_tolerance": 1e-4, 'maximum_iterations': 150, "linear_solver":"gmres", "preconditioner":"ilu"}})

Q_guess = project(S0*((outer(n0,n0)/inner(n0,n0))-Identity(d)/d),FunctionSpace(mesh,T), solver_type = 'gmres', preconditioner_type = 'ilu') 
q_guess = [Q_guess.sub(0),Q_guess.sub(1),Q_guess.sub(2),Q_guess.sub(4),Q_guess.sub(5)]
assign(q,q_guess)

print("Solved Frank energy Minimizer for initialization into LdG")

Volume = assemble(Constant(1.0)*dx) 
Energy = (LG_energy(q,a_B,L_c)/Volume)*dx 
Dv_Energy = derivative(Energy,q) #Functional derivative dE
Jacobian = derivative(Dv_Energy,q) #Jacobian
solve(Dv_Energy == 0,q,[*bc1],J=Jacobian, form_compiler_parameters=ffc_options, solver_parameters={"newton_solver":
                                       {"relative_tolerance": 1.0e-11,"absolute_tolerance": 1.0e-9, 'maximum_iterations': 550, "relaxation_parameter":0.5, "linear_solver":"gmres", "preconditioner":"sor"}}) 

Q = project(as_tensor(((q[0],q[1],q[2]),(q[1],q[3],q[4]),(q[2],q[4],-(q[0]+q[3])))), Tspace, solver_type = 'gmres', preconditioner_type = 'ilu')
with XDMFFile(dolfin.MPI.comm_world, save_dir+"/"+mesh_name+"_qsolution_LG.xdmf") as xdmf_outfile:
    xdmf_outfile.write_checkpoint(Q,"Q tensor",0,XDMFFile.Encoding.HDF5, True)
    
print("Solved variational problem for Q")
