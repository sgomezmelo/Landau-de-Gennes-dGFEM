import proplot as pplt
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import numpy as np
from fenics import *
import dolfin

save_fold = "Results"
save_subdir = input("Type name of subdirectory with Q solution xdmf file: ")
subfolder = input("Type name of folder with mesh: ")
mesh_name = input("Type name of mesh file: ")

geom_folder = "./Geometries/"
save_dir = "./"+save_fold + "/" + save_subdir

#Create the new mesh
mesh = dolfin.cpp.mesh.Mesh()

#Create Mesh Value Collection for Boundary conditions
mvc_subdomain = dolfin.MeshValueCollection("size_t", mesh, mesh.topology().dim())
mvc_boundaries = dolfin.MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)

#Read Mesh and Mesh subdomain from Gmsh into the empty mesh
with XDMFFile(MPI.comm_world, geom_folder+subfolder+"/"+mesh_name+".xdmf") as xdmf_infile:
    xdmf_infile.read(mesh)
    xdmf_infile.read(mvc_subdomain, "")

#Read Boundaries to specify boundary conditions
with XDMFFile(MPI.comm_world, geom_folder+subfolder+"/"+mesh_name+"_surf.xdmf") as xdmf_infile:
    xdmf_infile.read(mvc_boundaries, "")
    

T = TensorElement("DG", mesh.ufl_cell(), 0) #Define tensor space for Q tensor as symmetric tensor
Tspace = FunctionSpace(mesh,T)
Q = Function(Tspace)


with XDMFFile(dolfin.MPI.comm_world, save_dir+"/"+mesh_name+"_qsolution_LG.xdmf") as xdmf_outfile:
        xdmf_outfile.read_checkpoint(Q,"Q tensor")


lambs = [0.45, 0.55, 0.65]
n_e = 1.5758
n_o = 1.7378
N_grid = 300
N_z = 50
dtheta = np.pi/2.0
n_indep_comp = 5

theta_p  = float(input("Type angle of polarizer w.r.t x axis in multiples of pi: "))*np.pi
Rx = mesh.coordinates()[:, 0].max()
Ry = mesh.coordinates()[:, 1].max()
Lz = mesh.coordinates()[:, 2].max()
Rx0 = mesh.coordinates()[:, 0].min()
Ry0 = mesh.coordinates()[:, 1].min()
Lz0 = mesh.coordinates()[:, 2].min()

x = np.linspace(Rx0,Rx,N_grid)
y = np.linspace(Ry0,Ry,N_grid)
z = np.linspace(Lz0,Lz, N_z)
dz = (Lz-Lz0)/N_z
qsol = np.zeros((N_grid,N_grid,n_indep_comp))
e_P = np.array([np.cos(theta_p),np.sin(theta_p)])
e_A = np.array([np.cos(theta_p+dtheta),np.sin(theta_p+dtheta)])
P = np.zeros((2,2,N_grid,N_grid)).astype(complex)
nx = np.zeros((N_grid,N_grid))
ny = np.zeros((N_grid,N_grid))
I = np.zeros((len(lambs),N_grid,N_grid))

P[0,0,:,:] = 1.0+0*1j
P[1,1,:,:] = 1.0+0*1j

for l in range(len(lambs)):
    lamb = lambs[l]
    for k in range(N_z-1):
        for i in range(N_grid):
            for j in range(N_grid):
                cell  = mesh.bounding_box_tree().compute_first_entity_collision(Point(x[i],y[j],z[k]))
                max_cell = mesh.num_cells()
                if (cell<max_cell):
                    Q_t = np.reshape(Q(x[i],y[j],z[k]), (3,3))
                    eigvals, eigvec = eigh(Q_t)
                    n = eigvec[:,np.argmax(eigvals)]
                    if k == N_z-2:
                        nx[i,j] = n[0]
                        ny[i,j] = n[1]
                
                    cos_gamma = n[2]
                    sin_gamma = np.sqrt(n[0]**2+n[1]**2)
                
                    cos_a = n[0]/np.sqrt(n[0]**2+n[1]**2)
                    sin_a = n[1]/np.sqrt(n[0]**2+n[1]**2)
            
                    dn = n_o*n_e/np.sqrt((n_o*sin_gamma)**2 + (n_e*cos_gamma)**2) - n_o
                    phase_shift = np.exp(1j*2*np.pi*dn*dz/lamb)

                    UtSU = np.zeros((2,2)).astype(complex)
                    UtSU[0,0] = cos_a**2.0+phase_shift*sin_a**2.0
                    UtSU[1,1] = sin_a**2.0+phase_shift*cos_a**2.0
                    UtSU[0,1] = cos_a*sin_a*(1.0 - phase_shift)
                    UtSU[1,0] = cos_a*sin_a*(1.0 - phase_shift)

                    P[:,:,i,j] = np.einsum('nm, mp -> np',UtSU,P[:,:,i,j])

    P_eA = np.einsum('ijkl,j -> ikl', P, e_P)
    I[l,:,:] = np.absolute(np.einsum('ijl,i -> jl', P_eA, e_A))**2.0

    plt.figure()
    plt.imsave(save_dir+"/intensity_wavelength"+str(lamb)+"um_angle_"+str((theta_p/np.pi))+"pi.pdf",np.reshape(I[l,:,:],(N_grid, N_grid)), cmap = "magma")

total_I = np.reshape(np.sum(I, axis = 0),(N_grid, N_grid))
plt.figure()
plt.imsave(save_dir+"/"+mesh_name+"total_intensity_angle_"+str((theta_p/np.pi))+"pi.pdf",total_I, cmap = "magma")

step = 20
X1,X2 = np.meshgrid(x[::step],y[::step])
scal = 0.2
plt.figure()
plt.grid(visible = False)
plt.quiver(X1,X2,scal*ny[::step,::step],scal*nx[::step,::step], scale = 5.0)
plt.quiver(X1,X2,-scal*ny[::step,::step],-scal*nx[::step,::step], scale = 5.0)
plt.savefig(save_dir+"/"+mesh_name+"dir_field.pdf")

array_file_name = save_dir+"/intensity_"+mesh_name+".txt"
I_diag = np.diagonal(total_I)/np.max(total_I)
I_x = total_I[int(N_grid/2),:]/np.max(total_I)
I_array = np.asarray([x/Rx,np.sqrt(2)*x/Rx,I_diag,I_x]).T
np.savetxt(array_file_name,I_array)

fig = pplt.figure()
ax = fig.subplot(xlabel='r/R', ylabel='I (a.u.)')
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.format(fontname='Fira Math')

ax.plot(x/Rx,total_I[int(N_grid/2),:]/np.max(I), label = "x")
ax.plot(np.sqrt(2)*x/Rx,I_diag/np.max(I), label =  "diagonal")
ax.legend(loc='r', ncols=1, frame=False)
fig.save(save_dir+"/slices_"+mesh_name+".pdf")


