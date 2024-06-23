import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import time

###### USER CONTROLS ######
save_data = False
plot_data = True
time_run = True
###########################

plt.rcParams.update({'font.size': 22})
plt.rcParams['figure.dpi'] = 120

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# parameters
w = h = 20.48 # plate size, mm
nx = ny = int(1024) # grid size
dx, dy = w / nx, h / ny # intervals in x-, y- directions, mm
dx2, dy2 = dx * dx, dy * dy
D = 4.2 # Thermal diffusivity of steel, mm2/s
nsteps = 1000 # time
dt = dx2 * dy2 / (2 * D * (dx2 + dy2))  # Using CFL to calculate the largest dt
F = D * dt 
plot_ts = [0, 100, 500, 950] # time stamps to save plot for

# initialization settings
Tcool, Thot = 300, 2000
cx, cy = w / 2, h / 2  # center of the domain
r = 5.12
r2 = r**2

# split along y axis
num_rows = ny // size
start_y = rank * num_rows
end_y = start_y + num_rows

# full nx by ny array to store temperatures
full_T = np.full((ny+2*size,nx), Tcool, dtype=np.float64)  # default to 300 K

# scatter the chunks of full_T to all processes
local_T = np.empty((num_rows+2,nx), dtype=np.float64)
comm.Scatter(full_T, local_T, root=0)

# initialize array for local_T
for i in range(nx):
    for j in range(num_rows+2):
        global_y = start_y + j
        x, y = i * dx, global_y * dy
        p2 = (x - cx) ** 2 + (y - cy) ** 2
        if p2 < r2:
            radius = np.sqrt(p2)
            local_T[j, i] = Thot * np.cos(4 * radius) ** 4
        
# set edges to 300 K     
local_T[:, 0] = local_T[:, -1] = Tcool  # side boundary conditions
if rank == 0:
    local_T[0, :] = Tcool  # boundary at the top
if rank == size - 1:
    local_T[-1, 0] = Tcool  # boundary at the bottom
    
# final data set to save
shape = (nsteps, ny,nx)
final_T_for_all_times = np.empty(shape, dtype=np.float64)
    
start_t = time.time()
# compute time steps        
for m in range(nsteps):
    
    if rank > 0:
        # top boundary
        comm.Send(local_T[1, :], dest=rank-1, tag=1) # send second top row to process above
        comm.Recv(local_T[0,:], source=rank-1, tag=2) # receive from process above and insert as top row
        
    if rank < size - 1:
        # bottom boundary
        comm.Recv(local_T[-1,:], source=rank+1, tag=1) # receive and insert as the bottom row
        comm.Send(local_T[-2, :], dest=rank+1, tag=2) # send second bottom row to process below
        
    # update temperature array
    local_T[1:-1, 1:-1] = local_T[1:-1, 1:-1] + F * (
        (local_T[2:, 1:-1] - 2 * local_T[1:-1, 1:-1] + local_T[:-2, 1:-1]) / dy2
        + (local_T[1:-1, 2:] - 2 * local_T[1:-1, 1:-1] + local_T[1:-1, :-2]) / dx2 )
    
    # apply 300 K to sides
    local_T[0], local_T[-1] = Tcool, Tcool
        
    final_T = np.empty((ny,nx), dtype=np.float64)  # full array to gather into for one time step
    comm.Gather(local_T[1:-1,:], final_T, root=0) # gather all processes
    
    final_T_for_all_times[m] = final_T # save to full data array
    
    # plot time snap
    if (plot_data ==True):
        if m in plot_ts:

            if (rank == 0):
                fig, ax = plt.subplots()  # create a figure and a set of subplots
                im = ax.imshow(final_T, cmap="hot", vmin=Tcool, vmax=Thot)  # plot the data
                ax.set_title("{:.1f} ms".format((m + 1) * dt * 1000))
                ax.set_xlim(200, 800)
                ax.set_ylim(200, 800)
                ax.set_xticks([350, 500, 650, 800])
                ax.set_yticks([350, 500, 650, 800])

                # create color bar
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("K", labelpad=20)
                plt.tight_layout()
              
                plt.savefig("./iter_{}.pdf".format(m), dpi=300)
                plt.clf()

# print timing results
if (time_run == True):
    end_t = time.time()
    time_compute = end_t-start_t
    print("It took me:", time_compute, "seconds in total")
    
# save data to text file
if (save_data == True):
    if (rank==0):
        # save the 3D array to a text file, using ',' as the separator
        with open('data.txt', 'w') as f:
            for i in range(0, len(final_T_for_all_times), 20):
                # save each 2D array to the file, specifying ',' as the delimiter
                np.savetxt(f, final_T_for_all_times[i], fmt='%d', delimiter=',')

                # write a separator after each 2D array except the last one
                if i < len(final_T_for_all_times) - 1:
                    f.write("#" * 10 + '\n')  # using '##############' as a separator between the arrays




        