import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
plt.rcParams.update({'font.size': 16})
plt.rcParams['figure.dpi']= 120

# Set up
L = 1 # m, rod length
n = 1024 # number of points

# space
x = np.linspace(0, L, n) # evenly distribute points
dx = L/n # delta x

# time
t_final = 601 # s, length of simulation

#%% Initialization

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
chunk = n // size

# function to compute temperature
def compute_temperature(x_chunk):
    func = lambda x: 20 + (30 * np.exp(-100 * (-0.5 + x) ** 2))
    return np.array(list(map(func, x_chunk))) # init temperature vector

# prepare the chunks of x data
x_data = None
if rank == 0:
    x_data = np.array([ x[int(i*chunk): int(int(i*chunk)+chunk)] for i in range(size)], dtype=np.float64)

# scatter the chunks of x to all processes
r_data = np.empty(int(chunk), dtype=np.float64)
comm.Scatter(x_data, r_data, root=0)

# each process computes its portion of T_initial
T_chunk = np.empty(chunk+2, dtype=np.float64)
T_chunk.fill(20)
T_chunk[1:-1] = compute_temperature(r_data)
    
    
#%% Solving over time

# coefficient
alpha = 2.3e-4 # m^2/s, aluminum diffusion coefficient

# Courant Friedrichs Lewy condition to assure that time steps are small enough
cfl = 0.5
dt = dx**2 * cfl / alpha
t = np.arange(0, t_final, dt)
len_t = len(t)

# snapshots
snap_index = [0, len_t//6, len_t//5, len_t//4, len_t//3, len_t//2, len_t-2] # index at which to take a snapshot

# time-stepping loop
for index, time in enumerate(t):
    
    # boundary communication for non-edge processes
    if rank > 0:
        comm.send(T_chunk[1], dest=rank-1)
        T_chunk[0] = comm.recv(source=rank-1)
    if rank < size - 1:
        comm.send(T_chunk[-2], dest=rank+1)
        T_chunk[-1] = comm.recv(source=rank+1)
        
    # update interior points
    T_chunk[1:-1] += alpha * dt / dx**2 * (T_chunk[:-2] - 2 * T_chunk[1:-1] + T_chunk[2:])

    # save snapshot of data
    if (index in snap_index):
        T_full = np.empty(n, dtype=np.float64)  # full array to gather into
        comm.Gather(T_chunk[1:-1], T_full, root=0)
        
        # plot snapshot if rank = 0
        if(rank == 0):
            time_stamp = int(dt*index)
            plt.plot(T_full, label=f"t = {time_stamp} s")
                 
# final plotting             
plt.title('Temperature of an Aluminium Rod')
plt.xlabel('x (m)')
plt.ylabel('T [C$\degree$]')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig('Temperature_Aluminium_Rod.pdf', dpi=300)
    
    