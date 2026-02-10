import solver as sl
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from pathlib import Path



def main():
    # Define the directory path
    dir_path = Path('./simulation/')
    plt.rcParams['animation.ffmpeg_path'] ='C:\\ffmpeg\\bin\\ffmpeg.exe'

    # Iterate through all files in the directory
    for file in dir_path.iterdir():
        if file.is_file():
            filename = file.name.split(".")[0]
            print(f"Creating animation for: {filename}")
            time,data = sl.Solver.read_from_file(filename = f"./simulation/{filename}.dat")
            x = data[:,:,0]
            y = data[:,:,1]
            rho = data[:,:,8]
            generate_animation(time,x,y,rho,filename)

    
def generate_animation(time,x,y,rho,filename):  
    ## initialize plot
    fig, ax = plt.subplots()

    ## set limits of plot
    ax.set_xlim(-2e9,2e9)
    ax.set_ylim(-2e9,2e9)
    # ax.set_xlim(x.min() - np.abs(x.min()) * 0.1 - 1, x.max() + np.abs(x.max()) * 0.1)
    # ax.set_xlim(y.min() - np.abs(y.min()) * 0.1 - 1, y.max() + np.abs(y.max()) * 0.1)
    #ax.set_ylim(data[..., 1].min() - np.abs(data[..., 1].min()) * 0.1 - 1, data[..., 1].max() + np.abs(data[..., 1].max()) * 0.1)

    ## plot first dataslice
    scat = ax.scatter(x,y, animated=True)

    # update function
    data = np.zeros((*x.shape,2))
    data[:,:,0] = x
    data[:,:,1] = y
    def update(frame):
        # update positons of data points
        scat.set_offsets(data[frame,:,:])
        scat.set_array(rho[frame,:])

        return scat,

    # create animation
    ani = FuncAnimation(fig, update, frames=len(time), interval=50, blit=True)

    # save animation
    #ani.save(f"./animation/{filename}.gif", writer=PillowWriter(fps=20))

    FFwriter = animation.FFMpegWriter(fps=20)
    ani.save(f'./animation/{filename}.mp4', writer = FFwriter)

if __name__ == "__main__":
    main()
