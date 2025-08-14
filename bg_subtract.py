import h5py
import numpy as np
from tools.plot_Bscan import get_output_data, mpl_plot as mpl_plot_Bscan 

def direct_couping_subract(raw, bg, output, mode = ['rxs','rx1','Ez']):
    with h5py.File(raw, 'r') as f1:
        data1 = f1[mode[0]][mode[1]][mode[2]][()]
        dt = f1.attrs['dt']
        f1.close()
    with h5py.File(bg, 'r') as f1:
        data2 = f1[mode[0]][mode[1]][mode[2]][()]
        dt = f1.attrs['dt']
        f1.close()
    data_br = np.subtract(data1,data2)

    with h5py.File(output, 'w') as f_out:
        f_out.attrs['dt'] = dt
        dataset_path = '/'.join(mode)  # Make it dynamic
        f_out.create_dataset(dataset_path, data=data_br)

    # Draw data with normal plot
    rxnumber = 1
    rxcomponent = 'Ez'
    plt = mpl_plot_Bscan("merged_output_data", data_br, dt, rxnumber,rxcomponent)
    
    fig_width = 3
    fig_height = 5

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    plt.imshow(data_br, cmap='gray', aspect='auto')
    plt.axis('off')
    ax.margins(0, 0)  # Remove any extra margins or padding
    fig.tight_layout(pad=0)  # Remove any extra padding
    plt.show()

if __name__ == "__main__":
    raw = './Output_ge/WallObj/Wall_Obj0.out'
    bg = './Output_ge/Base/Base0.out'
    output = './Output_ge/Object0.out'
    direct_couping_subract(raw = raw , bg = bg , output= output)