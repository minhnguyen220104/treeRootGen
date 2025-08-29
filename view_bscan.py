import h5py
import numpy as np
from tools.plot_Bscan import get_output_data, mpl_plot as mpl_plot_Bscan 
def process_br(raw_ra):
    raw_br = raw_ra - np.mean(raw_ra, axis=1, keepdims=True)
    return raw_br

def view_bscan(raw, output, mode = ['rxs','rx1','Ez']):
    with h5py.File(raw, 'r') as f1:
        data = f1[mode[0]][mode[1]][mode[2]][()]
        dt = f1.attrs['dt']
        f1.close()

    with h5py.File(output, 'w') as f_out:
        f_out.attrs['dt'] = dt
        dataset_path = '/'.join(mode)  # Make it dynamic
        f_out.create_dataset(dataset_path, data=data)

    # Draw data with normal plot
    rxnumber = 1
    rxcomponent = 'Ez'
    # print(data.shape)
    # data = process_br(data[:,:])
    data = data[2000:,:]
    data = process_br(data)
    from scipy.ndimage import gaussian_filter
    # data = gaussian_filter(data, sigma)

    plt = mpl_plot_Bscan("merged_output_data", data, dt, rxnumber,rxcomponent)
    
    fig_width = 3
    fig_height = 8

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    plt.imshow(data, cmap='viridis', aspect='auto')
    plt.axis('off')
    ax.margins(0, 0)  # Remove any extra margins or padding
    fig.tight_layout(pad=0)  # Remove any extra padding
    plt.show()

if __name__ == "__main__":
    raw = './Input_ge/Roots/Roots17_merged.out'
    output = './Output_ge/Hete0.out'
    view_bscan(raw = raw , output= output)