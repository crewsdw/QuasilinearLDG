import numpy as np
import h5py


class Data:
    def __init__(self, folder, filename):
        self.write_filename = folder + filename + '.hdf5'
        self.info_name = folder + filename + '_info.txt'

    def create_file(self, distribution, spectrum, time):
        # Open file for writing
        with h5py.File(self.write_filename, 'w') as f:
            # Create datasets, dataset_distribution =
            f.create_dataset('distribution', data=np.array([distribution]),
                             chunks=True,
                             maxshape=(None, distribution.shape[0], distribution.shape[1]),
                             dtype='f')
            f.create_dataset('spectrum', data=np.array([spectrum]),
                             chunks=True,
                             maxshape=(None, spectrum.shape[0]),
                             dtype='f')
            f.create_dataset('time', data=[time], chunks=True, maxshape=(None,))

    def save_data(self, distribution, spectrum, time):
        # Open for appending
        with h5py.File(self.write_filename, 'a') as f:
            # Add new time line
            f['distribution'].resize((f['distribution'].shape[0] + 1), axis=0)
            f['spectrum'].resize((f['spectrum'].shape[0] + 1), axis=0)
            f['time'].resize((f['time'].shape[0] + 1), axis=0)
            # Save data
            f['distribution'][-1] = distribution
            f['spectrum'][-1] = spectrum
            f['time'][-1] = time

    def save_inventories(self, total_energy, total_density):
        with h5py.File(self.write_filename, 'a') as f:
            # Add new time line
            f['total_energy'].resize((f['total_energy'].shape[0] + 1), axis=0)
            f['total_density'].resize((f['total_density'].shape[0] + 1), axis=0)
            # Save data
            f['total_energy'][-1] = total_energy.get()
            f['total_density'][-1] = total_density.get()

    def read_file(self):
        # Open for reading
        with h5py.File(self.write_filename, 'r') as f:
            time = f['time'][()]
            distribution = f['distribution'][()]
            spectrum = f['spectrum'][()]
        return time, distribution, spectrum
