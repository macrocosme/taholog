import multiprocessing as mp
from datetime import datetime
import h5py
from utils import check_folder_exists_or_create
import os
import numpy as np

trunk_dir = '/data/bassa/taholog/'
cs_str = 'cs'
output_dir = '/scratch/vohl/output/'
processes=16

def writefile(output, beam):
    ctime_step = datetime.now()
    f = h5py.File(output, "w")
    f0 = f.create_group('0')
    b = f0.create_group('DATA')
    b['data'] = np.random.rand(1000, 1000)
    f.close()
    print (output, beam, datetime.now() - ctime_step)

def time_parallel_write():
    reference_ids = ['L2036944']
    beam_ids = ['0', '1', '2', '3']

    ctime_global = datetime.now()
    for ref in reference_ids:
        current_dir = f"{trunk_dir}{ref}/{cs_str}/"
        _outdir = check_folder_exists_or_create(f"{output_dir}{ref}", return_folder=True)
        os.chdir(current_dir)
        ctime_mid = datetime.now()
        pool = mp.Pool(processes=processes)
        for beam in beam_ids:
            pool.apply_async(writefile, 
                             args=(ref,beam))
        pool.close()
        pool.join()
        print (ref, datetime.now() - ctime_mid)
    print (datetime.now() - ctime_global)