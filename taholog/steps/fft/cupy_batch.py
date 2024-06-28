import cupy as cp
from cupy.cuda import stream

cp.fft.config.use_multi_gpus = True
cp.fft.config.set_cufft_gpus([0, 1])  # Adjust the GPU devices as needed

def fft_cupy_batch(cpu_data_batch, nspec, npol, nchan, polmap, device=0):
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    stream1 = stream.Stream()
    stream2 = stream.Stream()
    
    with cp.cuda.Device(device), stream1:
        # Allocate GPU memory for the batch
        batch_size = len(cpu_data_batch)
        gpu_fftdata_batch = cp.zeros((batch_size, nspec, npol, nchan), dtype=cp.complex64)
        cpu_fftdata_batch = []

        # Transfer data from CPU to GPU in batches asynchronously
        gpu_data_batch = []
        for cpu_data in cpu_data_batch:
            gpu_data_batch.append(cp.array(cpu_data.reshape(nspec, nchan, polmap.size), dtype=cp.float64))

        for i, gpu_data in enumerate(gpu_data_batch):
            with stream2:
                for p in range(npol):  # X Y
                    if len(polmap[p]) == 2:
                        i0, i1 = polmap[p]
                        gpu_fftdata_batch[i, :, p, :] = cp.fft.fftshift(
                            cp.fft.fft(gpu_data[:, :, i0] + 1j * gpu_data[:, :, i1], axis=1),
                            axes=(1,)
                        )
                    elif len(polmap[p]) == 1:
                        i0 = polmap[p]
                        gpu_fftdata_batch[i, :, p, :] = cp.fft.fftshift(
                            cp.fft.fft(gpu_data[:, :, i0], axis=1),
                            axes=(1,)
                        )

            # Transfer result back to CPU asynchronously
            with stream1:
                cpu_fftdata_batch.append(cp.asnumpy(gpu_fftdata_batch[i]))

        # Ensure all streams are synchronized
        stream1.synchronize()
        stream2.synchronize()

    return cpu_fftdata_batch

# Example usage
N = 10  # Number of times to run the function
batch_size = 2  # Size of each batch for processing

# Define necessary parameters
nspec = 1024  # Example value
npol = 2  # Example value
nchan = 2048  # Example value
polmap = [[0, 1], [2, 3]]  # Example value, adjust as necessary

cpu_data_list = [get_cpu_data() for _ in range(N)]  # Replace get_cpu_data with actual data fetching function

def run_fft_batches(cpu_data_list, batch_size, nspec, npol, nchan, polmap, num_gpus=2):
    def task(cpu_data_batch, device):
        return fft_cupy_batch(cpu_data_batch, nspec, npol, nchan, polmap, device)
    
    num_jobs = len(cpu_data_list)
    results = []
    
    for i in range(0, num_jobs, batch_size):
        batch = cpu_data_list[i:i+batch_size]
        results.extend(Parallel(n_jobs=num_gpus)(delayed(task)(batch, j % num_gpus) for j in range(len(batch))))

    return results

# Run the FFT in batches and maximize GPU usage
results = run_fft_batches(cpu_data_list, batch_size, nspec, npol, nchan, polmap, num_gpus=2)

# Process results as needed
for result in results:
    print(result.shape)
