# some code from stackoverflow on detecting a GPU
# 'how do I check if pytorch is using the gpu'

import torch
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

# Additional Info when using cuda
if device.type == 'cuda':
    # Ops the next two lines were for TensorFlow
    #numGPUs = len(tf.config.list_physical_devices('GPU'))
    #print('Number of GPUs Available: ', numGPUs)
    
    # Get the number of GPUs available using PyTorch
    numGPUs = torch.cuda.device_count()
    print('Number of GPUs Available: ', numGPUs)


    # Loop through GPUs and print information
    for i in range(0, numGPUs):
        print('\n***  GPU number {} information ***'.format(i+1))
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1))
        print('Cached:   ', round(torch.cuda.memory_reserved(0) /1024**3,1))
        global_free, total_GPU_memory = torch.cuda.mem_get_info()
        print('global free memory: {} bytes, total GPU memory: {} bytes'.format(global_free, total_GPU_memory))
        print('global free memory: {} GB, total GPU memory: {} GB'.format(
            round(global_free / 1024**3,3), round(total_GPU_memory / 1024**3,3)))
        a = torch.cuda.get_device_properties(0)
        print('Cuda Device Properties:')
        print('    name:  {}'.format(a.name))
        print('    major: {}, minor: {}'.format(a.major, a.minor))
        #print('    minor: {}'.format(a.minor))
        print('    total_memory: {}'.format(a.total_memory))
        print('    multi_processor_count: {}\n'.format(a.multi_processor_count))
    
