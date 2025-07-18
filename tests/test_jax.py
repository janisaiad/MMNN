import jax

def test_cuda_available():
    # Get all available devices
    devices = jax.devices()
    
    # Filter for GPU devices
    gpu_devices = [d for d in devices if d.platform == 'gpu']
    
    # Check if any GPU devices are available
    if gpu_devices:
        print(f"Found {len(gpu_devices)} GPU device(s):")
        for i, device in enumerate(gpu_devices):
            print(f"GPU {i}: {device}")
        print("\nJAX is configured to use CUDA")
    else:
        print("No GPU devices found - CUDA not available")
        
if __name__ == "__main__":
    test_cuda_available()

