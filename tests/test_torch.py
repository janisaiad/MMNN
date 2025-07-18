import torch
import pytest

def test_cuda_availability():
    is_cuda_available = torch.cuda.is_available()
    if is_cuda_available:
        device = torch.device("cuda:0")
        print(f"CUDA device: {device}")
        x = torch.tensor([1.0, 2.0, 3.0]).to(device)
        assert x.device.type == "cuda"
        assert torch.cuda.device_count() > 0
    else:
        pytest.skip("CUDA not available - skipping GPU tests")

def test_cuda_operations():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available - skipping GPU tests")
        
    device = torch.device("cuda:0")
    
    x = torch.randn(100, 100).to(device)
    y = torch.randn(100, 100).to(device)
    
    z = torch.matmul(x, y)
    assert z.device.type == "cuda"
    
    torch.cuda.empty_cache()
    assert torch.cuda.memory_allocated() >= 0

if __name__ == "__main__":
    test_cuda_availability()
    test_cuda_operations()