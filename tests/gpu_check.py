import torch
import monai

print("PyTorch version:", torch.__version__)
print("MONAI version", monai.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

x = torch.rand(3,3).cuda()
print("Tensor on GPU:", x)
