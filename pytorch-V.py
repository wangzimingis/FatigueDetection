import torch
print(torch.__version__)       # 会显示 2.11.0+cu124
print(torch.cuda.is_available()) # True
print(torch.cuda.get_device_name(0)) # RTX3060