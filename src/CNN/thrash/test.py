import h5py

# with h5py.File('data/train-image.hdf5', "r") as f:
#     def print_tree(name, obj):
#         print(name, obj)
#
#     f.visititems(print_tree)

# import io
# from PIL import Image
# import numpy as np
#
# with h5py.File('data/train-image.hdf5', "r") as f:
#     raw_bytes = f["ISIC_9999932"][()]
#     image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
#     image = np.array(image)
#
# from matplotlib import pyplot as plt
# plt.imshow(image, interpolation='nearest')
# plt.show()
#
# import torch
#
# print("PyTorch version:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())
#
# if torch.cuda.is_available():
#     print("CUDA device count:", torch.cuda.device_count())
#     print("Current device:", torch.cuda.current_device())
#     print("Device name:", torch.cuda.get_device_name(0))
# else:
#     print("Running on CPU")

