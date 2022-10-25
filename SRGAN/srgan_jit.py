import torch
import time

Model = torch.jit.load("./weights/script_Generator.pt")


for i in range(100000):
    randn = torch.randn(1,3,40,40).to('cuda', dtype=torch.float)
    st = time.time()
    result = Model(randn)
    ed = time.time()

    print(ed-st)
print(result.size())