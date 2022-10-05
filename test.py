import torch
import torch.nn as nn

mse = nn.MSELoss()

def PSNR(HRimage, LRimage):
    # peak signal to noise ratio r -> maximam value
    r = 255
    mseloss = mse(HRimage, LRimage)
    psnr = 20*torch.log10(r/torch.sqrt(mseloss))
    print(psnr)
    pass



HRimage = torch.randn(1,3,256,256)
LRimage = torch.randn(1,3,256,256)

PSNR(HRimage, LRimage)
