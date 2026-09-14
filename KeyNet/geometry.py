"""Homographies map source (x,y,1) pixel centres to destination coordinates."""
import torch
import torch.nn.functional as F


def transform(xy,H):
    flat=xy.reshape(len(H),-1,2)
    xyz=torch.cat((flat,torch.ones_like(flat[...,:1])),-1)
    dst=xyz@H.transpose(1,2)
    denom=dst[...,2:]
    # Keep the sign for projective matrices; never silently clamp negatives to positive.
    safe=torch.where(denom.abs()<1e-8,torch.full_like(denom,1e-8),denom)
    return (dst[...,:2]/safe).reshape_as(xy)


def warp(image,H,dsize=None):
    b,_,h,w=image.shape
    oh,ow=dsize or (h,w)
    yy,xx=torch.meshgrid(torch.arange(oh,device=image.device,dtype=image.dtype),torch.arange(ow,device=image.device,dtype=image.dtype),indexing='ij')
    xy=torch.stack((xx,yy),-1)[None].expand(b,-1,-1,-1)
    source=transform(xy,torch.linalg.inv(H))
    grid=torch.stack((source[...,0]*2/max(w-1,1)-1,source[...,1]*2/max(h-1,1)-1),-1)
    return F.grid_sample(image,grid,mode='bilinear',padding_mode='zeros',align_corners=True)


def border_mask(mask,border):
    result=mask.clone()
    if border:
        result[...,:border,:]=0; result[...,-border:,:]=0
        result[...,:,:border]=0; result[...,:,-border:]=0
    return result


def erode(mask,radius=1):
    return 1-F.max_pool2d(F.pad(1-mask,(radius,)*4,value=1),2*radius+1,stride=1)
