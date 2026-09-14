"""Inspect image/target/mask pairs BEFORE training, using its identical data loader."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from train import parse_args
from data import Pairs
from geometry import warp,erode

if __name__=='__main__':
    a=parse_args();ds=Pairs(a,'train');out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    rows=[];pictures=[]
    for i in range(min(16,len(ds))):
        b=ds[i];ma=erode(b['src_mask'][None]);mb=erode(b['dst_mask'][None])
        common=ma*(warp(mb,torch.linalg.inv(b['H'])[None])>.999)
        counts={}
        for size in a.windows:
            h,w=common.shape[-2:];crop=common[...,:h-h%size,:w-w%size]
            counts[str(size)]=int((torch.nn.functional.unfold(crop,size,stride=size).amin(1)>.999).sum())
        rows.append({'path':ds.records[i]['path'],'valid_fraction':float(common.mean()),'valid_windows':counts})
        panels=[]
        for key in ('src','dst','src_mask','dst_mask'):
            panels.append(Image.fromarray(np.uint8(b[key][0].numpy()*255)).convert('RGB'))
        line=Image.new('RGB',(a.patch_size*4,a.patch_size))
        for j,im in enumerate(panels):line.paste(im,(j*a.patch_size,0))
        pictures.append(line)
    canvas=Image.new('RGB',(a.patch_size*4,a.patch_size*len(pictures)))
    for i,im in enumerate(pictures):canvas.paste(im,(0,i*a.patch_size))
    canvas.save(out/'pairs.png');(out/'pairs.json').write_text(json.dumps(rows,indent=2))
    print('Saved pairs.png: source | transformed | source valid mask | transformed valid mask')
