import argparse
import csv
import json
import math
import re
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from geometry import warp,border_mask


def subject_from_path(path):
    p=Path(path)
    m=re.fullmatch(r'S5(\d{3})[LR]\d{2}',p.stem,re.I)
    if m: return m[1]
    m=re.fullmatch(r'(\d{3,4})[_-][LR][_-]\d+',p.stem,re.I)
    if m: return m[1]
    for i in range(1,len(p.parts)-1):
        if p.parts[i].upper() in ('L','R') and p.parts[i-1].isdigit(): return p.parts[i-1]
    raise ValueError(f'Unknown subject for {path}; supply metadata CSV path,subject')


def prepare(root,manifest,masks=None,metadata=None,seed=42):
    root=Path(root).resolve(); masks=Path(masks).resolve() if masks else None
    if masks==root: raise ValueError('Use a separate mask directory')
    if Path(manifest).exists(): raise ValueError('Manifest already exists; use it or choose a new filename')
    if metadata:
        with open(metadata,encoding='utf-8-sig',newline='') as f: entries=list(csv.DictReader(f))
    else:
        entries=[{'path':p.relative_to(root).as_posix()} for p in sorted(root.rglob('*'))
                 if p.is_file() and p.suffix.lower() in ('.bmp','.png','.jpg','.jpeg') and not (masks and p.is_relative_to(masks))]
    if not entries: raise ValueError('No images found')
    records=[]
    for e in entries:
        rel=Path(e['path']); p=(root/rel).resolve()
        # Проверка на вложенность пути для Python < 3.9
        try:
            p.resolve().relative_to(root.resolve())
        except ValueError:
            raise ValueError('Paths must be relative and contained')

        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Paths must be relative and contained')
        with Image.open(p) as im: size=im.size
        mask_path=(masks/rel).with_suffix('.png') if masks else None
        if mask_path:
            with Image.open(mask_path) as im:
                if im.size!=size: raise ValueError(f'Image/mask size mismatch: {rel}')
                a=np.asarray(im.convert('L'))
                if not set(np.unique(a)).issubset({0,1,255}) or not a.any(): raise ValueError(f'Invalid binary mask: {rel}')
        records.append({
            'path':rel.as_posix(),
            'subject':str(e.get('subject') or subject_from_path(rel)),
            'size':list(size)
        })
    people=sorted({r['subject'] for r in records})
    if len(people)<6: raise ValueError('Need >=6 subjects for three splits')
    np.random.default_rng(seed).shuffle(people); n=max(1,int(.15*len(people)))
    groups={s:('test' if i<n else 'val' if i<2*n else 'train') for i,s in enumerate(people)}
    for r in records: r['split']=groups[r['subject']]
    manifest=Path(manifest); manifest.parent.mkdir(parents=True,exist_ok=True)
    result={'format':1,'seed':seed,'records':records}
    manifest.write_text(json.dumps(result,indent=2))
    return {s:{'images':sum(r['split']==s for r in records),'subjects':len({r['subject'] for r in records if r['split']==s})} for s in ('train','val','test')}


def load_manifest(path):
    data=json.loads(Path(path).read_text())
    if data.get('format')!=1 or not data.get('records'): raise ValueError('Invalid manifest')
    people={}; paths=set()
    for r in data['records']:
        if r['split'] not in ('train','val','test'): raise ValueError('Unknown split')
        if r['subject'] in people and people[r['subject']]!=r['split']: raise ValueError('Subject leakage')
        people[r['subject']]=r['split']
        if r['path'] in paths: raise ValueError('Duplicate image path')
        rel=Path(r['path'])
        if rel.is_absolute() or '..' in rel.parts: raise ValueError('Unsafe path')
        paths.add(r['path'])
    return data


class Pairs(Dataset):
    """Self-supervised pairs; subject labels only determine train/val/test membership."""
    def __init__(self,args,split):
        self.args=args; self.split=split; self.epoch=0
        self.root=Path(args.data_dir); self.masks=Path(args.mask_dir) if args.mask_dir else None
        self.records=[r for r in load_manifest(args.manifest)['records'] if r['split']==split]
        if not self.records: raise ValueError(f'Empty {split} split')
        for r in self.records:
            if min(r['size'])<args.patch_size: raise ValueError(f'Image smaller than patch_size: {r["path"]} {r["size"]}')
    def __len__(self): return len(self.records)
    def __getitem__(self,i):
        args=self.args; rec=self.records[i]; size=args.patch_size
        rng=np.random.default_rng(args.seed+i+1000003*(self.epoch if self.split=='train' else 0))
        with Image.open(self.root/rec['path']) as im: full=np.array(im.convert('L'),dtype=np.float32)/255
        if self.masks and not args.ignore_masks:
            with Image.open((self.masks/rec['path']).with_suffix('.png')) as im: mask=(np.array(im.convert('L'))>0).astype(np.float32)
        else: mask=np.ones_like(full)
        h,w=full.shape
        # Uniform random crop instead of fixed central crop. Equal image/patch size is valid.
        top=int(rng.integers(0,h-size+1)); left=int(rng.integers(0,w-size+1))
        src=torch.from_numpy(full[top:top+size,left:left+size].copy())[None]
        sm=torch.from_numpy(mask[top:top+size,left:left+size].copy())[None]
        angle=math.radians(rng.uniform(-args.max_angle,args.max_angle)) if args.geometry=='affine' else 0.
        scale=rng.uniform(1/args.max_scale,args.max_scale) if args.geometry=='affine' else 1.
        shear=rng.uniform(-args.max_shear,args.max_shear) if args.geometry=='affine' else 0.
        dx=rng.uniform(-args.max_shift,args.max_shift)
        dy=rng.uniform(-args.max_shift,args.max_shift) if args.geometry=='affine' else 0.
        rotation=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]],dtype=np.float32)
        linear=np.array([[1.,shear],[0.,1.]],dtype=np.float32)@rotation*scale
        centre=np.array([(size-1)/2,(size-1)/2],dtype=np.float32)
        H=np.eye(3,dtype=np.float32); H[:2,:2]=linear; H[:2,2]=centre-linear@centre+[dx,dy]
        H=torch.from_numpy(H)
        dst=warp(src[None],H[None])[0]
        dm=(warp(sm[None],H[None])[0]>.999).float()
        # Grayscale photometry; no HSV/channel-shuffling for NIR data.
        dst=(dst*rng.uniform(.85,1.15)+rng.uniform(-.03,.03)).clamp(0,1)
        input_sm=sm; input_dm=dm
        sm=border_mask(sm,args.border); dm=border_mask(dm,args.border)
        return {'src':src,'dst':dst,'src_mask':sm,'dst_mask':dm,'src_input_mask':input_sm,'dst_input_mask':input_dm,'H':H,'index':i}


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',required=True); p.add_argument('--manifest',required=True)
    p.add_argument('--mask-dir'); p.add_argument('--metadata'); p.add_argument('--seed',type=int,default=42)
    a=p.parse_args(); print(json.dumps(prepare(a.data_dir,a.manifest,a.mask_dir,a.metadata,a.seed),indent=2))