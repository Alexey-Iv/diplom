import argparse
import json
from pathlib import Path
import platform
import torch
from torch.utils.data import DataLoader
from data import Pairs
from checkpoints import initialize
from keyNet.model.keynet_architecture import keynet
from train_utils import fix_randseed, train_epoch, validate


def parse_args(argv=None):
    p = argparse.ArgumentParser(description = 'Training-only repair of the uploaded KeyNet project')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--manifest', required=True) 
    p.add_argument('--out', required=True)
    p.add_argument('--mask-dir')
    p.add_argument('--ignore-masks', action='store_true')
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--patch-size', type=int, default=64)
    p.add_argument('--border', type=int, default=4)
    p.add_argument('--geometry', choices=['affine','iris-shift'], default='affine')
    p.add_argument('--max-angle', type=float,default=3.)
    p.add_argument('--max-scale', type=float,default=1.)
    p.add_argument('--max-shear', type=float,default=0.)
    p.add_argument('--max-shift', type=float,default=3.)
    p.add_argument('--windows', type=lambda s: [int(x) for x in s.split(',')], default=[8,16,24])
    p.add_argument('--factors', type=lambda s: [float(x) for x in s.split(',')], default=[256.,64.,16.])
    p.add_argument('--coordinate-weighting', default=True)
    p.add_argument('--score-activation', choices=['relu','softplus'], default='relu')
    p.add_argument('--hermite', action='store_true')
    p.add_argument('--init') 
    p.add_argument('--expand-input', action='store_true')
    p.add_argument('--resume')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--grad-clip', type=float, default=5.)
    p.add_argument('--topk', type=int, default=25)
    p.add_argument('--nms-size', type=int, default=5)
    p.add_argument('--pixel-threshold', type=float,default=3.)
    p.add_argument('--num-filters', type=int,default=8)
    p.add_argument('--num-learnable-blocks', type=int, default=3)
    p.add_argument('--num-levels-within-net', type=int, default=3)
    p.add_argument('--factor-scaling-pyramid', type=float, default=1.5)
    p.add_argument('--conv-kernel-size', type=int, default=5)
    p.add_argument('--max-steps', type=int)
    a = p.parse_args(argv)
    
    
    if a.init and a.resume: p.error('--init and --resume are different operations; choose one')
    if a.expand_input and not (a.init and a.hermite): p.error('--expand-input requires --init and --hermite')
    if a.patch_size<32 or a.border<0 or 2*a.border>=a.patch_size: p.error('Need patch>=32 and 0<=2*border<patch')
    if min(a.max_angle,a.max_shear,a.max_shift)<0 or a.max_scale<1: p.error('Invalid augmentation limits')
    if a.max_steps is not None and a.max_steps<1: p.error('max-steps must be >=1')
    if len(a.windows)!=len(a.factors) or any(s<2 or s>a.patch_size for s in a.windows) or any(w<=0 for w in a.factors): p.error('Invalid MSIP windows/factors')
    if a.conv_kernel_size%2==0 or a.conv_kernel_size<1: p.error('Odd convolution kernel required')
    return a


def run(args):
    torch.set_num_threads(args.threads)
    fix_randseed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    
    if (out/'last.pt').exists() and not args.resume: raise ValueError('Run exists; use --resume or new --out')
    
    train_data=Pairs(args,'train')
    val_data=Pairs(args,'val')
    
    g = torch.Generator()
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=g, num_workers=0)
    val_loader = DataLoader(val_data, batch_size = args.batch_size, shuffle=False, num_workers = 0)

    model = keynet(args, torch.device(args.device)).to(args.device)
    config = vars(args).copy()
    # Сохраняем путь к манифесту и его время модификации вместо SHA-256
    manifest_path = Path(args.manifest)
    config['manifest_path'] = str(manifest_path.resolve())
    config['manifest_mtime'] = manifest_path.stat().st_mtime if manifest_path.exists() else None
    
    # Сохраняем путь к init-файлу вместо его хэша
    config['init_path'] = str(Path(args.init).resolve()) if args.init else None
    if args.init:
        init_path = Path(args.init)
        config['init_mtime'] = init_path.stat().st_mtime if init_path.exists() else None
        print(json.dumps({'initialization':initialize(model,args.init,args.expand_input)}),flush=True)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',patience=5,factor=.5)
    
    start=0; best=-1.; best_epoch=-1
    if args.resume:
        ck=torch.load(args.resume,map_location='cpu',weights_only=True)
        ignored={'epochs','resume','init','init_path','init_mtime','expand_input','out','data_dir','mask_dir','manifest','manifest_path','manifest_mtime','device','threads'}
        current={k:v for k,v in config.items() if k not in ignored}; old={k:v for k,v in ck['config'].items() if k not in ignored}
        if current!=old: raise ValueError('Resume configuration/manifest differs')
        if ck.get('format')!='keynet-training-1': raise ValueError('Use --init for legacy or weights-only checkpoint')
        model.load_state_dict(ck['model'],strict=True); optimizer.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
        torch.set_rng_state(ck['torch_rng'])
        if torch.cuda.is_available() and ck['cuda_rng']: torch.cuda.set_rng_state_all(ck['cuda_rng'])
        start=ck['epoch']+1; best=ck['best']; best_epoch=ck['best_epoch']
        config['init_path']=ck['config'].get('init_path')
    (out/'config.json').write_text(json.dumps(config,indent=2))
    (out/'environment.json').write_text(json.dumps({'torch':str(torch.__version__),'python':platform.python_version(),'device':args.device},indent=2))
    def snapshot(epoch):
        return {'format':'keynet-training-1','config':config,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'epoch':epoch,'best':best,'best_epoch':best_epoch,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    if not args.resume:
        baseline=validate(val_loader,model,args)
        (out/'before_training.json').write_text(json.dumps(baseline,indent=2))
        best=baseline['val_repeatability_px']; best_epoch=-1
        torch.save(snapshot(-1),out/'best.pt')
    for epoch in range(start,args.epochs):
        train_data.epoch=epoch; g.manual_seed(args.seed+epoch)
        stats=train_epoch(train_loader,model,optimizer,args); val=validate(val_loader,model,args)
        score=val['val_repeatability_px']; improved=score>best
        if improved: best=score; best_epoch=epoch
        scheduler.step(score)
        row={'epoch':epoch,**stats,**val,'lr':optimizer.param_groups[0]['lr'],'smoke_only':args.max_steps is not None}
        with (out/'history.json').open('a') as f: f.write(json.dumps(row)+'\n')
        ck=snapshot(epoch)
        torch.save(ck,out/'last.pt')
        if improved: torch.save(ck,out/'best.pt')
        print(json.dumps(row),flush=True)
    return {'best_repeatability':best,'best_epoch':best_epoch,'out':str(out),'smoke_only':args.max_steps is not None}


if __name__=='__main__': print(json.dumps(run(parse_args()),indent=2))