# -*- coding: utf-8 -*-
"""Optional: re-evaluate all nine APDD-DNV trained best.pt on their own VAL sets.

Does not train. Prints/saves a consistently aligned 3-dataset x 3-model VAL table.
Keep this program next to APDD_DNV_3Models_Unified_TEST.py in the E10 project.
Requires Ultralytics 8.4.3 custom fork. No TEST images are evaluated here.
"""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import APDD_DNV_3Models_Unified_TEST as common

OUTPUT = common.BASE_DIR / 'runs' / 'APDD_DNV_3Models_Unified_VAL'
FIELDS = ['Dataset','Model','Alpha','Images','Instances','EvaluatedClasses',
          'Precision(%)','Recall(%)','mAP50(%)','mAP50-95(%)','Status','SHA256','Error']


def parse():
    p=argparse.ArgumentParser(description='Run independent-process VAL on all three APDD-DNV models')
    p.add_argument('--datasets', nargs='+', choices=list(common.DATA_ROOTS), default=['night','mix','day'])
    p.add_argument('--models', nargs='+', choices=['yolov8','dyhead','mendnet'], default=['yolov8','dyhead','mendnet'])
    p.add_argument('--list', action='store_true')
    p.add_argument('--skip-existing', action='store_true')
    p.add_argument('--physical-gpu', default='0')
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--output',type=Path,default=OUTPUT)
    return p.parse_args()


def main():
    args=parse()
    common.require(args.batch>0 and args.workers>=0,'Invalid batch/workers')
    common.require(len(set(args.datasets))==len(args.datasets), 'Duplicate datasets')
    tasks=[]
    roots={}
    for ds in args.datasets:
        root=common.dataset_root(ds)
        common.require(root.is_dir(),f'Missing {ds} images root: {root}')
        val_images=common.images(root/'images'/'val')
        common.require(len(val_images)==common.EXPECTED_SPLITS[ds][1],
                       f'{ds}: expected {common.EXPECTED_SPLITS[ds][1]} Val images, got {len(val_images)}')
        gt={cid:0 for cid in range(6)}
        for img in val_images:
            label=root/'labels'/'val'/(img.stem+'.txt')
            if label.is_file():
                for line in label.read_text(encoding='utf-8-sig').splitlines():
                    if line.strip():
                        cls=int(line.split()[0]);common.require(cls in gt,f'Bad class {cls} in {label}')
                        gt[cls]+=1
        roots[ds]=(root,val_images,gt)
        print(f'[VAL_DATA] {ds} images={len(val_images)} GT={gt}',flush=True)
        for model in args.models:
            pt=common.checkpoint_for(ds,model)
            print(f'[WEIGHT] {ds}/{model} {"FOUND" if pt.is_file() else "MISSING"} {pt}',flush=True)
            common.require(pt.is_file(),f'Missing best.pt: {pt}')
            tasks.append((ds,model,pt))
    if args.list:
        print('[VAL_LIST_COMPLETE] no GPU inference, no file writes')
        return
    os.environ['CUDA_VISIBLE_DEVICES']='' if args.physical_gpu.lower()=='cpu' else args.physical_gpu
    import torch
    import ultralytics
    import ultralytics.utils.torch_utils as torch_utils
    from ultralytics.nn.tasks import BaseModel
    # The Windows THOP deform_conv2d fault is in model.info(), not genuine image inference.
    torch_utils.get_flops=lambda *a,**k: 0.0
    BaseModel.info=lambda self,*a,**k: None
    from ultralytics import YOLO
    common.require(ultralytics.__version__=='8.4.3', 'Expected Ultralytics 8.4.3')
    common.require(Path(ultralytics.__file__).resolve().parent.parent==common.BASE_DIR.resolve(),
                   f'Wrong fork: {ultralytics.__file__}')
    device='cpu' if args.physical_gpu.lower()=='cpu' else 0
    common.require(device=='cpu' or torch.cuda.is_available(), 'CUDA unavailable')
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    table=[]
    for ds,model,pt in tasks:
        root, val_images, gt=roots[ds]
        ptsha=common.sha(pt)
        identity={'checkpoint_sha256':ptsha,'dataset':ds,'model':model,
                  'images':len(val_images), 'val_sha256':
                  hashlib.sha256('\n'.join(f'{p.name}|{common.sha(p)}|'+
                  (common.sha(root/'labels'/'val'/(p.stem+'.txt')) if (root/'labels'/'val'/(p.stem+'.txt')).is_file() else 'NO_LABEL')
                  for p in val_images).encode('utf-8')).hexdigest(),
                  'protocol':{'split':'val','imgsz':896,'batch':args.batch,'rect':True,
                              'half':device!='cpu','conf':0.001,'iou':0.7,
                              'max_det':300,'workers':args.workers}}
        signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        d=output/ds/model
        rec=d/'aligned_val_metrics.json'
        if args.skip_existing and rec.is_file():
            cached=json.loads(rec.read_text(encoding='utf-8'))
            if cached.get('signature')==signature and cached.get('Status')=='OK':
                print(f'[VAL_REUSED] {ds}/{model}')
                table.append({k:cached.get(k,'') for k in FIELDS})
                continue
        d.mkdir(parents=True,exist_ok=True)
        obj=None
        try:
            alpha=common.check_checkpoint_structure(pt,model)
            yaml_path=output/'_dataset_yaml'/f'{ds}.yaml'
            yaml_path.parent.mkdir(parents=True,exist_ok=True)
            import yaml
            yaml_obj={'path':root.as_posix(),'train':'images/train','val':'images/val',
                      'test':'images/test','nc':6,'names':dict(enumerate(common.CLASSES))}
            yaml_str=yaml.safe_dump(yaml_obj,sort_keys=False,allow_unicode=True)
            if yaml_path.is_file():
                common.require(yaml_path.read_text(encoding='utf-8')==yaml_str,
                               f'Existing dataset YAML changed: {yaml_path}')
            else:
                yaml_path.write_text(yaml_str,encoding='utf-8')
            obj=YOLO(str(pt))
            m=obj.val(data=str(yaml_path),split='val',imgsz=896,batch=args.batch,
                      rect=True,half=(device!='cpu'),conf=0.001,iou=0.7,max_det=300,
                      workers=args.workers,seed=42,deterministic=True,plots=False,
                      save_txt=False,save_json=False,augment=False,device=device,
                      project=str(output/'_ultralytics'),name=f'{ds}_{model}',exist_ok=True,verbose=False)
            idxs=[int(k) for k in m.box.ap_class_index]
            expected_ids=[cid for cid in range(6) if gt[cid]>0]
            common.require(sorted(idxs)==expected_ids,
                           f'AP classes {idxs} disagree with GT {expected_ids}')
            perclass={common.CLASSES[k]:float(m.box.ap[idxs.index(k)]) for k in idxs}
            row={'Dataset':ds,'Model':model,'Alpha':alpha if alpha is not None else '',
                 'Images':len(val_images),'Instances':sum(gt.values()),
                 'EvaluatedClasses':len(idxs),
                 'Precision(%)':100*float(m.box.mp),
                 'Recall(%)':100*float(m.box.mr),
                 'mAP50(%)':100*float(m.box.map50),
                 'mAP50-95(%)':100*float(m.box.map),
                 'Status':'OK','SHA256':ptsha,'Error':''}
            common.require(abs(sum(perclass.values())/len(perclass)-float(m.box.map))<1e-7,
                           'Per-class AP mean does not equal global mAP')
            doc={**row,'signature':signature,'weights':str(pt),'protocol':identity['protocol'],
                 'evaluated_class_ids':idxs,'per_class_AP50_95':perclass,
                 'ground_truth_instances':gt}
            rec.write_text(json.dumps(doc,indent=2,ensure_ascii=False),encoding='utf-8')
            print(f'[VAL_DONE] {ds}/{model} mAP50={row["mAP50(%)"]:.4f}% '
                  f'mAP50-95={row["mAP50-95(%)"]:.4f}%',flush=True)
        except Exception as e:
            row={'Dataset':ds,'Model':model,'Status':'ERROR','SHA256':ptsha,
                 'Error':f'{type(e).__name__}: {e}'}
            (d/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
            print(f'[VAL_ERROR] {ds}/{model}: {e}',flush=True)
        finally:
            del obj
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
        table.append(row)
        with (output/'VAL_3Datasets_3Models.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS,extrasaction='ignore');writer.writeheader();writer.writerows(table)
    # write summary also if every result was cached
    with (output/'VAL_3Datasets_3Models.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS,extrasaction='ignore');writer.writeheader();writer.writerows(table)
    print(f'[VAL_COMPLETE] {sum(x.get("Status")=="OK" for x in table)}/{len(tasks)}; {output}',flush=True)
    if any(x.get('Status')!='OK' for x in table):sys.exit(2)

if __name__=='__main__':main()
