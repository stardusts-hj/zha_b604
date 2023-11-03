import torch


ckpt = 'act_b64_300.pkl'
out_path = 'act_b64_300_convert.pkl'


model = torch.load(ckpt)


for i in range(1,4):
    for j in range(2):
        # conv
        if f'module.backone.backbone.block{i}.conv.{j*2}.weight' in model:
            model[f'module.backone.backbone.block{i}.conv.{j}.0.weight'] = model.pop(f'module.backone.backbone.block{i}.conv.{j*2}.weight')
            model[f'module.backone.backbone.block{i}.conv.{j}.0.bias'] = model.pop(f'module.backone.backbone.block{i}.conv.{j*2}.bias')
        # PRELU
        if f'module.backone.backbone.block{i}.conv.{j*2+1}.weight' in model:
            model[f'module.backone.backbone.block{i}.conv.{j}.1.weight'] = model.pop(f'module.backone.backbone.block{i}.conv.{j*2+1}.weight')

torch.save(model, out_path)