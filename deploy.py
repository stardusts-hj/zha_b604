import torch
import torch.nn as nn
from thop import profile
import os
from ptflops import get_model_complexity_info
def save_network(net, net_label='deploy', param_key='params'):
    """Save networks.

    Args:
        net (nn.Module | list[nn.Module]): Network(s) to be saved.
        net_label (str): Network label.
        current_iter (int): Current iter number.
        param_key (str | list[str]): The parameter key(s) to save network.
            Default: 'params'.
    """

    save_filename = f'{net_label}.pkl'
    save_path = os.path.join('./', save_filename)

    net = net if isinstance(net, list) else [net]
    param_key = param_key if isinstance(param_key, list) else [param_key]
    assert len(net) == len(
        param_key), 'The lengths of net and param_key should be the same.'

    save_dict = {}
    
    for net_, param_key_ in zip(net, param_key):
        state_dict = net_.state_dict()
        for key, param in list(state_dict.items()):
            if "total" in key:
                del(state_dict[key])
            elif "reparam" in key:
                del(state_dict[key])
                sp = key.split('.')
                sp = [i for i in sp if "reparam" not in i ]
                key = '.'.join(sp)
                state_dict[key] = param.cpu()
            else:
                state_dict[key] = param.cpu()
        save_dict[param_key_] = state_dict

    torch.save(save_dict, save_path)


def convert(param):
        return {
        k.replace("module.", ""): v
            for k, v in param.items()
            if "module." in k and 'attn_mask' not in k and 'HW' not in k
        }

from Trainer import Model
size = (3, 1080, 1920)
I0_ = (torch.randn(*size).cuda() / 255.).unsqueeze(0)
I2_ = (torch.randn(*size).cuda() / 255.).unsqueeze(0)
from benchmark.utils.padder import InputPadder
padder = InputPadder(I0_.shape, divisor=32)
I0_, I2_ = padder.pad(I0_, I2_)
print(I0_.shape)
model_path = 'output_final_stage0/250.pkl'
model = Model(-1, 'final_stage0')
model.device()
model.load_model(model_path, -1)
model.eval()



x = torch.randn(1,48,256,256).cuda()

y1 = model.net.refine.block1.convblock[0](x)
for m in model.net.modules():
    if hasattr(m, 'switch_to_deploy'):
        m.switch_to_deploy()

y2 = model.net.refine.block1.convblock[0](x)

print(((y1-y2)**2).mean()/ (y1**2).mean())
save_network(model.net)
# model.save_model('deploy.pkl')

# input_data = torch.randn((1, 3, args.crop_size[0], args.crop_size[1])).to(device)


