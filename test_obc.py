from share import *

from torch.utils.data import DataLoader
from dataset.test_dataset_obc import MyDataset, resizeNormalize
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler_SC as DDIMSampler

import os
import numpy as np
import torch
import einops
from PIL import Image

strength = 1
ddim_steps = 50
eta = 0
scale = [2, 1]

# Configs
batch_size = 30
resume_path = 'obc_content_model.ckpt'
log_dir = '/log_obc/'
dataset_dir = '/log_obc/prompt_handprint_forscantrain.json'

with open('./scan_sort_data_label.txt', 'r') as f:
    label = f.readlines()

if __name__ == '__main__':

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./configs/content.yaml').cpu()
    pretrained_model = load_state_dict(resume_path, location='cpu')
    msg = model.load_state_dict(pretrained_model, strict=True)
    print(msg)

    model = model.cuda()

    dataset = MyDataset(dir=dataset_dir) #, transform=resizeNormalize((256, 256)))
    dataloader = DataLoader(dataset, num_workers=2, batch_size=batch_size, shuffle=False)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    model.eval()

    j = 0
    for iteration_B, batch in enumerate(dataloader):
        # seed = 1
        # seed_everything(seed)

        with torch.no_grad():
            content = batch['hint'].cuda().permute(0, 3, 1, 2)
            nullcontent = torch.zeros(content.shape).cuda()
            prompt = batch['txt']
            style = batch['jpg'].cuda().permute(0, 3, 1, 2)
            nullstyle = (torch.zeros(style.shape) - 1).cuda()

            cond = {"c_concat": [content], "c_crossattn": [model.get_learned_conditioning(prompt, style)]}
            un_cond = {"c_concat": [nullcontent], "c_crossattn": [model.get_learned_conditioning(prompt, nullstyle)]}
            un_cond_content = {"c_concat": [content], "c_crossattn": [model.get_learned_conditioning(prompt, nullstyle)]}

            shape = (4, 256 // 8, 256 // 8)

            model.control_scales = [strength] * 13
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch['hint'].size(0),
                                                         shape, cond, verbose=False, eta=eta,
                                                         unconditional_guidance_scale=scale,
                                                         unconditional_conditioning=[un_cond, un_cond_content])

            x_samples = model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        for i in range(x_samples.shape[0]):
            root = os.path.join(log_dir)
            root = os.path.join(root)
            os.makedirs(root, exist_ok=True)
            filename = "{}_b_{:06}.bmp".format(batch['label'][i], j)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(x_samples[i]).save(path)
            j += 1
