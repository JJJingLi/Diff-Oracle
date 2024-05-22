from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset

from dataset.train_dataset_content import MyDataset, resizeNormalize
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint

# Configs
resume_path = 'obc_style_model.ckpt'
log_dir = '/log_oracle241/'
dataset_dir = '/log_oracle241/prompt_scan_cut_train.json'

batch_size = 12
logger_freq = 300
learning_rate = 6e-5
sd_locked = True
only_mid_control = False
milestones = [20]

if __name__ == '__main__':
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./configs/content.yaml').cpu()
    pretrained_model = load_state_dict(resume_path, location='cpu')

    pretrained_model_all = {}
    for k in model.state_dict().keys():
        if 'control_model' not in k:
            pretrained_model_all[k] = pretrained_model[k].clone()
        else:
            if 'control_model2' in k:
                if 'zero_convs' not in k and 'input_hint_block' not in k:
                    pretrained_model_all[k] = pretrained_model[k.replace('control_model2', 'control_model')].clone()
                else:
                    print('!!')
            else:
                pretrained_model_all[k] = pretrained_model[k].clone()

    msg = model.load_state_dict(pretrained_model_all, strict=False)
    print(msg)


    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    model.milestones = milestones

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    # Misc
    dataset = MyDataset(dir=dataset_dir, imagesize=256, transform=resizeNormalize((256, 256)))

    dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True)

    logger = ImageLogger(batch_frequency=logger_freq, log_dir=log_dir)
    checkpoint_callback = ModelCheckpoint(save_top_k=-1, every_n_epochs=2)
    trainer = pl.Trainer(gpus=4, max_epochs=20, strategy='ddp', default_root_dir=log_dir, callbacks=[logger, checkpoint_callback])

    # Train!
    trainer.fit(model, dataloader)

