from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset

from dataset.train_dataset_style import MyDataset, resizeNormalize
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint

# Configs
resume_path = '/Data_PHD/phd19_jing_li/checkpoints/Oracle_diff/control_sd15_ini.ckpt'
log_dir = '/log_oracle241/'
dataset_dir = '/log_oracle241/prompt_scan_cut_train.json'

batch_size = 14
logger_freq = 300
learning_rate = 6e-4
sd_locked = True
only_mid_control = False
milestones = [70, 130]

if __name__ == '__main__':
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./configs/style.yaml').cpu()
    msg = model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
    print(msg)

    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    model.milestones = milestones

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    # Misc
    dataset1 = MyDataset(dir=dataset_dir, imagesize=256, transform=resizeNormalize((256, 256)))

    dataloader = DataLoader(dataset1, num_workers=4, batch_size=batch_size, shuffle=True)

    logger = ImageLogger(batch_frequency=logger_freq, log_dir=log_dir)
    checkpoint_callback = ModelCheckpoint(save_top_k=-1, every_n_epochs=2)
    trainer = pl.Trainer(gpus=4, max_epochs=130, strategy='ddp', default_root_dir=log_dir, callbacks=[logger, checkpoint_callback])

    # Train!
    trainer.fit(model, dataloader)

