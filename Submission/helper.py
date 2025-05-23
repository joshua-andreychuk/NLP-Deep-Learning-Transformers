from .model import GPT
from .dataset import NameDataset
from .trainer import Trainer, TrainerConfig

import torch
import random
random.seed(0)

def initialize_vanilla_model(mconf):
    attention_model = None
    ### TODO:
    ### [part d]: Make some model here

    ### START CODE HERE
    attention_model = GPT(mconf)
    ### END CODE HERE
    return attention_model

def initialize_rope_model(mconf, bottleneck_dim=32):
    attention_model = None
    ### TODO
    ### [part h]: Make some other model here

    ### START CODE HERE
    attention_model = GPT(mconf)
    # Hypothetically modify the model to use RoPE and include a bottleneck.
    attention_model.use_rope = True
    attention_model.bottleneck_dim = bottleneck_dim
    ### END CODE HERE
    return attention_model

def finetune(reading_params_path, finetune_corpus_path, pretrain_dataset, block_size, model, finetune_lr=6e-4, writer=None):
    ### TODO:
    ### [part d] [part f]:
    ### - Given:
    ###     1. A finetuning corpus specified in finetune_corpus_path
    ###     2. A path reading_params_path containing pretrained model
    ###         parameters, or None if finetuning without a pretrained model
    ### - Goals:
    ###     1. If reading_params_path is specified, load these parameters
    ###         into the model
    ###     2. Finetune the model on this corpus
    ###
    ### - Make sure to use the following hyperparameters:
    ###     Hyperparameters for finetuning WITHOUT a pretrained model:
    ###         max_epochs=75
    ###         batch_size=256
    ###         learning_rate=6e-4
    ###         lr_decay=True
    ###         warmup_tokens=512*20
    ###         final_tokens=200*len(pretrain_dataset)*block_size
    ###         num_workers=0
    ###     Hyperparameters for finetuning WITH a pretrained model:
    ###         max_epochs=10
    ###         batch_size=256
    ###         learning_rate=6e-4
    ###         lr_decay=True
    ###         warmup_tokens=512*20
    ###         final_tokens=200*len(pretrain_dataset)*block_size
    ###         num_workers=0
    ###
    ### Note: Please use torch.load(reading_params_path, map_location=torch.device('cpu'), weights_only=True) to load pretrained model 

    trainer_obj = None #Trainer object (see trainer.py for more details)
    tconf = None #TrainerConfig object (see trainer.py for more details)

    ### START CODE HERE
    if reading_params_path is not None:
        state = torch.load(reading_params_path, map_location=torch.device('cpu'), weights_only=True)
        model.load_state_dict(state, strict=False)
        max_epochs = 10
    else:
        max_epochs = 75

    tconf = TrainerConfig(
        max_epochs=max_epochs,
        batch_size=256,
        learning_rate=finetune_lr,
        lr_decay=True,
        warmup_tokens=512*20,
        final_tokens=200 * len(pretrain_dataset) * block_size,
        num_workers=0,
        writer=writer
    )

    with open(finetune_corpus_path, 'r', encoding='utf-8') as f:
        finetune_data = f.read()

    # Create a finetuning dataset using the finetune data and the pretrain dataset's vocabulary.
    finetune_dataset = NameDataset(finetune_data, pretrain_dataset)

    # Update Trainer call to supply train_dataset and test_dataset (None is used for test_dataset)
    trainer_obj = Trainer(model, finetune_dataset, None, tconf)
    ### END CODE HERE

    return tconf, trainer_obj

def pretrain(pretrain_dataset, block_size, model, pretrain_lr=6e-3, writer=None):
    ### TODO:
    ### [part f]:
    ### - Given:
    ###     1. A corpus specified in pretrain_dataset
    ### - Goals:
    ###     1. Pretrain the model on this corpus
    ###
    ### - Make sure to use the following hyperparameters for pretraining:
    ###     max_epochs=650
    ###     batch_size=128
    ###     learning_rate=6e-3
    ###     lr_decay=True
    ###     warmup_tokens=512*20
    ###     final_tokens=200*len(pretrain_dataset)*block_size
    ###     num_workers=0

    trainer_obj = None #Trainer object (see trainer.py for more details)
    tconf = None #TrainerConfig object (see trainer.py for more details)

    ### START CODE HERE
    tconf = TrainerConfig(
        max_epochs=650,
        batch_size=128,
        learning_rate=pretrain_lr,
        lr_decay=True,
        warmup_tokens=512*20,
        final_tokens=200 * len(pretrain_dataset) * block_size,
        num_workers=0,
        writer=writer
    )
    # Update Trainer call to supply train_dataset and test_dataset (None is used for test_dataset)
    trainer_obj = Trainer(model, pretrain_dataset, None, tconf)
    ### END CODE HERE

    return tconf, trainer_obj

def train(model, writing_params_path, trainer_obj):
    ### TODO:
    ### - Given:
    ###     An output path writing_params_path for the model parameters
    ### [part d]:
    ###
    ### Note: trainer_obj is of type Trainer (see trainer.py for more details)

    ### START CODE HERE
    trainer_obj.train()
    torch.save(model.state_dict(), writing_params_path)
    ### END CODE HERE
    return
