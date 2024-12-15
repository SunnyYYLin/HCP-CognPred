from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from data_process import HCPDataset
from config import *
from model.behav_pred import BehaviorPredictionModel
import torch
import torchmetrics
from torch.utils.data import DataLoader


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.tensor(predictions)
    labels = torch.tensor(labels).squeeze(1)  
    mse = (predictions - labels) ** 2
    abs_labels_squared = labels ** 2 + 1e-3
    adjusted_mse = mse / abs_labels_squared
    mean_adjusted_mse = torch.mean(adjusted_mse)
    
    return {'adjusted_mse': mean_adjusted_mse.item()}

# Load the dataset
print("Loading the dataset...")
dataset = HCPDataset()
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
print("Dataset loaded.")

# Load the model
config = FNNConfig()
model = BehaviorPredictionModel(config)

# Load the trainer
args = TrainingArguments(
    output_dir='checkpoints',
    num_train_epochs=1000,
    per_device_train_batch_size=100,
    per_device_eval_batch_size=100,
    logging_strategy='steps',
    eval_strategy='epoch',
    logging_dir='logs',
    logging_steps=20,
    save_strategy='epoch',
    save_steps=1,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='adjusted_mse',
    fp16=True,
    greater_is_better=False,
)
call_backs = [
    EarlyStoppingCallback(early_stopping_patience=10)
]

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,  
    eval_dataset=val_dataset,    
    data_collator=None, 
    callbacks=call_backs,
    compute_metrics=compute_metrics 
)
#checkpoint_path = 'checkpoints/checkpoint-404'
trainer.train()#(resume_from_checkpoint=checkpoint_path)