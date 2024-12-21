from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from data_process import HCPDataset
from config import *
from model.behav_pred import BehaviorPredictionModel
from torch.utils.data import random_split
from metrics import CognPredMetrics

# Set the config
backbone_config = FNNConfig(
    hidden_dims=[2048, 512, 128],
    dropout=0.1
)
config = PipelineConfig(
    pred_vars=['CardSort_Unadj', 'Flanker_Unadj'],
    backbone_config=backbone_config
    
)

# Load the dataset
print("Loading the dataset...")
dataset = HCPDataset(config)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

# Load the model
model = BehaviorPredictionModel(config)
print(model)

# Load the trainer
args = TrainingArguments(
    output_dir='checkpoints',
    num_train_epochs=1024,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    logging_strategy='steps',
    eval_strategy='epoch',
    eval_steps=1,
    logging_dir='logs',
    logging_steps=1,
    save_strategy='epoch',
    save_steps=1,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='mape',
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
    compute_metrics=CognPredMetrics(config)
)
trainer.train()