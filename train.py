from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from data_process import HCPDataset
from config import *
from model.behav_pred import BehaviorPredictionModel
from torch.utils.data import random_split
from metrics import CognPredMetrics

# Set the config
backbone_config = GCNConfig(
    hidden_dims=[128, 64],
    dropout=0.0
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
output_dir = Path(__file__).parent / 'checkpoints' / config.abbrev()
logging_dir = Path(__file__).parent / 'logs' / config.abbrev()
args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1024,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    logging_strategy='steps',
    eval_strategy='epoch',
    eval_steps=1,
    logging_dir=logging_dir,
    logging_steps=1,
    save_strategy='epoch',
    save_steps=8,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model='r2',
    max_grad_norm=1e4,
    fp16=True,
    use_cpu=False,
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