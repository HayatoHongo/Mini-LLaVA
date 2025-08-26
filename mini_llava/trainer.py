from transformers import Trainer, TrainingArguments, TrainerCallback
import math

class MMTrainer(Trainer):
    def __init__(self, *args, custom_train_dataloader=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_train_dataloader = custom_train_dataloader
    def get_train_dataloader(self):
        return self.custom_train_dataloader

class SimplePrinter(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = int(state.global_step)
            mx = state.max_steps or "?"
            short = {k:(round(v,4) if isinstance(v,(int,float)) else v) for k,v in logs.items()}
            print(f"[log] step {step}/{mx} | {short}", flush=True)

def train_safe(model, tokenizer, train_dataloader, max_steps=10):
    args = TrainingArguments(
        output_dir="/content/Mini-LLaVA/results",
        logging_strategy="steps",
        logging_steps=1,                 # ← 毎ステップ出す
        logging_first_step=True,
        report_to="none",
        disable_tqdm=True,               # ← TQDMを完全停止（stdout干渉防止）
        num_train_epochs=1,
        max_steps=max_steps,
        learning_rate=5e-5,
        warmup_steps=0,
        weight_decay=0.0,
        gradient_accumulation_steps=1,
        fp16=False,                      # ← まずは安全にOFF（通ったらONにしてOK）
        dataloader_num_workers=0,        
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    print(f"[info] steps/epoch={len(train_dataloader)}, run steps={max_steps}")
    tr = MMTrainer(model=model, args=args, tokenizer=tokenizer, custom_train_dataloader=train_dataloader)
    tr.add_callback(SimplePrinter())
    tr.train()
    print("done.")
    return tr