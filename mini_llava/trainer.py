from transformers import Trainer, TrainingArguments, TrainerCallback
from peft import LoraConfig, get_peft_model
import math

class MMTrainer(Trainer):
    def __init__(self, *args, custom_train_dataloader=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_train_dataloader = custom_train_dataloader
    def get_train_dataloader(self):
        return self.custom_train_dataloader

# 途中経過を強制的に表示するコールバック
class SimplePrinter(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs: return
        # 例: {'loss': 3.21, 'grad_norm': 0.9, 'learning_rate': 5e-5, 'epoch': 0.12, 'step': 10}
        step = int(state.global_step)
        maxs = state.max_steps if state.max_steps is not None else "?"
        print(f"[log] step {step}/{maxs} | logs: { {k: round(v,4) if isinstance(v,(int,float)) else v for k,v in logs.items()} }")

def train_mini_llava_2(model, tokenizer, train_dataloader, eval_dataloader=None, use_lora=False,
                     max_steps=200,  # ← スモークテスト用。ちゃんと回す時は増やす/外す
                     num_train_epochs=1,  # max_steps を優先。外すなら epoch 指定でもOK
                     learning_rate=5e-5):

    # Colabで重くなりがちなので軽量設定＆ログ多め
    training_args = TrainingArguments(
        output_dir="/content/Mini-LLaVA/results",
        # 進捗/保存/評価の頻度
        logging_strategy="steps",
        logging_steps=1,           # ← 1ステップごとにログ
        logging_first_step=True,    # ← 最初の1ステップ目から出す
        save_strategy="steps",
        save_steps=200,             # ← スモーク中は大きめでOK
        eval_steps=200,
        report_to="none",           # wandb等を使わない
        disable_tqdm=False,         # 進捗バーを出す

        # 学習ステップ制御
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,        # ← これが指定されていると epoch より優先される

        # 性能/安定化系
        learning_rate=learning_rate,
        warmup_steps=0,             # 小規模なので0推奨（500は重すぎ）
        weight_decay=0.01,
        gradient_accumulation_steps=1,

        # GPU/メモリ
        fp16=True,                  # T4 は fp16 可
        bf16=False,
        dataloader_num_workers=2,   # 画像読み込みを少し並列化
        dataloader_pin_memory=True,

        # これ超重要：カスタムdictをそのままTrainerに渡すために列削除を無効化
        remove_unused_columns=False,
    )

    # 参考：総ステップを見積もって事前に表示（安心用）
    steps_per_epoch = len(train_dataloader)
    est_total = max_steps if max_steps and max_steps > 0 else math.ceil(steps_per_epoch * num_train_epochs)
    print(f"[info] steps/epoch = {steps_per_epoch}, 予定総ステップ ≈ {est_total}")

    # LoRA（mm_projector だけLoRA化する設計のまま）
    if use_lora:
        lora_config = LoraConfig(
            r=16, lora_alpha=32, target_modules=["mm_projector"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

    trainer = MMTrainer(
        model=model,
        args=training_args,
        custom_train_dataloader=train_dataloader,
        tokenizer=tokenizer
    )
    trainer.add_callback(SimplePrinter())  # ← ログ出力を確実化

    # 学習
    trainer.train()

    # 保存
    trainer.save_model("/content/Mini-LLaVA/final_model")
    print("✅ Model saved to /content/Mini-LLaVA/final_model")
