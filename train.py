# pip install datasets peft bitsandbytes accelerate wandb "git+https://github.com/huggingface/transformers" autoawq lightning nltk loguru
import os
import re

import lightning as L
import numpy as np
import torch
from loguru import logger
from datasets import concatenate_datasets
from datasets import load_dataset, load_from_disk
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from nltk import edit_distance
from peft import LoraConfig
from peft import prepare_model_for_kbit_training, get_peft_model
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoProcessor
from transformers import BitsAndBytesConfig, Idefics2ForConditionalGeneration
try:
    from deepspeed.ops.adam import FusedAdam
except ImportError:
    from torch.optim import Adam as FusedAdam


def convert_key_info_to_qa(records: dict) -> dict:
    """
    Convert key information into QA format.

    Parameters
    ----------
    records : dict
        The records from the dataset.

    Returns
    -------
    dict
        The QA format of the records.
    """
    key2question = {
        "agreement_date": "When is the signing date of this agreement?",
        "effective_date": "When is the effective date of the contract?",
        "expiration_date": "When is the service end date or expiration date of the contract?",
        "party_address": "What is the address of the party to the contract?",
        "party_name": "What are the names of the contracting party?",
        "counterparty_address": "What is the address of the counterparty to the contract?",
        "counterparty_name": "What are the names of the contracting counterparty?",
        "counterparty_signer_name": "What is the name of the counterparty signer for each party to the agreement?",
        "counterparty_signer_title": "What is the counterparty signer’s title?",
        "auto_renewal": "Whether the contract term automatically renews (true/false).",
        "governing_law": "Where is the jurisdiction or choice of law?",
        "venue": "where is the location of the courts where legal proceedings will take place?",
        "payment_frequency": "what is the cadence for which payments are made (e.g., monthly, annually, one-time)?",
        "payment_term": "When an invoice is due after issuance (e.g. Net 30)?",
        "renewal_term": "What is the length of time the renewal period will last (e.g., 1 year, 2 years, 24 months etc.)?",
        "agreement_term": "What is the term of the contract as an amount of time (e.g., 24 months)?",
        "termination_for_cause": "Whether one or all parties may terminate the contract with cause, such as a breach of contract (true/false).",
        "termination_for_convenience": "Whether one or all parties may terminate the contract without cause, or at their convenience (true/false).",
        "termination_notice_period": "What is the period by which notice of termination must be given (e.g., 30 days)?",
        "opt_out_length": "What is the required notice period to NOT renew (e.g., 30 days)?",
        "contract_value": "What is the total fixed fee amount including currency codes or symbols?",
    }
    images = records["images"][0]
    questions, answers = [], []
    for key in key2question:
        answer = records[key][0]
        if answer != "N/A":
            questions.append(key2question[key])
            answers.append(answer)

    output = {
        "images": [images for _ in questions],
        "answer": answers,
        "question": questions
    }
    return output


class Idefics2Dataset(Dataset):
    """
    PyTorch Dataset for Idefics2. This class takes a HuggingFace Dataset as input.
    """

    def __init__(self, ds, max_page):
        super().__init__()
        self.dataset = ds
        self.max_page = max_page
        assert max_page > 0

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple:
        sample = self.dataset[idx]

        if len(sample["images"]) > self.max_page:
            images = sample["images"][:self.max_page // 2] + sample["images"][-self.max_page // 2:]
        else:
            images = sample["images"]

        question = sample["question"]
        answer = sample["answer"]

        return images, question, answer


def train_collate_fn(examples):
    global processor
    global image_token_id

    texts = []
    images = []
    for example in examples:
        images_example, question, answer = example
        content = [{"type": "image"} for _ in range(len(images_example))]
        content += [{"type": "text", "text": question}]

        # Create inputs
        messages = [
            {
                "role": "user",
                "content": content,
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ]
            },
        ]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=False)
        texts.append(prompt)
        images.append(images_example)

    batch = processor(text=texts, images=images, padding=True, truncation=True, max_length=MAX_LENGTH,
                      return_tensors="pt")

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    batch["labels"] = labels

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    pixel_attention_mask = batch["pixel_attention_mask"]
    labels = batch["labels"].long()

    return input_ids, attention_mask, pixel_values, pixel_attention_mask, labels


def eval_collate_fn(examples):
    images = []
    texts = []
    answers = []
    for example in examples:
        images_example, question, answer = example

        content = [{"type": "image"} for _ in range(len(images_example))]
        content += [{"type": "text", "text": question}]

        messages = [
            {
                "role": "user",
                "content": content,
            },
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        images.append(images_example)
        texts.append(text.strip())
        answers.append(answer)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    pixel_attention_mask = batch["pixel_attention_mask"]

    return input_ids, attention_mask, pixel_values, pixel_attention_mask, answers


class Idefics2ModelPLModule(L.LightningModule):
    def __init__(self, model_config, inp_processor, torch_model):
        super().__init__()
        self.config = model_config
        self.processor = inp_processor
        self.model = torch_model

        self.batch_size = config.get("batch_size")

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, pixel_attention_mask, labels = batch
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             pixel_values=pixel_values,
                             pixel_attention_mask=pixel_attention_mask,
                             labels=labels
                             )
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        input_ids, attention_mask, pixel_values, pixel_attention_mask, answers = batch

        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            max_new_tokens=MAX_LENGTH
        )
        # turn them back into text, chopping of the prompt
        # important: we don't skip special tokens here, because we want to see them in the output
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)

        scores = []
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"((?<=>) | (?=</s_))", "", pred)
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if self.config.get("verbose", False) and len(scores) == 1:
                logger.debug(f"""
Prediction: `{pred}`
Answer: `{answer}`
Normed ED: `{scores[0]:.3f}`""")

        self.log("val_edit_distance", np.mean(scores))

        return scores

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = FusedAdam(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self):
        return DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.batch_size, shuffle=True,
                          num_workers=6)

    def val_dataloader(self):
        return DataLoader(validation_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size, shuffle=False,
                          num_workers=6)


class PushToHubCallback(Callback):
    def on_train_epoch_end(self, pl_trainer, pl_module):
        logger.info(f"Pushing model to the hub, epoch {pl_trainer.current_epoch}")
        pl_module.model.push_to_hub(
            FINETUNED_REPO_ID,
            commit_message=f"Training in progress, epoch {pl_trainer.current_epoch}"
        )

    def on_train_end(self, pl_trainer, pl_module):
        logger.info("Pushing model to the hub after training")
        pl_module.processor.push_to_hub(FINETUNED_REPO_ID, commit_message="Training done")
        pl_module.model.push_to_hub(FINETUNED_REPO_ID, commit_message="Training done")


if __name__ == "__main__":

    MAX_LENGTH = 1024
    USE_LORA = False
    USE_QLORA = True
    MAX_PAGE = 5
    FINETUNED_REPO_ID = "chenghao/idefics2-edgar"
    WANDB_PROJECT = "Idefics2-EDGAR"
    WANDB_NAME = "demo-run"
    config = {
        "max_epochs": 10,
        # "val_check_interval": 0.2,
        "check_val_every_n_epoch": 1,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 12,
        "lr": 1e-4,
        "batch_size": 2,
        "precision": "16-mixed",
        "seed": 42,
        "warmup_steps": 50,
        "result_path": "./result",
        "verbose": True,
    }

    wandb_logger = WandbLogger(project=WANDB_PROJECT, name=WANDB_NAME)

    processor = AutoProcessor.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        do_image_splitting=False,
        size={"longest_edge": 490, "shortest_edge": 350}
    )
    mean = processor.image_processor.image_mean
    std = processor.image_processor.image_std

    if os.path.exists("local-dataset"):
        dataset = load_from_disk("local-dataset")
    else:
        dude_dataset = load_dataset("jordyvl/DUDE_subset_100val")
        edgar_dataset = load_dataset("chenghao/sec-material-contracts-qa")
        flattened_edgar_dataset = edgar_dataset['train'].map(
            convert_key_info_to_qa, batched=True, batch_size=1,
            remove_columns=edgar_dataset['train'].column_names, num_proc=10)
        # flattened_edgar_dataset = flattened_edgar_dataset.filter(lambda x: len(x['images']) <= MAX_PAGE, num_proc=10)
        dude_dataset = dude_dataset.remove_columns(["questionId"])
        flattened_edgar_dataset = flattened_edgar_dataset.cast(dude_dataset['train'].features)
        all_dataset = concatenate_datasets([dude_dataset['train'], flattened_edgar_dataset])
        dataset = all_dataset.train_test_split(test_size=0.2)

    train_dataset = Idefics2Dataset(dataset["train"], max_page=MAX_PAGE)
    validation_dataset = Idefics2Dataset(dataset["test"], max_page=MAX_PAGE)

    if USE_QLORA or USE_LORA:
        if USE_QLORA:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        else:
            quantization_config = None

        model = Idefics2ForConditionalGeneration.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
        )
    else:
        # for full fine-tuning, we can speed up the model using Flash Attention
        # only available on certain devices, see
        # https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
        model = Idefics2ForConditionalGeneration.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2",
        )

    if USE_QLORA or USE_LORA:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=".*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$",
            init_lora_weights="gaussian",
        )

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)

    image_token_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<image>")]

    model_module = Idefics2ModelPLModule(config, processor, model)

    early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=3, verbose=False, mode="min")
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        strategy="deepspeed_stage_2",
        max_epochs=config.get("max_epochs"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        accumulate_grad_batches=config.get("accumulate_grad_batches"),
        precision=config.get("precision"),
        num_sanity_val_steps=10,
        logger=wandb_logger,
        callbacks=[PushToHubCallback(), early_stop_callback],
    )

    trainer.fit(model_module)

