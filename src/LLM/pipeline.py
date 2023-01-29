import numpy as np
import torch
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, AutoConfig, DataCollatorForSeq2Seq, \
    Seq2SeqTrainer
import evaluate
from datasets import load_dataset, DatasetDict, concatenate_datasets

import util


def pipeline(**kwargs):
    train_dataset = load_dataset('json', data_files=kwargs["data_json"], field="train", split="train")
    eval_dataset = load_dataset('json', data_files=kwargs["data_json"], field="validation", split="train")
    test_dataset = load_dataset('json', data_files=kwargs["data_json"], field="test", split="train")
    ds = DatasetDict({"train": train_dataset, "test": test_dataset, "validation": eval_dataset})

    tokenizer = AutoTokenizer.from_pretrained(kwargs["checkpoint"], max_length=1024, padding="max_length",
                                              truncation=True)

    def tokenize__data(data):
        # Max token size is 14536 and 215 for inputs and labels, respectively.
        # Here I restrict these token size.
        input_feature = tokenizer(data["text"], truncation=True, max_length=1024)
        label = tokenizer(data["summary"], truncation=True, max_length=100)
        return {
            "input_ids": input_feature["input_ids"],
            "attention_mask": input_feature["attention_mask"],
            "labels": label["input_ids"],
        }

    tokenizer.add_tokens(['[MASK]'], special_tokens=True)

    ds = ds.map(
        tokenize__data,
        remove_columns=["id", "summary", "text"],
        batched=True,
        batch_size=kwargs["batch_size"])

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    config = AutoConfig.from_pretrained(
        kwargs["checkpoint"],
        max_length=100
    )
    model = AutoModelForCausalLM.from_pretrained(kwargs["checkpoint"], config=config)

    if not kwargs["do_train"]:
        model.load_state_dict(torch.load(kwargs["model_file"], map_location=torch.device(device)))

    model.to(device)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")

    training_args = TrainingArguments(
        output_dir=kwargs["train_dir"],
        seed=kwargs["seed"],
        overwrite_output_dir=True,
        label_names=["labels"],
        learning_rate=kwargs["learning_rate"],
        num_train_epochs=kwargs["epoch"],
        per_device_train_batch_size=kwargs["batch_size"],
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1"
    )

    rouge_metric = evaluate.load("rouge")

    def tokenize_sentence(arg):
        encoded_arg = tokenizer(arg)
        return tokenizer.convert_ids_to_tokens(encoded_arg.input_ids)

    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        predictions = ["\n".join(np.char.strip(prediction)) for prediction in predictions]
        labels = ["\n".join(np.char.strip(label)) for label in labels]

        return rouge_metric.compute(predictions=predictions, references=labels, tokenizer=tokenize_sentence)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collactor=data_collator,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    if kwargs["do_train"]:
        trainer.train()

    trainer.save_model()


if __name__ == "__main__":
    kwargs = vars(util.get_args())
    kwargs["do_train"] = True
    pipeline(**kwargs)
