from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, AutoConfig, \
    DataCollatorForSeq2Seq, \
    Seq2SeqTrainer
from torch.utils.data import DataLoader
from nltk.tokenize import sent_tokenize

import util


def pipeline(**kwargs):
    train_dataset = load_dataset('json', data_files=kwargs["data_json"], field="train", split="train")
    eval_dataset = load_dataset('json', data_files=kwargs["data_json"], field="validation", split="train")
    test_dataset = load_dataset('json', data_files=kwargs["data_json"], field="test", split="train")
    ds = DatasetDict({"train": train_dataset, "test": test_dataset, "validation": eval_dataset})

    tokenizer = AutoTokenizer.from_pretrained(kwargs["checkpoint"], max_length=1024, padding="max_length",
                                              truncation=True)

    def tokenize__data(data):
        input_feature = tokenizer(data["text"], padding=True, truncation=True, max_length=1024)
        label = tokenizer(data["summary"], padding=True, truncation=True, max_length=100)
        return {
            "input_ids": input_feature["input_ids"],
            "attention_mask": input_feature["attention_mask"],
            "labels": label["input_ids"],
        }

    ds_for_train = ds.map(
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
    model = AutoModelForSeq2SeqLM.from_pretrained(kwargs["checkpoint"], config=config)

    if not kwargs["do_train"]:
        model.load_state_dict(torch.load(kwargs["model_file"], map_location=torch.device(device)))

    model.to(device)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="max_length", max_length=1024,
                                           return_tensors="pt")

    training_args = Seq2SeqTrainingArguments(
        output_dir=kwargs["output_dir"],
        seed=kwargs["seed"],
        overwrite_output_dir=True,
        label_names=["labels"],
        learning_rate=kwargs["learning_rate"],
        num_train_epochs=kwargs["epoch"],
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        generation_max_length=100,
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

        return rouge_metric.compute(predictions=predictions, references=labels, tokenizer=tokenize_sentence)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=ds_for_train["train"],
        eval_dataset=ds_for_train["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    if kwargs["do_train"]:
        trainer.train()
        trainer.save_model()

    evaluation(ds_for_train, ds["validation"], data_collator, model, device, tokenizer)


def evaluation(ds_for_train, eval_dataset, data_collator, model, device, tokenizer):
    Path("outputs/D3").mkdir(parents=True, exist_ok=True)

    eval_dataloader = DataLoader(
        ds_for_train["validation"].with_format("torch"),
        collate_fn=data_collator,
        batch_size=len(eval_dataset)
    )
    i = 0
    for batch in eval_dataloader:
        with torch.no_grad():
            predictions = model.generate(
                batch["input_ids"].to(device),
                num_beams=15,
                num_return_sequences=1,
                no_repeat_ngram_size=1,
                remove_invalid_values=True,
                max_length=128,
            )
        labels = batch["labels"]

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    ids = eval_dataset["id"]

    for i in range(0, len(eval_dataset)):
        print("***** Summary Text (Gold Text) *****")
        print(labels[i])
        print("***** Summary Text (Generated Text) *****")
        print(predictions[i])

        raw_prediction = predictions[i]
        raw_prediction = sent_tokenize(raw_prediction)
        with open("outputs/D3/{}-A.M.100.{}.3".format(ids[i][-1], ids[i][:-1]), "w") as output_file:
            output_file.write("\n".join(map(str, raw_prediction)))


if __name__ == "__main__":
    kwargs = vars(util.get_args())
    kwargs["do_train"] = True
    pipeline(**kwargs)
