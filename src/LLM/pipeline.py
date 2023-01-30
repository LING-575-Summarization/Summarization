from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, AutoConfig, DataCollatorForSeq2Seq, \
    Seq2SeqTrainer
import evaluate
from datasets import load_dataset, DatasetDict
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
        label = tokenizer(data["summary"], padding=True, truncation=True, max_length=kwargs["max_output_length"])
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
        max_length=kwargs["max_output_length"]
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(kwargs["checkpoint"], config=config)
    model.resize_token_embeddings(len(tokenizer.vocab))

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
        generation_max_length=kwargs["max_output_length"],
    )

    rouge_metric = evaluate.load("rouge")

    def tokenize_sentence(arg):
        encoded_arg = tokenizer(arg)
        return tokenizer.convert_ids_to_tokens(encoded_arg.input_ids)

    def get_pred_label(predictions, labels):
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        print(predictions)
        print(labels)

        text_predicitons = [" \n ".join(sent_tokenize(p.replace("<n>", " \n "))) for p in predictions]
        text_labels = [" \n ".join(sent_tokenize(l)) for l in labels]

        print(text_predicitons)
        print(text_labels)
        return text_predicitons, text_labels

    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        predictions, labels = get_pred_label(predictions, labels)
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

    # Start Evaluation
    output_dir_path = util.get_root_dir() + "outputs/D3"
    Path(output_dir_path).mkdir(parents=True, exist_ok=True)

    final_validation_predictions = trainer.predict(ds_for_train["validation"])
    validation_predictions, validation_labels, validation_metrics = final_validation_predictions

    predictions, labels = get_pred_label(validation_predictions, validation_labels)

    ids = eval_dataset["id"]

    for i in range(0, len(eval_dataset)):
        print("***** Summary Text (Gold Text) *****")
        print(labels[i])
        print("***** Summary Text (Generated Text) *****")
        print(predictions[i])

        with open(output_dir_path + "/{}-A.M.100.{}.3".format(ids[i][:-1], ids[i][-1]), "w") as output_file:
            output_file.write(predictions[i])


if __name__ == "__main__":
    kwargs = vars(util.get_args())
    kwargs["do_train"] = True
    pipeline(**kwargs)
