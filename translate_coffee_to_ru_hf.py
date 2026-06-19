import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


INPUT_PATH = Path("coffee_finetuning_20240914_witi_total.jsonl")
OUTPUT_PATH = Path("coffee_finetuning_20240914_witi_total_ru.jsonl")
PROGRESS_PATH = Path("translation_progress.json")
MODEL_NAME = "Helsinki-NLP/opus-mt-ko-ru"
BATCH_SIZE = 64
TEXT_FIELDS = ("instruction", "response")


def write_progress(done_texts: int, total_texts: int, done_records: int, total_records: int, status: str) -> None:
    payload = {
        "status": status,
        "done_texts": done_texts,
        "total_texts": total_texts,
        "done_records": done_records,
        "total_records": total_records,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    PROGRESS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def translate_batch(texts, tokenizer, model, device):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=192)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def main() -> None:
    records = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    total_records = len(records)
    total_texts = total_records * len(TEXT_FIELDS)
    write_progress(0, total_texts, 0, total_records, "loading_model")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    translated_records = [dict(record) for record in records]
    jobs = [(idx, field, records[idx][field]) for idx in range(total_records) for field in TEXT_FIELDS]

    done_texts = 0
    write_progress(done_texts, total_texts, 0, total_records, "translating")
    started_at = time.time()
    for offset in range(0, len(jobs), BATCH_SIZE):
        batch = jobs[offset : offset + BATCH_SIZE]
        translations = translate_batch([text for _, _, text in batch], tokenizer, model, device)
        for (record_idx, field, _), translated in zip(batch, translations):
            translated_records[record_idx][field] = translated
        done_texts += len(batch)
        done_records = min(done_texts // len(TEXT_FIELDS), total_records)
        write_progress(done_texts, total_texts, done_records, total_records, "translating")
        elapsed = max(time.time() - started_at, 0.001)
        print(f"{done_texts}/{total_texts} texts ({done_texts / total_texts:.1%}), {done_texts / elapsed:.2f} texts/s", flush=True)

    tmp_path = OUTPUT_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as out:
        for record in translated_records:
            out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp_path.replace(OUTPUT_PATH)
    write_progress(total_texts, total_texts, total_records, total_records, "done")
    print(f"wrote {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
