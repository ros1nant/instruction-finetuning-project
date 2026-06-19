LoRA adapter
============

Эта папка содержит сохраненный LoRA-адаптер после fine-tuning.

Файлы:

`adapter_model.safetensors` - веса LoRA-адаптера.

`adapter_config.json` - настройки LoRA: `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `task_type=CAUSAL_LM`.

`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` - tokenizer и chat template, сохраненные вместе с адаптером.

Важное замечание: основной notebook называется `mistral_qlora_finetuning.ipynb` и внутри настроен под Qwen-3B, но сохраненный tokenizer и chat template в этой папке выглядят как Qwen (`Qwen2Tokenizer` и Qwen chat template). Поэтому этот адаптер нужно загружать только с той базовой моделью, на которой он реально обучался. Если базовая модель не совпадает, адаптер может не загрузиться или будет работать некорректно.
