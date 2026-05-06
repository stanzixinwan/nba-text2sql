# Spider Eval Status (CUDA, max-examples=200)

## Completed

- `models/full_t5-base/final` -> `eval/full_t5-base_spider.json`
  - Execution accuracy: `0/200 = 0.0%`
  - Exact match: `53/200 = 26.5%`

- `models/lora_t5-base_r16/final` -> `eval/lora_t5-base_r16_spider.json`
  - Execution accuracy: `0/200 = 0.0%`
  - Exact match: `24/200 = 12.0%`

- `t5-base` zero-shot baseline -> `eval/baseline_spider_zeroshot_t5-base.json`
  - Execution accuracy: `0/200 = 0.0%`
  - Exact match: `0/200 = 0.0%`

- `google/flan-t5-base` zero-shot baseline -> `eval/baseline_spider_zeroshot_flan-t5-base.json`
  - Execution accuracy: `0/200 = 0.0%`
  - Exact match: `0/200 = 0.0%`

## Blocked

- `models/lora_codet5p-220m_r16/final` (with `--base-model Salesforce/codet5p-220m`)
  - Fails at tokenizer load in `transformers`:
    - `TypeError: Input must be a List[Union[str, AddedToken]]`
  - Not a missing-checkpoint issue; this is a tokenizer compatibility/runtime issue.

## Missing checkpoint

- `models/qlora_t5-base/...` is not present locally.
