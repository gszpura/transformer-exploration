# transformer-exploration

Portfolio repo for a 6-month prep plan targeting Mistral AI research engineering roles. Everything is built from scratch in PyTorch — no `nn.Transformer`, no `F.scaled_dot_product_attention`, no `transformers` lib until an exercise is complete, then compared against.

## Structure

- `exercises/` — numbered exercises, one ≈ one sitting (2–5 h)
- `notes/` — derivations, profiling reports, run reports, retros

## Roadmap

1. **Phase 1 (weeks 1–6)** — autograd engine, training pipelines, BPE tokenizer, MHA/GQA/MQA, GPT assembly, 10M-param training run. Milestone: MHA from memory in <30 min.
2. **Phase 2 (weeks 7–12)** — LLM theory + Mistral papers (Mistral 7B, Mixtral, Magistral).
3. **Phase 3 (weeks 13–18)** — LoRA fine-tune + mini eval harness + vLLM deploy (portfolio project).
4. **Phase 4 (weeks 19–22)** — distributed training + serving internals.
5. **Phase 5 (weeks 23–26)** — interview campaign.

## Setup

```bash
uv sync
uv run python -c "import torch; print(torch.__version__)"
```

Requires a reboot after NVIDIA driver updates (see notes) or falls back to CPU/Colab.
