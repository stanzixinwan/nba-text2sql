# Presentation Q&A Prep

1. **Why execution accuracy over exact match?**  
Execution accuracy captures semantic equivalence under SQL rewrites; exact match is too strict for practical system quality.

2. **Why Applied track instead of Research track?**  
The core goal is an end-to-end runnable system with deployment-style evaluation and demo usability.

3. **Why Training focus?**  
The largest effort is on regime comparisons (full/LoRA/QLoRA), adaptation points, and ablations.

4. **Why Spider -> NBA transfer?**  
Spider gives broad SQL supervision; NBA tests realistic domain shift and schema specificity.

5. **Why choose T5-base and CodeT5+ 220M?**  
They are BERT-scale+ seq2seq models feasible on a single GPU while still strong enough for meaningful ablations.

6. **How do you prevent train/test leakage on NBA?**  
A deterministic 150/50 split is stored in `data/nba/nba_split.json`; all final comparisons use `--split test`.

7. **Why does RAG underperform oracle mode?**  
Retriever misses key tables on some questions; generation then lacks required schema context.

8. **What is the strongest empirical result?**  
CodeT5+ LoRA with full NBA adaptation reaches the best held-out execution accuracy in current runs.

9. **What are the major failure categories?**  
Schema linking, structural decoding, value grounding, and aggregation errors.

10. **What compute constraints shaped design decisions?**  
Single-GPU budget favored PEFT methods and staged adaptation instead of large full fine-tuning sweeps.

11. **How reproducible are results?**  
Run configs are saved per checkpoint; command templates are scripted; report tables are auto-generated from eval JSONs.

12. **What is the next most impactful improvement?**  
Improve retrieval quality (hybrid retrieval/re-ranking) before scaling model size, because schema misses dominate.

13. **How would you make this publishable?**  
Add multi-seed confidence intervals, larger benchmark coverage, and stronger novelty in retrieval-conditioned generation.

14. **Why include both oracle and RAG settings?**  
Oracle provides an upper bound for retrieval-conditioned prompting and isolates retrieval vs generation errors.
