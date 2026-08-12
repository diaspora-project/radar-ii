# 📡 RADAR II

Code for the paper **"Toward Autonomous Radio Follow-up of Multi-messenger Transients with `RADAR`: Language-Model Benchmarking, Accelerated Inference, and Agentic Jansky VLA Scheduling"** (Hategan, O'Dwyer, Corsi, Huerta, Li, Foster et al.). 

`RADAR` (Radio Afterglow Detection and AI-driven Response) is a federated, privacy-enhancing framework for radio follow-up of gravitational-wave events, introduced in [Patel et al. (2025), ApJS 280, 71](https://ui.adsabs.harvard.edu/abs/2025ApJS..280...71P). This paper extends it in three directions:

1. **Accelerated federated inference**: network round-trips through the `Octopus` fabric, not `afterglowpy` evaluation, dominated the original 3–4 hr fit (~40× slower than local `emcee`). Server-side thread pools and site-side subprocess pools restore ≈120 samples s⁻¹, matching the local parallel baseline. The JAX surrogate `FIESTA` is ~5× faster per call but loses to `afterglowpy` under parallel sampling due to GIL contention.
2. **LLM benchmarking of the GCN parser**: GPT-5.5, Claude Opus 4.7, and Gemini 3.5 Flash against the human-curated GW170817 radio catalog, using the original prompt set. Best event-match F1 = 0.893 ± 0.002, a 16% gain over GPT-4.1.
3. **Agentic VLA scheduling**: an LLM agent converts natural-language observing requests into scheduling blocks (SBs) that upload cleanly to the NRAO OPT; a 21-criterion auto-grader scores them against manually written SBs (123/126 criteria passed across six SBs).

## Layout

| Path | Paper section | Contents |
|------|---------------|----------|
| [`gcn-ai-parser/`](gcn-ai-parser/) | §2 | `extract_gcn.py` runs the four-prompt extraction chain over GCN Circulars; `evaluate.py` computes P/R/F1 and GCN-level recall (mean ± std over 3 runs) for Table 1 |
| [`sb-auto-grader/`](sb-auto-grader/) | §4 | VLA SB grader — rules in `rubric.yaml`, engine in `src/score_sb.py`, full pipeline in `src/llm_judge.py`. See its [README](sb-auto-grader/README.md) |

```bash
# parse + score GCN circulars
cd gcn-ai-parser && python extract_gcn.py --model GPT-5.5 --output result/gpt55_run1.json && python evaluate.py

# grade example SBs, fully offline
cd sb-auto-grader && python src/llm_judge.py examples/*_AItest.txt --ground-auto --no-llm
```

