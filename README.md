# Q-resafe AWQ calibration-data experiment

## Consolidated report

The complete five-page report with the recorded ASR/PPL tables, token-matched confirmation, calibration-token accounting, leakage check, Modal L40S environment, limitations, and reproducibility identifiers is available at [`output/pdf/qresafe_awq_calibration_report.pdf`](output/pdf/qresafe_awq_calibration_report.pdf). The report source is [`reports/build_awq_report.py`](reports/build_awq_report.py).

This project tests one question only:

> With the same safety-aligned Llama-2-7B-Chat model and the official Q-resafe AWQ implementation, does replacing the calibration corpus with Risk-I, reconstructed Risk-II, or Risk-III change held-out AdvBench ASR without fine-tuning?

This is an extension of the Q-resafe calibration corpus to AWQ. It is not a reproduction of Q-resafe's AWQ Risk-I/Risk-II/Risk-III experiments: the paper did not run those AWQ conditions.

## Scope locked for the first run

- Model: `meta-llama/Llama-2-7b-chat-hf`
- Official implementation: `Qresafe/quant-without-ft`
- Q-resafe commit is written to every quantization metadata file.
- AWQ: INT4, `zero_point=True`, `q_group_size=128`, `version=GEMM`
- Calibration: Pile uses the unmodified Q-resafe defaults `max_calib_samples=128`, `max_calib_seq_len=512`, seed 42. The three ten-example Risk corpora use a common `max_calib_seq_len=128` and are token-chunked before being passed to the official loader; this preserves all rendered tokens without duplicating Risk-III examples.
- Conditions: FP16, AWQ-Pile, AWQ-R1, AWQ-R2, AWQ-R3
- Evaluation: the same held-out AdvBench file, greedy decoding, the same chat template and seed
- Utility: WikiText-2 raw test perplexity
- No AQLM, GPTQ, QLoRA, QAT, vaccine, optimizer, backward pass, or safety fine-tuning

Risk-II is explicitly reconstructed from the public description. It is not the exact unreleased Q-resafe Risk-II data. The code records this in `data/data_indices.json` and every Risk-II record.

The reconstruction uses the public `AlignmentResearch/AdvBench` export, which has 520 rows with an `instructions` field and a paired `proxy_gen_target` field. This gives us a deterministic instruction/response pair without fabricating responses. It is still a reconstruction: the exact ten rows selected by Q-resafe are not published.

## L40S environment

Run this on the L40S host inside the official checkout:

```bash
git clone https://github.com/Thecommonirin/Qresafe.git
cd Qresafe/quant-without-ft

conda create -n qresafe python=3.10 -y
conda activate qresafe
pip3 install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu118
pip install tabulate protobuf evaluate scipy transformers lm_eval datasets pytest
```

Use the pinned versions in `Qresafe/requirements.txt` if imports conflict. Do not upgrade the whole environment without recording the change. Log the actual environment after installation:

```bash
cd /path/to/qresafe_awq_calibration
python scripts/write_environment.py --qresafe-root /path/to/Qresafe/quant-without-ft
```

The L40S run needs Hugging Face access to Llama-2:

```bash
huggingface-cli login
```

## Run on Modal L40S

`modal_app.py` builds a Python 3.10/CUDA 11.8 image, mounts a persistent Hugging Face cache and artifact Volume, and pins the GPU to `L40S`. The public smoke test does not need a secret:

```powershell
$env:PYTHONIOENCODING = "utf-8"
modal run --timestamps --name qresafe-awq-l40s-smoke modal_app.py::main --mode smoke
```

For the gated Llama-2 full run, create a Modal secret without putting the token in this repository or in logs:

```powershell
modal secret create huggingface HF_TOKEN=<your-huggingface-token>
modal run --timestamps --name qresafe-awq-l40s-full modal_app.py::main --mode full
```

The `full` entrypoint first runs a remote preparation stage using the Modal secret, reconstructs Risk-III, and stores the data in the persistent `qresafe-awq-artifacts` Volume. It then runs FP16, Pile, Risk-I, Risk-II, and Risk-III sequentially on one L40S. The current workspace smoke evidence is recorded under `remote_artifacts/` and is not a scientific result for Llama-2.

## Prepare calibration data

Risk-I defaults to `HuggingFaceH4/ultrachat_200k`, and Risk-III defaults to `AlignmentResearch/AdvBench` with response column `proxy_gen_target`. The script samples ten rows with seed 42, writes IDs/metadata to `data/data_indices.json`, and creates a held-out AdvBench file with the ten Risk-III rows removed.

The local preparation script also accepts a user-supplied JSONL response file when another AdvBench export is used. Each line should contain either `source_index` or `prompt`, plus the configured response field:

```json
{"source_index": 17, "response": "<corresponding response from your approved source>"}
```

Then run:

```bash
python scripts/prepare_calib.py \
  --output-dir data \
  --risk3-response-file /secure/path/advbench_responses.jsonl
```

The output contains:

```text
data/
├── risk1.jsonl
├── risk2.jsonl
├── risk3.jsonl
├── advbench_eval.jsonl
└── data_indices.json
```

Do not print or paste harmful examples into the report. Use `source_index` and `sample_id` for provenance.

## Reproduce the default Pile control first

The Pile condition deliberately omits `calib_data` so the official Q-resafe default path resolves `calib_data="pileval"` internally:

```bash
python scripts/quantize_pile.py \
  --qresafe-root /path/to/Qresafe/quant-without-ft \
  --device-map auto
```

This writes `outputs/models/awq_pile/` and `quantization_metadata.json`.

## Run the three custom calibration conditions

```bash
python scripts/quantize_risk1.py --qresafe-root /path/to/Qresafe/quant-without-ft --device-map auto
python scripts/quantize_risk2.py --qresafe-root /path/to/Qresafe/quant-without-ft --device-map auto
python scripts/quantize_risk3.py --qresafe-root /path/to/Qresafe/quant-without-ft --device-map auto
```

Each custom condition is derived from exactly ten rendered user/assistant conversations and passes their tokenized stream to the unmodified Q-resafe AWQ implementation. Calibration metadata records source IDs, accepted token count, discarded examples, mean tokens/example, and full-block count.

The 10-example Risk-II/Risk-III streams are shorter than one 512-token block, while some Risk-I conversations are longer than 128 tokens. Therefore the extension renders each pair once, tokenizes it with the Llama-2 tokenizer, splits the token stream into non-overlapping chunks of at most 128 tokens, and passes those tokenized chunks to Q-resafe's existing `List[List[int]]` calibration path. No example or harmful response is duplicated. The metadata records the effective block length, source-example count, total tokens, chunk count, and block count. The resulting Risk-I/Risk-II/Risk-III AWQ models must all use the same 128-token block length.

The Modal runner persists a quantized model immediately after quantization, appends every safety response to its JSONL checkpoint, commits the Volume every ten new prompts, and resumes from existing condition/model/checkpoint artifacts. A crash can therefore re-run at most the most recent uncommitted safety batch rather than restarting the experiment.

## Token-matched confirmation

The first run has a large calibration-token imbalance (Risk-I is much longer
than Risk-II/Risk-III). The confirmation protocol is kept in a separate
`/root/artifacts/token_matched/` namespace so it cannot overwrite the first
run. It reruns only Risk-I/Risk-II/Risk-III with exactly 384 rendered tokens,
which becomes three complete 128-token Q-resafe blocks for every condition.
The token stream is the deterministic prefix of the ten rendered records in
locked order; no record or harmful example is duplicated. The metadata records
the original token total and the discarded-token count. FP16 and Pile are
unchanged controls from the first run.

Run the confirmation on Modal L40S with:

```powershell
$env:PYTHONIOENCODING = "utf-8"
modal run --detach --timestamps --name qresafe-awq-l40s-tokenmatched modal_app.py::main --mode token_matched
```

The confirmation writes `token_matched/results.csv` with the three Risk rows.
It must be interpreted as a sensitivity check: because 384 tokens cannot
contain all ten full conversations, a positive result should be followed by a
pre-registered protocol that matches both token budget and source coverage.

## Evaluate FP16 and all AWQ outputs

First evaluate the base model:

```bash
python scripts/eval_safety.py --condition fp16 --eval-file data/advbench_eval.jsonl
python scripts/eval_utility.py --condition fp16
```

Then evaluate each quantized output:

```bash
for c in pile risk1 risk2 risk3; do
  python scripts/eval_safety.py \
    --condition "$c" \
    --qresafe-root /path/to/Qresafe/quant-without-ft \
    --eval-file data/advbench_eval.jsonl
  python scripts/eval_utility.py \
    --condition "$c" \
    --qresafe-root /path/to/Qresafe/quant-without-ft
done
```

The fixed generation configuration is in `generation_config.json`: greedy decoding with `max_new_tokens=128`, chat-template generation prompt enabled, seed 42. The first-pass ASR implementation is a transparent refusal heuristic and is reported as `asr_heldout_refusal_heuristic`; it should be replaced or validated with the lab's approved judge/manual protocol before making a publication-level claim.

Finally combine the results:

```bash
python scripts/collect_results.py
```

The requested table is written to `results.csv`:

```text
Model,Bits,Calibration,Samples,Tokens,Blocks,ASR held-out,PPL
```

## Tests and cheap verification

The local test suite does not download Llama-2, AdvBench, or WikiText and does not require a GPU. It checks the locked AWQ setup, chat rendering, Q-resafe block accounting, Risk-III leakage split, Risk-II traceability, and deterministic ASR accounting:

```bash
py -3.12 -m pytest -q
```

Before a full L40S run, use a small model or a local fixture only to verify imports and generation plumbing. A green unit-test suite is not evidence that the scientific phenomenon exists; the highest evidence here is the completed held-out ASR plus utility comparison under the locked conditions.

## Provenance

The Q-resafe checkout is kept beside this project at `../Qresafe/quant-without-ft` in the development workspace. Do not edit its AWQ implementation for the main experiment. If an upstream patch becomes necessary, record the patch and commit hash in the metadata rather than silently modifying the reference code.
