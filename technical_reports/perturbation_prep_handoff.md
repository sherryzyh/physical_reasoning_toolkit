# Perturbation Prep Handoff

## Purpose

This document is the running handoff for dataset-specific perturbation-prep work,
subset inference prep, OpenAI batch submission, and next-session follow-up.

Use one file for all datasets. Append a new dated section for each dataset/run
instead of creating dataset-specific handoff filenames.

## Shared Workflow

1. Save the target subset IDs to
   `uncertainty_quantification_physical_reasoning/perturbations/<dataset>/problem_ids_for_perturbation.json`.
2. Prepare the batch inference input from that ID set.
3. Submit the OpenAI batch job.
4. Record the batch ID, artifact paths, status, and any dataset-specific quirks.
5. In the follow-up session, fetch the completed batch output and parse it into
   per-problem inference files.

## Shared Paths

- Problem ID file:
  `uncertainty_quantification_physical_reasoning/perturbations/<dataset>/problem_ids_for_perturbation.json`
- Batch input/output directory:
  `uncertainty_quantification_physical_reasoning/batch_results/batch_inference/response_with_answer/<dataset>_<model>/`
- Parsed inference output directory:
  `uncertainty_quantification_physical_reasoning/experiment_results/inference/response_with_answer_tag/<dataset>_<model>/`

## Shared Commands

Prepare batch input:

```bash
.venv/bin/python uncertainty_quantification_physical_reasoning/scripts/script_physical_reasoning/response_with_answer/openai_batch_api/prepare_batch_inference.py \
  --dataset <dataset> \
  --model <model> \
  --problem-ids-file uncertainty_quantification_physical_reasoning/perturbations/<dataset>/problem_ids_for_perturbation.json
```

Submit batch:

```bash
.venv/bin/python uncertainty_quantification_physical_reasoning/scripts/script_physical_reasoning/response_with_answer/openai_batch_api/start_batch_inference.py \
  --input-file uncertainty_quantification_physical_reasoning/batch_results/batch_inference/response_with_answer/<dataset>_<model>/batch_input.jsonl
```

Fetch results after the OpenAI batch finishes:

```bash
.venv/bin/python uncertainty_quantification_physical_reasoning/scripts/script_physical_reasoning/response_with_answer/openai_batch_api/fetch_batch_results.py \
  -b <batch_id> \
  --batch-dir uncertainty_quantification_physical_reasoning/batch_results/batch_inference/response_with_answer/<dataset>_<model>
```

Parse fetched batch output into per-problem JSON files:

```bash
.venv/bin/python uncertainty_quantification_physical_reasoning/scripts/script_physical_reasoning/response_with_answer/openai_batch_api/parse_batch_results_with_answer.py \
  --batch-dir uncertainty_quantification_physical_reasoning/batch_results/batch_inference/response_with_answer/<dataset>_<model>
```

Basic count checks:

```bash
jq 'length' uncertainty_quantification_physical_reasoning/perturbations/<dataset>/problem_ids_for_perturbation.json
wc -l uncertainty_quantification_physical_reasoning/batch_results/batch_inference/response_with_answer/<dataset>_<model>/batch_input.jsonl
find uncertainty_quantification_physical_reasoning/experiment_results/inference/response_with_answer_tag/<dataset>_<model> -name '*.json' | wc -l
```

## Entry: 2026-04-14 / PhysBench / GPT-5.2 / Inference

### Goal

Run inference for the 200-problem `val` subset of `physbench` with `gpt-5.2`.

### Current Status

- The original single-batch submission failed at provider validation.
- Failure batch ID: `batch_69de8e54e7e4819092d6618871fd73ba`
- Failure reason: `maximum_input_file_size_exceeded`
- Reported limit from OpenAI for this model: `209715200` bytes
- Original batch input size:
  `uncertainty_quantification_physical_reasoning/batch_results/batch_inference/response_with_answer/physbench_gpt-5.2/batch_input.jsonl`
  = `500093631` bytes across `200` requests
- The failed batch metadata is retained at:
  `uncertainty_quantification_physical_reasoning/batch_results/batch_inference/response_with_answer/physbench_gpt-5.2/batch_metadata.json`
- The batch input was split into `3` shard directories under the same
  `response_with_answer/` root and all three shard batches were submitted.
- Current shard statuses when this handoff was updated: all `validating`

`shard 1/3`

- Batch ID: `batch_69dea7d1c8148190966cb2fde9db35aa`
- OpenAI input file ID: `file-Y6zvC1Tu8sLroZ9s2f5k6d`
- Batch directory:
  `uncertainty_quantification_physical_reasoning/batch_results/batch_inference/response_with_answer/physbench_gpt-5.2_shard-01`
- Requests: `61`
- Input size: `187918683` bytes

`shard 2/3`

- Batch ID: `batch_69dea7e495d08190887030bc952dc0ff`
- OpenAI input file ID: `file-23F6UC4jToxW7oE9n73oNX`
- Batch directory:
  `uncertainty_quantification_physical_reasoning/batch_results/batch_inference/response_with_answer/physbench_gpt-5.2_shard-02`
- Requests: `68`
- Input size: `186866447` bytes

`shard 3/3`

- Batch ID: `batch_69dea7f5568c81908b5b4d09bbdce215`
- OpenAI input file ID: `file-5TSNnGFKjpqB6JW9j4vzjb`
- Batch directory:
  `uncertainty_quantification_physical_reasoning/batch_results/batch_inference/response_with_answer/physbench_gpt-5.2_shard-03`
- Requests: `71`
- Input size: `125308501` bytes

### Important Artifacts

- Problem ID file:
  `uncertainty_quantification_physical_reasoning/perturbations/physbench/problem_ids_for_perturbation.json`
- Failed single-batch directory:
  `uncertainty_quantification_physical_reasoning/batch_results/batch_inference/response_with_answer/physbench_gpt-5.2/`
- Failed single-batch input:
  `uncertainty_quantification_physical_reasoning/batch_results/batch_inference/response_with_answer/physbench_gpt-5.2/batch_input.jsonl`
- Shard 1 directory:
  `uncertainty_quantification_physical_reasoning/batch_results/batch_inference/response_with_answer/physbench_gpt-5.2_shard-01/`
- Shard 2 directory:
  `uncertainty_quantification_physical_reasoning/batch_results/batch_inference/response_with_answer/physbench_gpt-5.2_shard-02/`
- Shard 3 directory:
  `uncertainty_quantification_physical_reasoning/batch_results/batch_inference/response_with_answer/physbench_gpt-5.2_shard-03/`
- Cached sampled video frames:
  `uncertainty_quantification_physical_reasoning/batch_results/batch_inference/response_with_answer/physbench_gpt-5.2/_video_frames/`
- External dataset cache used by PRKit:
  `/Users/yinghuan/PHYSICAL_REASONING_DATASETS/PhysBench`

### Relevant Local Code Behavior

- `physbench` uses `problem.question` directly instead of the generic question
  formatter, so multiple-choice options are not duplicated.
- `video_paths` are supported by decoding a small set of evenly spaced video
  frames and attaching those frames as `input_image` entries.
- Cached frames are written under the batch directory's `_video_frames/`
  subtree.
- The prompt text replaces `<video>` with a note telling the model that sampled
  frames are attached in chronological order.

### Environment Notes

- The local virtualenv needed:

```bash
.venv/bin/pip install imageio imageio-ffmpeg
```

- PhysBench media is installed in the external cache at:
  `/Users/yinghuan/PHYSICAL_REASONING_DATASETS/PhysBench`
- Because the `val` split includes many video-backed problems, the
  single-file batch input exceeded the OpenAI input-file limit for `gpt-5.2`.
- Follow-up inference submissions for PhysBench must be sharded into multiple
  smaller batch directories.

### Exact Commands Used So Far

Prepare the original PhysBench batch input from the saved ID set:

```bash
.venv/bin/python uncertainty_quantification_physical_reasoning/scripts/script_physical_reasoning/response_with_answer/openai_batch_api/prepare_batch_inference.py \
  --dataset physbench \
  --model gpt-5.2 \
  --problem-ids-file uncertainty_quantification_physical_reasoning/perturbations/physbench/problem_ids_for_perturbation.json
```

Submit the original PhysBench batch:

```bash
.venv/bin/python uncertainty_quantification_physical_reasoning/scripts/script_physical_reasoning/response_with_answer/openai_batch_api/start_batch_inference.py \
  --input-file uncertainty_quantification_physical_reasoning/batch_results/batch_inference/response_with_answer/physbench_gpt-5.2/batch_input.jsonl
```

Retrieve the failure details:

```bash
.venv/bin/python uncertainty_quantification_physical_reasoning/scripts/script_physical_reasoning/response_with_answer/openai_batch_api/fetch_batch_results.py \
  -b batch_69de8e54e7e4819092d6618871fd73ba \
  --batch-dir uncertainty_quantification_physical_reasoning/batch_results/batch_inference/response_with_answer/physbench_gpt-5.2
```

Split the oversized input into shard directories below the `209715200`-byte
limit and write `shard_info.json` beside each shard input:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import json

src = Path("uncertainty_quantification_physical_reasoning/batch_results/batch_inference/response_with_answer/physbench_gpt-5.2/batch_input.jsonl")
parent = src.parent.parent
limit = 180 * 1024 * 1024
prefix = "physbench_gpt-5.2_shard-"

lines = []
with src.open(encoding="utf-8") as f:
    for raw in f:
        if raw.strip():
            obj = json.loads(raw)
            lines.append((raw, len(raw.encode("utf-8")), obj.get("custom_id")))

shards = []
current = []
current_bytes = 0
for raw, nbytes, custom_id in lines:
    if current and current_bytes + nbytes > limit:
        shards.append((current, current_bytes))
        current = []
        current_bytes = 0
    current.append((raw, nbytes, custom_id))
    current_bytes += nbytes
if current:
    shards.append((current, current_bytes))

for idx, (entries, total_bytes) in enumerate(shards, start=1):
    shard_dir = parent / f"{prefix}{idx:02d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    with (shard_dir / "batch_input.jsonl").open("w", encoding="utf-8") as out:
        for raw, _, _ in entries:
            out.write(raw)
    with (shard_dir / "shard_info.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "physbench",
                "model": "gpt-5.2",
                "source_failed_batch_id": "batch_69de8e54e7e4819092d6618871fd73ba",
                "input_file_size_bytes": total_bytes,
                "request_count": len(entries),
                "first_custom_id": entries[0][2],
                "last_custom_id": entries[-1][2],
                "shard_index": idx,
                "shard_count": len(shards),
            },
            f,
            indent=2,
        )
PY
```

Submit the three inference shards with clean `dataset=model=physbench/gpt-5.2`
metadata:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path

from uncertainty_quantification_physical_reasoning.scripts.script_physical_reasoning.response_with_answer.openai_batch_api.start_batch_inference import start_batch_inference

base = Path("uncertainty_quantification_physical_reasoning/batch_results/batch_inference/response_with_answer")
for idx, name in enumerate(
    [
        "physbench_gpt-5.2_shard-01",
        "physbench_gpt-5.2_shard-02",
        "physbench_gpt-5.2_shard-03",
    ],
    start=1,
):
    start_batch_inference(
        base / name / "batch_input.jsonl",
        metadata={
            "dataset": "physbench",
            "model": "gpt-5.2",
            "task": "inference",
            "shard": f"{idx}/3",
            "source_failed_batch_id": "batch_69de8e54e7e4819092d6618871fd73ba",
        },
        dataset_name="physbench",
        model_name="gpt-5.2",
    )
PY
```

### Next Session Follow-Up

1. Fetch shard 1 when the batch is no longer `validating`, `in_progress`, or
   `finalizing`:

```bash
.venv/bin/python uncertainty_quantification_physical_reasoning/scripts/script_physical_reasoning/response_with_answer/openai_batch_api/fetch_batch_results.py \
  -b batch_69dea7d1c8148190966cb2fde9db35aa \
  --batch-dir uncertainty_quantification_physical_reasoning/batch_results/batch_inference/response_with_answer/physbench_gpt-5.2_shard-01
```

2. Fetch shard 2:

```bash
.venv/bin/python uncertainty_quantification_physical_reasoning/scripts/script_physical_reasoning/response_with_answer/openai_batch_api/fetch_batch_results.py \
  -b batch_69dea7e495d08190887030bc952dc0ff \
  --batch-dir uncertainty_quantification_physical_reasoning/batch_results/batch_inference/response_with_answer/physbench_gpt-5.2_shard-02
```

3. Fetch shard 3:

```bash
.venv/bin/python uncertainty_quantification_physical_reasoning/scripts/script_physical_reasoning/response_with_answer/openai_batch_api/fetch_batch_results.py \
  -b batch_69dea7f5568c81908b5b4d09bbdce215 \
  --batch-dir uncertainty_quantification_physical_reasoning/batch_results/batch_inference/response_with_answer/physbench_gpt-5.2_shard-03
```

4. Parse each completed shard into the shared inference output directory:

```bash
.venv/bin/python uncertainty_quantification_physical_reasoning/scripts/script_physical_reasoning/response_with_answer/openai_batch_api/parse_batch_results_with_answer.py \
  --batch-dir uncertainty_quantification_physical_reasoning/batch_results/batch_inference/response_with_answer/physbench_gpt-5.2_shard-01

.venv/bin/python uncertainty_quantification_physical_reasoning/scripts/script_physical_reasoning/response_with_answer/openai_batch_api/parse_batch_results_with_answer.py \
  --batch-dir uncertainty_quantification_physical_reasoning/batch_results/batch_inference/response_with_answer/physbench_gpt-5.2_shard-02

.venv/bin/python uncertainty_quantification_physical_reasoning/scripts/script_physical_reasoning/response_with_answer/openai_batch_api/parse_batch_results_with_answer.py \
  --batch-dir uncertainty_quantification_physical_reasoning/batch_results/batch_inference/response_with_answer/physbench_gpt-5.2_shard-03
```

5. Verify the parsed output count reaches `200` in:

```bash
uncertainty_quantification_physical_reasoning/experiment_results/inference/response_with_answer_tag/physbench_gpt-5.2/
```

6. Continue downstream perturbation or evaluation steps after all `200`
   inference files are present.

## Entry: 2026-04-14 / PhysBench / GPT-5.2 / Physpara + Semident

### Goal

Prepare and submit `physics_aware` (`physpara`) and `semantic_identical`
(`semident`) paraphrase batch jobs for the 200-problem `val` subset of
`physbench`.

### Current Status

- Prepared the PhysBench `physics_aware` batch input for `gpt-5.2`.
- Prepared the PhysBench `semantic_identical` batch input for `gpt-5.2`.
- Verified both batch inputs contain exactly `200` JSONL requests.
- Submitted both OpenAI batch jobs successfully.

`physics_aware` / `physpara`

- Batch ID: `batch_69dea76190c881909709451abb5e3080`
- Current status: `validating`
- OpenAI input file ID: `file-LMJS7wzVTMN5HmTfJB8Vmm`
- Batch metadata file:
  `uncertainty_quantification_physical_reasoning/batch_results/physics_aware_paraphrases/physbench/gpt-5.2/batch_metadata.json`

`semantic_identical` / `semident`

- Batch ID: `batch_69dea761426081908ef5f111659dbc4a`
- Current status: `validating`
- OpenAI input file ID: `file-EZ6ewW5zhVjUeD1g5qx31A`
- Batch metadata file:
  `uncertainty_quantification_physical_reasoning/batch_results/sementic_identical_paraphrases/physbench/gpt-5.2/batch_metadata.json`

### Important Artifacts

- Problem ID file:
  `uncertainty_quantification_physical_reasoning/perturbations/physbench/problem_ids_for_perturbation.json`
- Physpara batch root:
  `uncertainty_quantification_physical_reasoning/batch_results/physics_aware_paraphrases/physbench/`
- Physpara batch input:
  `uncertainty_quantification_physical_reasoning/batch_results/physics_aware_paraphrases/physbench/gpt-5.2/batch_input.jsonl`
- Semident batch root:
  `uncertainty_quantification_physical_reasoning/batch_results/sementic_identical_paraphrases/physbench/`
- Semident batch input:
  `uncertainty_quantification_physical_reasoning/batch_results/sementic_identical_paraphrases/physbench/gpt-5.2/batch_input.jsonl`
- Expected merged perturbation targets:
  `uncertainty_quantification_physical_reasoning/perturbations/physbench/physics_aware_paraphrases/`
  and
  `uncertainty_quantification_physical_reasoning/perturbations/physbench/semantic_identical_paraphrases/`

### Notes

- The semident batch-results root is spelled
  `sementic_identical_paraphrases` in the current code and filesystem.
  Preserve that exact spelling in commands and paths.
- The generic paraphrase batch tooling rewrites the question text only; it does
  not attach PhysBench media. This is acceptable for text-perturbation prep, but
  it is distinct from the multimodal inference path above.

### Exact Commands Used

Prepare the PhysBench physpara batch input:

```bash
.venv/bin/python uncertainty_quantification_physical_reasoning/scripts/script_perturbation_prep/generic_paraphrase_batch/prepare_generic_paraphrase_batch.py \
  --perturbation-type physics_aware \
  --dataset physbench \
  --models gpt-5.2 \
  --problem-ids-file uncertainty_quantification_physical_reasoning/perturbations/physbench/problem_ids_for_perturbation.json
```

Prepare the PhysBench semident batch input:

```bash
.venv/bin/python uncertainty_quantification_physical_reasoning/scripts/script_perturbation_prep/generic_paraphrase_batch/prepare_generic_paraphrase_batch.py \
  --perturbation-type semantic_identical \
  --dataset physbench \
  --models gpt-5.2 \
  --problem-ids-file uncertainty_quantification_physical_reasoning/perturbations/physbench/problem_ids_for_perturbation.json
```

Submit the PhysBench physpara batch:

```bash
.venv/bin/python uncertainty_quantification_physical_reasoning/scripts/script_perturbation_prep/generic_paraphrase_batch/submit_generic_paraphrase_batch.py \
  --perturbation-type physics_aware \
  --dataset physbench \
  --models gpt-5.2 \
  --api openai
```

Submit the PhysBench semident batch:

```bash
.venv/bin/python uncertainty_quantification_physical_reasoning/scripts/script_perturbation_prep/generic_paraphrase_batch/submit_generic_paraphrase_batch.py \
  --perturbation-type semantic_identical \
  --dataset physbench \
  --models gpt-5.2 \
  --api openai
```

### Next Session Follow-Up

1. Fetch and merge PhysBench physpara outputs when the batch is no longer
   `validating`, `in_progress`, or `finalizing`.

```bash
.venv/bin/python uncertainty_quantification_physical_reasoning/scripts/script_perturbation_prep/generic_paraphrase_batch/collect_generic_paraphrase_batch.py \
  --perturbation-type physics_aware \
  --dataset physbench \
  --models gpt-5.2 \
  --api openai
```

2. Fetch and merge PhysBench semident outputs.

```bash
.venv/bin/python uncertainty_quantification_physical_reasoning/scripts/script_perturbation_prep/generic_paraphrase_batch/collect_generic_paraphrase_batch.py \
  --perturbation-type semantic_identical \
  --dataset physbench \
  --models gpt-5.2 \
  --api openai
```

3. Verify the merged per-problem perturbation counts under
   `perturbations/physbench/`.

## Template For Future Dataset Entries

Copy this section and fill it in for the next dataset/run.

````md
## Entry: YYYY-MM-DD / <Dataset> / <Model>

### Goal

<short objective>

### Completed

- Saved problem IDs to:
  `uncertainty_quantification_physical_reasoning/perturbations/<dataset>/problem_ids_for_perturbation.json`
- Prepared batch input at:
  `uncertainty_quantification_physical_reasoning/batch_results/batch_inference/response_with_answer/<dataset>_<model>/batch_input.jsonl`
- Submitted batch:
  `<batch_id>`

### Batch Status At Handoff

- Batch ID: `<batch_id>`
- Current status: `<status>`
- Metadata file:
  `uncertainty_quantification_physical_reasoning/batch_results/batch_inference/response_with_answer/<dataset>_<model>/batch_metadata.json`

### Important Artifacts

- Problem ID file:
  `uncertainty_quantification_physical_reasoning/perturbations/<dataset>/problem_ids_for_perturbation.json`
- Batch directory:
  `uncertainty_quantification_physical_reasoning/batch_results/batch_inference/response_with_answer/<dataset>_<model>/`
- External dataset cache:
  `<cache path if relevant>`

### Relevant Local Code Changes

- <dataset-specific loader/prep changes>

### Environment Notes

- <extra dependencies, media requirements, auth requirements>

### Exact Commands Used

```bash
<prepare command>
<submit command>
```

### Next Session Follow-Up

1. <fetch command>
2. <parse command>
3. <verification command>
4. <next downstream step>
````
