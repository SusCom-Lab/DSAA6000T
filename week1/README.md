# Week 1: Profiling a Llama 8B model by component

## Goal

This week introduces GPU inference profiling using the real modules loaded from a student-provided Hugging Face-compatible Llama 8B checkpoint. The notebook measures representative components, explains what each component computes, and separates low-overhead latency measurement from detailed kernel tracing.

The notebook performs inference only. It does not train or modify the model.

## Llama execution path

```text
Token IDs → [Embedding] → [Decoder layer] × N → [Final RMSNorm] → [LM head] → logits

Inside one decoder layer:

x ──→ [RMSNorm] → [Q/K/V] → [RoPE] → [Causal attention] → [O projection] ──┐
└──────────────────────────────────────────────────────────────────────────(+) → h

h ──→ [RMSNorm] → [Gate + Up] → [SiLU(gate) ⊙ up] → [Down projection] ────┐
└──────────────────────────────────────────────────────────────────────────(+) → y
```

The attention and MLP rows reported by the notebook are **inclusive**: attention includes its Q/K/V and output projections, while MLP includes its gate/up/down projections and activation. Inclusive rows must not be added to their child rows.

## Components measured

| Component | Role | Main scaling variable |
| --- | --- | --- |
| Token embedding | Converts token IDs into hidden vectors | Number of input tokens |
| RMSNorm | Normalizes each hidden vector | Tokens × hidden size |
| Q/K/V projections | Produce queries, keys, and values | Tokens and projection dimensions |
| RoPE preparation | Produces rotary position factors | Sequence length and head dimension |
| Attention, inclusive | Q/K/V projections, causal attention, and output projection | Prefill query-key pairs |
| Output projection | Maps attention output back to hidden size | Number of input tokens |
| Gate/up/down projections | Dense matrix multiplications in the SwiGLU MLP | Tokens, hidden size, intermediate size |
| MLP, inclusive | Complete SwiGLU feed-forward block | Number of input tokens |
| Decoder layer, inclusive | One complete transformer layer | All operations above |
| Final RMSNorm | Normalizes the final hidden state | Number of input tokens |
| LM head | Produces next-token vocabulary logits | Batch size, hidden size, vocabulary size |

For a prefill sequence of length $L$, causal attention evaluates approximately

$$
\frac{L(L+1)}{2}
$$

query-key pairs per request. Dense projections scale approximately linearly with $L$, while prefill attention grows approximately quadratically with $L$.

## Measurement method

The notebook uses two complementary methods:

1. **CUDA events** for repeated latency measurements. It warms each component first, synchronizes the GPU, and reports median and 90th-percentile latency.
2. **`torch.profiler`** for a detailed operator and CUDA-kernel trace of one decoder layer. This trace is useful for attribution but has greater measurement overhead.

The component benchmark captures the real inputs received by a middle decoder layer during a full model forward pass, then replays each captured module independently. This avoids replacing the model with unrelated synthetic layer definitions.

### CUDA streams

A CUDA stream is an ordered queue of GPU work. PyTorch normally submits kernels to the current stream and returns control to Python before those kernels finish.

```text
Python thread:  launch QKV ─ launch attention ─ launch MLP ─ continue
                       │             │              │
                       ▼             ▼              ▼
CUDA stream 0:  ──[ QKV kernels ]─[ attention ]─[ MLP kernels ]──▶ time
```

Operations in the same stream execute in submission order. Operations in different streams may overlap when dependencies and GPU resources allow it:

```text
CUDA stream 0 (compute):  ──[ GEMM ]────────[ attention ]────────▶
CUDA stream 1 (copy):     ───────[ host-to-device copy ]────────▶
```

Because kernel launches are asynchronous, ordinary Python wall-clock timing can measure only launch overhead instead of GPU execution time.

### CUDA events

A CUDA event is a marker inserted into a stream. The GPU records its timestamp when all earlier work in that stream has reached the marker.

```text
CUDA stream 0:  ──[ start event ]─[ component kernels ]─[ end event ]──▶
                         │                                  │
                         └──── elapsed GPU time ─────────────┘
```

The basic measurement pattern is:

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

torch.cuda.synchronize()  # finish unrelated work before measurement
start.record()            # insert start marker into the current stream
component()
end.record()              # insert end marker after the component
end.synchronize()         # wait until the GPU reaches the end marker
elapsed_ms = start.elapsed_time(end)
```

`elapsed_time` measures GPU timeline time between the two events in milliseconds. The notebook first performs unmeasured warm-up calls, creates one event pair per repetition, and synchronizes once after all repetitions. This avoids placing a device-wide synchronization inside every measured component call.

Events belong to a stream. For cross-stream dependencies, one stream can record an event and another can wait for it:

```python
producer_stream.record_event(data_ready)
consumer_stream.wait_event(data_ready)
```

This creates an execution dependency without blocking the Python thread. Event timing still needs controlled streams and an otherwise quiet GPU: concurrent work in another stream can contend for GPU resources and change the measured latency.

## Configuration

Create a private Week 1 configuration file:

```bash
cd week1
cp .env.example .env
```

Edit `.env` and set at least:

```dotenv
LLAMA_MODEL_PATH=/path/to/your/llama-8b-checkpoint
CUDA_VISIBLE_DEVICES=0
```

`CUDA_VISIBLE_DEVICES` contains physical GPU IDs. For example, `CUDA_VISIBLE_DEVICES=2`
exposes only physical GPU 2, which the notebook addresses as `cuda:0`. Use `0,2` to expose
two GPUs; `PROFILE_DEVICE=cuda:0` then selects the first GPU in that visible list.

The notebook loads `.env` before importing PyTorch. Restart the notebook kernel after changing
the visible GPU selection. If `.env` is absent or `LLAMA_MODEL_PATH` is empty, the notebook asks
for the model directory interactively. The checkpoint must be Hugging Face-compatible; the
notebook uses `local_files_only=True` and does not download weights.

## Outputs

By default, the notebook creates `week1/profile-results/` and writes:

- `llama8b_component_latency.csv`: raw component statistics for every tested sequence length;
- `llama8b_component_latency.png`: component latency at the largest tested length;
- `llama8b_latency_scaling.png`: latency scaling across sequence lengths;
- `llama8b_decoder_layer_trace.json`: Chrome trace exported by `torch.profiler`.

Set `PROFILE_OUTPUT_DIR` in `week1/.env` to choose another output directory. Relative paths are
resolved from the `week1` directory.

## Interpreting the results

- Compare medians rather than a single timing sample.
- Use the 90th percentile to identify instability.
- Compare only runs using the same model, dtype, attention backend, batch size, and GPU state.
- Do not add inclusive attention, inclusive MLP, and their child-module times together.
- Treat isolated module timings as attribution data, not as an exact reconstruction of full-model latency.
- `torch.profiler` shape and memory tracing adds overhead; use CUDA-event results for the primary latency comparison.

## References

- [NVIDIA CUDA Programming Guide: asynchronous execution, streams, and events](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
- [PyTorch CUDA Event documentation](https://docs.pytorch.org/docs/stable/generated/torch.cuda.streams.Event.html)
- [PyTorch Profiler documentation](https://docs.pytorch.org/docs/stable/profiler.html)
- [Hugging Face Llama model documentation](https://huggingface.co/docs/transformers/model_doc/llama)
