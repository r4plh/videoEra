# Low Frame-rate Speech Codec (LFSC) — The Codec Behind Koel-TTS


---

## The Starting Point: What Problem Does LFSC Solve?

Audio LLMs (like Koel-TTS) predict audio tokens autoregressively — one timestep at a time. The speed of generation is directly limited by how many timesteps you need to predict per second of audio.

| Codec | Frame Rate | Codebooks | Tokens per second of audio |
|-------|-----------|-----------|---------------------------|
| DAC 16kHz | 50 fps | 12 (sequential, RVQ) | 50 × 12 = 600 sequential predictions |
| Encodec | 75 fps | 8 (sequential, RVQ) | 75 × 8 = 600 sequential predictions |
| Mimi (Kyutai) | 12.5 fps | 8 (sequential, RVQ) | 12.5 × 8 = 100 sequential predictions |
| **LFSC (Koel-TTS)** | **21.5 fps** | **8 (parallel, FSQ)** | **21.5 × 1 = 21.5 predictions** |

LFSC achieves the fewest effective predictions per second because: (1) low frame rate means fewer timesteps, and (2) FSQ independence means all 8 codebooks are predicted in parallel in a single decoder step.

---

## Architecture Overview

Like DAC, LFSC follows the encoder-quantizer-decoder pattern:

```
Raw audio waveform (22.05 kHz)
    ↓
Convolutional Encoder (high downsampling rate)
    ↓
Continuous latent: T × D (where T ≈ 21.5 per second)
    ↓
FSQ Quantization (8 codebooks, parallel)
    ↓
Discrete tokens: T × 8
    ↓
Convolutional Decoder
    ↓
Reconstructed audio waveform (22.05 kHz)
```

For 1 second of 22.05 kHz audio:
- Input: 22,050 samples
- After encoder: ~21.5 timesteps × D-dimensional continuous vectors
- After FSQ: ~21.5 timesteps × 8 discrete codes
- Total: ~172 integers represent 1 second of audio

Compare with DAC 16kHz: 50 timesteps × 12 codes = 600 integers per second.

---

## How Finite Scalar Quantization (FSQ) Works

This is the core difference from your DAC experience. Let's trace it step by step.

### The Key Idea

In RVQ (DAC), you have one high-dimensional vector and you quantize it through a chain of codebooks, each fixing the previous one's errors.

In FSQ, you take the continuous vector, project it to a small number of dimensions (4 in LFSC), and **independently round each dimension to discrete levels**. No chain. No residuals. No dependencies between codebooks.

### Step by Step Through FSQ

**Step 1 — Encoder output**: The encoder produces a continuous vector at each timestep. Let's say it's D-dimensional.

**Step 2 — Project to low-rank space**: Each of the 8 codebooks has its own learned projection that maps the D-dimensional vector to 4 dimensions:

```
z [D-dim] → projection_cb1 → z_cb1 [4-dim]
z [D-dim] → projection_cb2 → z_cb2 [4-dim]
...
z [D-dim] → projection_cb8 → z_cb8 [4-dim]
```

All 8 projections happen independently and in parallel.

**Step 3 — Quantize each dimension independently**: Each of the 4 dimensions is rounded to one of a fixed number of levels. LFSC uses codebook levels [8, 7, 6, 6]:

```
For codebook 1, z_cb1 = [0.73, -0.21, 0.45, 1.12]

Dimension 1: 8 levels → round 0.73 to nearest of 8 values → say level 5
Dimension 2: 7 levels → round -0.21 to nearest of 7 values → say level 2  
Dimension 3: 6 levels → round 0.45 to nearest of 6 values → say level 3
Dimension 4: 6 levels → round 1.12 to nearest of 6 values → say level 5

Quantized: z_cb1_quantized = [level_5, level_2, level_3, level_5]
```

The total number of possible codes per codebook = 8 × 7 × 6 × 6 = 2016.

This 4-tuple of levels IS the discrete code. It can be mapped to a single integer index (0 to 2015) for storage/transmission.

**Step 4 — Project back**: Each codebook projects its quantized 4-dim vector back to D-dim:

```
z_cb1_quantized [4-dim] → inverse_projection_cb1 → z_hat_cb1 [D-dim]
```

**Step 5 — Sum all codebook contributions**:

```
z_reconstructed = z_hat_cb1 + z_hat_cb2 + ... + z_hat_cb8
```

This sum goes to the decoder to reconstruct audio.

### Comparison: RVQ vs FSQ at a Glance

```
RVQ (DAC):
  z → CB1 quantize → residual₁ → CB2 quantize → residual₂ → CB3 quantize → ...
  SEQUENTIAL: each codebook needs the previous one's residual
  CB2 without CB1 = meaningless

FSQ (LFSC):
  z → CB1 quantize (independently)
  z → CB2 quantize (independently)  
  z → CB3 quantize (independently)
  ...all happen in PARALLEL
  Each codebook makes sense on its own
```

---

## Why Independence Matters for Koel-TTS

This is the practical payoff. At each decoder timestep, Koel-TTS needs to predict the audio tokens. With FSQ:

```
Decoder output at timestep t → Linear layer → 8 × 2016 logits
                                              → 8 independent softmax operations
                                              → 8 code predictions, ALL AT ONCE

One decoder step = one complete audio frame (all 8 codebooks)
```

With RVQ (if Koel-TTS used DAC instead):

```
Decoder needs to predict CB1 first
Then condition on CB1 to predict CB2
Then condition on CB1+CB2 to predict CB3
...
8 sequential steps per audio frame
```

For 10 seconds of audio:
- LFSC (FSQ): 215 decoder steps total
- DAC (RVQ): 215 × 12 = 2,580 decoder steps (or needs complex delay patterns)

---

## Codebook Size: Why 2016?

The original Spectral Codec (which LFSC builds upon) used 1000 codes per codebook at a higher frame rate. LFSC operates at 4× lower frame rate (21.5 vs ~86 fps), meaning each token must represent 4× more audio information. To compensate, they increased the codebook size from 1000 to 2016.

```
Levels per dimension: [8, 7, 6, 6]
Total codes: 8 × 7 × 6 × 6 = 2016

Compare with DAC: 1024 codes per codebook (but 8-dim vectors, not scalar)
```

---

## Bitrate Calculation

```
Bitrate = frame_rate × num_codebooks × bits_per_code
        = 21.5 × 8 × log2(2016)
        = 21.5 × 8 × ~11
        = 21.5 × 88
        ≈ 1,892 bps
        ≈ 1.89 kbps
```

Compare:
- DAC 16kHz: 50 × 12 × 10 = 6.0 kbps
- Mimi: 12.5 × 8 × 11 = 1.1 kbps
- LFSC: 1.89 kbps

Low bitrate = high compression = fewer bits to represent audio = efficient.

---

## Training Strategy

LFSC training has two phases (similar concept to how some codec papers train):

**Phase 1 — Pre-train without FSQ**: Train the encoder-decoder with continuous latents (no quantization). This lets the model learn good audio reconstruction first. ~62,000 steps on 96 A100 GPUs.

**Phase 2 — Fine-tune with FSQ enabled**: Enable the quantization bottleneck and fine-tune. The model adapts to work with discrete tokens. ~62,000 more steps.

This two-phase approach was used because FSQ convergence is faster when starting from a good continuous model.

Training losses (similar to DAC/HiFi-GAN family):
- Reconstruction losses (spectral, time-domain)
- Adversarial losses (discriminators judge if output sounds real)
- WavLM-based discriminator (uses a pretrained SSL model as an additional discriminator — improved speech intelligibility)

Training data: MLS English (25.5k hours) + Common Voice (3.2k hours, 105 languages)
Audio: 22.05 kHz, bandwidth filtered to 11 kHz
Training excerpts: 1.1-second clips

---

## The "No Delay Pattern" Advantage

With RVQ codecs (DAC, Encodec), downstream models like MusicGen needed "delay patterns" — a technique where codebook predictions are staggered in time to allow parallel-ish generation despite the sequential dependency:

```
RVQ delay pattern (MusicGen style):
  Time:    1    2    3    4    5
  CB1:    c1₁  c1₂  c1₃  c1₄  c1₅
  CB2:     -   c2₁  c2₂  c2₃  c2₄
  CB3:     -    -   c3₁  c3₂  c3₃
```

This adds latency (first audio chunk delayed by number of codebooks) and complexity.

With FSQ, no delay patterns needed. All codebooks predicted truly in parallel at each step. First audio chunk available immediately. This is critical for Koel-TTS's low-latency streaming goal.

---

## Practical Specifications

| Property | Value |
|----------|-------|
| Sample rate | 22.05 kHz |
| Frame rate | 21.5 fps |
| Codebooks | 8 |
| Quantization | FSQ |
| Dimensions per codebook | 4 |
| Levels per dimension | [8, 7, 6, 6] |
| Codes per codebook | 2016 |
| Bitrate | 1.89 kbps |
| Training data | ~28.7k hours (MLS + Common Voice) |
| Audio bandwidth | 11 kHz |

---

## Summary: LFSC vs DAC

| Aspect | DAC (your GenHencer) | LFSC (Koel-TTS) |
|--------|---------------------|-----------------|
| Quantization | RVQ (residual, sequential) | FSQ (independent, parallel) |
| Frame rate | 50 fps | 21.5 fps |
| Codebooks | 12 | 8 |
| Codes per CB | 1024 | 2016 |
| Code dimension | 8-dim vectors | 4 scalar dimensions |
| Bitrate | 6.0 kbps | 1.89 kbps |
| CB dependency | Sequential (coarse-to-fine) | Independent (parallel) |
| Sample rate | 16 kHz | 22.05 kHz |
| Downstream prediction | Sequential across CBs or delay patterns | All CBs in one step |

The fundamental shift: RVQ asks "what did the previous codebook miss?" FSQ asks "what does each dimension independently look like?" Both reconstruct audio by summing contributions, but the independence in FSQ unlocks parallel prediction that makes AR models like Koel-TTS practical and fast.
