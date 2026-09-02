# VoiceGuard V2 — Target Architecture

**Date:** 2026-09-02
**Status:** Design document — not yet implemented
**Prerequisite:** [VOICEGUARD_CODEBASE_AUDIT.md](./VOICEGUARD_CODEBASE_AUDIT.md)

---

## Table of Contents

1. [Overall System Architecture](#1-overall-system-architecture)
2. [Frontend/Client Architecture](#2-frontendclient-architecture)
3. [Backend Architecture](#3-backend-architecture)
4. [Voice Biometric Pipeline](#4-voice-biometric-pipeline)
5. [Speaker Verification Architecture](#5-speaker-verification-architecture)
6. [CNN-LSTM-Attention Model Architecture](#6-cnn-lstm-attention-model-architecture)
7. [Audio Preprocessing Pipeline](#7-audio-preprocessing-pipeline)
8. [Feature Extraction Pipeline](#8-feature-extraction-pipeline)
9. [Anti-Spoofing Architecture](#9-anti-spoofing-architecture)
10. [Replay Attack Detection](#10-replay-attack-detection)
11. [Deepfake/Synthetic Speech Detection](#11-deepfakesynthetic-speech-detection)
12. [Voice Conversion/Mimicry Detection](#12-voice-conversionmimicry-detection)
13. [Dynamic Challenge-Response Liveness](#13-dynamic-challenge-response-liveness)
14. [Biometric Template Protection](#14-biometric-template-protection)
15. [User Authentication/Session Architecture](#15-user-authenticationsession-architecture)
16. [Transaction Authorization Architecture](#16-transaction-authorization-architecture)
17. [Database Schema](#17-database-schema)
18. [Model Storage/Versioning](#18-model-storageversioning)
19. [API Boundaries](#19-api-boundaries)
20. [Security Boundaries](#20-security-boundaries)
21. [Configuration/Secrets Management](#21-configurationsecrets-management)
22. [Logging/Audit Architecture](#22-loggingaudit-architecture)
23. [Testing Strategy](#23-testing-strategy)
24. [ML Training/Evaluation Strategy](#24-ml-trainingevaluation-strategy)
25. [Deployment Architecture](#25-deployment-architecture)
26. [Android/Mobile Strategy](#26-androidmobile-strategy)
27. [Future Scalability](#27-future-scalability)

---

## 1. Overall System Architecture

### 1.1 Current State

The entire application is a single 1,646-line Python file (`app.py`) containing the GUI (Kivy), ML models, database, authentication logic, and payment flow all coupled together in a monolithic structure with in-memory-only storage.

### 1.2 Target Architecture

The V2 system decomposes into four independent layers that communicate through well-defined interfaces:

```
+------------------------------------------------------------------+
|                        CLIENT LAYER                               |
|                                                                    |
|  +----------------+  +-----------------+  +------------------+   |
|  |  Android App   |  |  Desktop App    |  |  Web Client      |   |
|  |  (Kotlin)      |  |  (Python/Kivy)  |  |  (React - Future)|   |
|  +-------+--------+  +--------+--------+  +--------+---------+   |
|          |                    |                     |             |
|          +------------+-------+---------------------+             |
|                       | gRPC / REST                               |
+-----------------------+-------------------------------------------+
                        |
+-----------------------+-------------------------------------------+
|                  API GATEWAY LAYER                                |
|                                                                   |
|  +---------------------v---------------------------------------+ |
|  |              FastAPI Application Server                      | |
|  |                                                              | |
|  |  +----------+ +------------+ +--------------+ +----------+  | |
|  |  | Auth     | | User       | | Transaction  | | Voice    |  | |
|  |  | Routes   | | Routes     | | Routes       | | Routes   |  | |
|  |  +----+-----+ +-----+------+ +------+-------+ +----+-----+  | |
|  +-----+-----------+-----------+-----------+-----------+--------+ |
|        |           |           |           |                      |
+--------+-----------+-----------+-----------+----------------------+
         |           |           |           |
+--------v-----------v-----------v-----------v----------------------+
|                    SERVICE LAYER                                   |
|                                                                    |
|  +--------------+ +------------+ +--------------+ +----------+   |
|  | User Service | | Auth       | | Transaction  | | Voice    |   |
|  |              | | Service    | | Service      | | Pipeline |   |
|  +------+-------+ +-----+------+ +------+-------+ +----+-----+   |
|         |               |               |              |          |
|  +------v---------------v---------------v--------------v-------+  |
|  |                   Data Access Layer                           |  |
|  +--------------------------+-----------------------------------+  |
+-----------------------------+--------------------------------------+
                              |
+-----------------------------+--------------------------------------+
|                         DATA LAYER                                 |
|                                                                    |
|  +----------+ +-----------+ +------------+ +------------------+  |
|  |PostgreSQL| |  Redis    | | Model      | | Object Storage   |  |
|  |          | |           | | Registry   | | (MinIO/S3)       |  |
|  +----------+ +-----------+ +------------+ +------------------+  |
+-------------------------------------------------------------------+
```

### 1.3 Decomposition Summary

| Layer | Responsibility | Communication |
|-------|---------------|---------------|
| **Client** | UI, audio capture, on-device preprocessing | gRPC/REST over TLS |
| **API Gateway** | Routing, authentication, rate limiting, validation | HTTP/gRPC |
| **Services** | Business logic, orchestration | Internal Python imports / async |
| **Data** | Persistence, caching, model versioning | DB drivers, Redis protocol, S3 API |

### 1.4 Migration Path from Current State

| Current | V2 Replacement | Notes |
|---------|---------------|-------|
| Single `app.py` | Multi-package Python project with `pyproject.toml` | Preserve core ML logic, restructure around modules |
| Kivy-only client | Desktop client (Kivy) preserved, Android added later | Desktop remains primary initially |
| In-memory dicts | PostgreSQL + Redis | Required for any persistence |
| No API layer | FastAPI application | REST + WebSocket for streaming |
| Fernet key per run | External key management (env vars / Vault) | Must survive restarts |

---

## 2. Frontend/Client Architecture

### 2.1 Current State

Three Kivy screens (`LoginScreen`, `RegistrationScreen`, `PaymentScreen`) in `app.py` with:

- Background audio recording via `sounddevice.InputStream` in daemon threads
- Real-time waveform visualization via `VoiceWaveform` widget
- Hold-to-record button interaction pattern
- Popup-based feedback (success, decline, errors)

### 2.2 What Can Be Reused

| Component | Location | Strategy |
|-----------|----------|----------|
| `VoiceWaveform` widget | `app.py:544-564` | Extract as standalone widget |
| `PaytmButton` | `app.py:507-525` | Rename to `VGButton`, extract styling |
| `PaytmCard` | `app.py:528-543` | Rename to `VGCard`, extract |
| Audio recording thread pattern | Triplicated in screens | Extract into reusable `AudioRecorder` class |
| Hold-to-record UX pattern | All three screens | Preserve interaction model |
| Progress indicators | Registration + auth screens | Reuse |

### 2.3 What Must Be Replaced

| Component | Issue | Replacement |
|-----------|-------|-------------|
| `LoginScreen` UI (lines 567-792) | Monolithic, no input validation, hardcoded styles | Refactored screen with form validation |
| `RegistrationScreen` UI (lines 794-1105) | Duplicated audio recording logic, no error recovery | Extract recording to service, add quality feedback |
| `PaymentScreen` UI (lines 1108-1615) | Duplicated recording, challenge not validated, balance in local var | Separate data binding from UI |
| Challenge phrase display | Displayed but never sent to backend for validation | Send audio + challenge ID to backend |
| Balance management | `self.user_balance` float on screen widget | Fetch from server, display read-only |

### 2.4 What Must Be Newly Built

- **`AudioRecorder` service class** — single shared class eliminating triplicated `_record_audio`/`_audio_callback`
- **Form validation layer** — username format, email format, amount bounds, recipient validation
- **Error handling framework** — structured error responses from backend
- **State management** — separation of UI state from business state
- **WebSocket connection** — for real-time auth status, streaming audio upload
- **Settings/Preferences screen** — microphone selection (currently hardcoded)
- **On-device preprocessing** — audio quality checks locally before upload

### 2.5 Target Client Package Structure

```
client/
  voiceguard/
    __init__.py
    app.py                  # VoiceGuardApp entry point
    screens/
      __init__.py
      login.py              # LoginScreen
      registration.py       # RegistrationScreen
      payment.py            # PaymentScreen
      settings.py           # NEW: microphone settings
    widgets/
      __init__.py
      waveform.py           # VoiceWaveform (extracted)
      vg_button.py          # VGButton (renamed from PaytmButton)
      vg_card.py            # VGCard (renamed from PaytmCard)
    services/
      __init__.py
      audio_recorder.py     # NEW: shared audio recording
      api_client.py         # NEW: HTTP client for backend
      auth_manager.py       # NEW: session/token management
    config.py               # NEW: application configuration
  pyproject.toml
  tests/
```

---

## 3. Backend Architecture

### 3.1 Current State

No backend exists. All logic (authentication, enrollment, payment, storage) runs in-process inside the Kivy application using Python dicts as storage.

### 3.2 Target: FastAPI Application

```
backend/
  pyproject.toml
  voiceguard/
    __init__.py
    main.py                # FastAPI app factory
    config.py              # Pydantic settings
    dependencies.py        # Dependency injection
    routes/
      __init__.py
      auth.py              # POST /enroll, POST /login
      users.py             # GET/PUT /users/me
      transactions.py      # POST /transactions, GET /transactions
      voice.py             # POST /voice/verify (streaming)
    services/
      __init__.py
      user_service.py
      auth_service.py
      transaction_service.py
      voice_service.py     # Orchestrates voice pipeline
    models/
      __init__.py
      user.py              # SQLAlchemy models
      voice_model.py
      transaction.py
    schemas/
      __init__.py
      user.py              # Pydantic request/response schemas
      voice.py
      transaction.py
    voice/
      __init__.py
      pipeline.py          # Main voice pipeline orchestrator
      preprocessing.py
      features.py
      embedding.py
      verification.py
      anti_spoofing.py
      challenge.py         # Challenge-response liveness
    ml/
      __init__.py
      models/
        __init__.py
        cnn_lstm_attention.py  # NEW: target model
        embedding_net.py       # MIGRATED from VoiceEmbeddingNet
        anti_spoofing_models.py
      training/
        __init__.py
        train_speaker.py
        train_anti_spoof.py
        evaluate.py
      registry.py            # Model version management
    security/
      __init__.py
      crypto.py             # Fernet + key derivation (from current code)
      tokens.py             # JWT session tokens
      rate_limit.py         # Rate limiting
    db/
      __init__.py
      session.py            # SQLAlchemy async session
      migrations/           # Alembic migrations
  tests/
```

### 3.3 What Can Be Reused from Current Code

| Code | Location | Reuse Strategy |
|------|----------|----------------|
| `SecureDatabase` Fernet encryption logic | `app.py:68-81` | Extract to `security/crypto.py`, persist key externally |
| `VoiceAuthenticationEngine.enroll_user()` | `app.py:356-390` | Move core logic to `services/voice_service.py` |
| `VoiceAuthenticationEngine.authenticate()` | `app.py:392-467` | Move to `services/auth_service.py`, add challenge validation |
| `_train_gmm()` | `app.py:488-495` | Retain in `voice/verification.py` |
| `_gmm_score()` | `app.py:497-503` | Retain in `voice/verification.py` |
| Scoring formula (line 447) | `app.py:447` | Parameterize in config, initially retain |

### 3.4 What Must Be Replaced

| Component | Current State | V2 |
|-----------|--------------|-----|
| Storage | In-memory Python dicts | PostgreSQL (users, transactions) + Redis (sessions, cache) |
| User management | Plain dict with encrypted JSON blob | SQLAlchemy model with proper columns |
| Transaction logs | Python list | Database table with ACID guarantees |
| Auth attempts tracking | Dict with manual timeout | Redis TTL-based tracking |
| Configuration | Hardcoded constants throughout | Pydantic `BaseSettings` with env vars |
| Error handling | Unstructured, crashes propagate | HTTPException with structured error responses |

### 3.5 What Must Be Newly Built

- FastAPI application with middleware stack (CORS, logging, rate limiting, request ID)
- SQLAlchemy async models and Alembic migration pipeline
- JWT-based session management (access + refresh tokens)
- Pydantic request/response validation schemas
- Dependency injection for DB sessions, service instances, auth context
- Background task queue for model retraining, audit log flushing
- Health check endpoints
- OpenAPI documentation (auto-generated by FastAPI)

---

## 4. Voice Biometric Pipeline

### 4.1 Current State

```
Audio Input
    |
    v
VoiceFeatureExtractor.extract_features()     <-- app.py:117-159
    +-- Normalize amplitude
    +-- Spectral subtraction noise reduction
    +-- Energy-based VAD
    +-- MFCC (40) + delta
    +-- Mel-spectrogram (128 mels)
    +-- Spectral centroid, rolloff, ZCR
    +-- Pitch (pyin)
    +-- RMS energy
    +-- Formants (LPC)
    |
    v
_prepare_feature_vector()                     <-- app.py:469-486
    +-- Concatenate mfcc_mean, mfcc_std, mfcc_delta, mel_mean -> pad/truncate to 256
    |
    v
VoiceEmbeddingNet (random weights)            <-- app.py:222-238
    +-- 256 -> 512 -> 256 -> 128 (L2-normalized)
    |
    v
Cosine similarity + GMM score                 <-- app.py:434-447
    +-- 0.6 * max_sim + 0.3 * avg_sim + 0.1 * gmm_score
```

### 4.2 Target Pipeline

```
Audio Input (16kHz mono float32)
    |
    v
+---------------------------------------------+
|  Stage 1: Preprocessing                      |
|  voice/preprocessing.py                      |
|                                              |
|  +-- Amplitude normalization                 |
|  +-- Advanced noise reduction                |
|  |   (replace spectral subtraction with      |
|  |    adaptive Wiener filter or              |
|  |    RNNoise-style approach)                |
|  +-- Robust VAD                              |
|  |   (replace energy-only with               |
|  |    energy + zero-crossing + spectral      |
|  |    flux, or WebRTC VAD wrapper)           |
|  +-- Pre-emphasis filter                     |
|  +-- Silence trimming                        |
+-----------------+----------------------------+
                  |
                  v
+---------------------------------------------+
|  Stage 2: Feature Extraction                 |
|  voice/features.py                           |
|                                              |
|  Primary (for CNN-LSTM-Attention):           |
|  +-- Mel-spectrogram (80 mels)              |
|  |   -> fed directly to CNN front-end        |
|  |   (replace fixed-size vector approach)    |
|                                              |
|  Secondary (for GMM / scoring):              |
|  +-- MFCC (40) + delta + delta-delta        |
|  +-- Spectral features                       |
|  +-- Prosodic features (pitch, energy)       |
|  +-- Formants (LPC)                          |
|                                              |
|  Anti-spoofing features:                     |
|  +-- Spectral envelope statistics            |
|  +-- Phase spectrum features                 |
|  +-- High-frequency band energy ratio        |
|  +-- Cepstral variance features              |
+-----------------+----------------------------+
                  |
                  v
+---------------------------------------------+
|  Stage 3: Speaker Embedding                  |
|  ml/models/cnn_lstm_attention.py             |
|                                              |
|  Input: mel-spectrogram (T x 80)            |
|  +-- CNN front-end (conv blocks)             |
|  +-- Bi-LSTM sequence modeling               |
|  +-- Attention pooling                       |
|  +-- 256-dim L2-normalized embedding         |
+-----------------+----------------------------+
                  |
                  v
+---------------------------------------------+
|  Stage 4: Anti-Spoofing                      |
|  voice/anti_spoofing.py                      |
|                                              |
|  +-- Replay detection (heuristic + ML)       |
|  +-- Deepfake detection (neural)             |
|  +-- Voice conversion detection              |
|  +-- Challenge-response validation           |
|  (see Sections 9-13)                         |
+-----------------+----------------------------+
                  |
                  v
+---------------------------------------------+
|  Stage 5: Verification                       |
|  voice/verification.py                       |
|                                              |
|  +-- Cosine similarity vs. stored            |
|  |   embeddings (centroid + extremes)        |
|  +-- GMM likelihood score                    |
|  +-- Adaptive threshold                      |
|  |   (replace graduated lower-first)         |
|  +-- Final decision                          |
+---------------------------------------------+
```

### 4.3 Migration Mapping

| Current Component | Status | V2 Action |
|-------------------|--------|-----------|
| `VoiceFeatureExtractor` (lines 109-218) | Implemented but primitive | **Replace** noise reduction and VAD, **reuse** feature extraction, **extend** with anti-spoofing features |
| `VoiceEmbeddingNet` (lines 222-238) | Implemented, untrained | **Replace** with CNN-LSTM-Attention. Keep old as fallback |
| `_prepare_feature_vector` (lines 469-486) | Truncates/pads to fixed 256 | **Replace** -- CNN operates on variable-length mel-spectrograms |
| Scoring formula (line 447) | Implemented | **Parameterize** in config, **extend** with additional signals |
| GMM scoring (lines 488-503) | 2-component diagonal | **Reuse** with configurable n_components (increase to 8-16) |
| Graduated thresholds (lines 353-354, 449-453) | 0.72 then 0.80 | **Replace** -- first-attempt-lower is a security weakness |

---

## 5. Speaker Verification Architecture

### 5.1 Current State

Speaker verification uses random-weight neural network embeddings (cosine similarity), GMM likelihood scoring (2-component), combined weighted score, and binary decision against threshold (0.72 first attempt, 0.80 otherwise).

### 5.2 Target Architecture

#### Scoring Strategy

```
Enrollment (N samples >= 5):
    |
    +-- For each sample -> extract mel-spectrogram
    +-- Pass through CNN-LSTM-Attention -> embedding (256-dim)
    +-- Store: enrollment_embeddings[] (N vectors)
    +-- Compute: centroid, intra-class variance, convex hull vertices
    +-- Train: per-user GMM-UBM (8-component, diagonal)

Authentication (1 sample):
    |
    +-- Extract mel-spectrogram -> CNN-LSTM-Attention -> embedding
    |
    +-- Signal-level checks:
    |   +-- Audio quality score (SNR, duration, clipping, silence)
    |   +-- Anti-spoofing composite score (see Sections 9-12)
    |   +-- Challenge-response match score (see Section 13)
    |
    +-- Speaker matching:
    |   +-- Cosine similarity to centroid of enrollment embeddings
    |   +-- Max cosine similarity to any enrollment embedding
    |   +-- Min cosine similarity (detect outlier samples)
    |   +-- Mahalanobis distance using enrollment covariance
    |   +-- GMM log-likelihood ratio (user model vs. background)
    |
    +-- Decision:
        +-- Compute composite score (configurable weights)
        +-- Apply threshold (single adaptive threshold, not graduated)
        +-- If score marginally below threshold -> request additional sample
        +-- If score well below threshold -> reject
```

#### Adaptive Threshold (Replacing Graduated Thresholds)

The current system uses a lower threshold for first attempts (0.72 vs 0.80), which weakens security. V2 replaces this with:

- **Single base threshold** (e.g., 0.82) applied uniformly
- **Confidence band**: scores between threshold-0.05 and threshold trigger "soft accept + request confirmation"
- **Attempt limiting**: maximum 5 attempts per session, then lockout with escalating cooldown (30s then 1m then 5m)
- **Adaptive per-user threshold**: computed at enrollment based on intra-class variance of enrollment samples

### 5.3 What Can Be Reused

| Component | Source | Action |
|-----------|--------|--------|
| Cosine similarity computation | `app.py:434-438` | Reuse, extend to centroid computation |
| GMM scoring | `app.py:488-503` | Reuse, increase component count |
| Feature extraction for GMM | `app.py:469-486` | Keep as secondary feature vector for GMM |
| Attempt tracking | `app.py:88-105` | Reuse logic, move to Redis-backed implementation |

### 5.4 What Must Be Newly Built

- Centroid + covariance computation at enrollment
- Mahalanobis distance scoring
- Adaptive per-user threshold calculation
- Confidence-band soft accept flow
- Attempt limiting with cooldown escalation
- Score calibration pipeline (tune thresholds against validation set)

---

## 6. CNN-LSTM-Attention Model Architecture

### 6.1 Current State

`VoiceEmbeddingNet` (`app.py:222-238`): a 3-layer fully connected network (256->512->256->128) with BatchNorm1d and Dropout. Initialized with random PyTorch weights. Never trained.

### 6.2 Target: CNN-LSTM-Attention Network

```
Input: Mel-spectrogram
       Shape: (batch, 1, n_mels=80, T)
       where T = variable-length (trimmed/padded to max ~300 frames = ~3s)

+----------------------------------------------+
|  CNN Front-End (per-frame feature learning)   |
|                                               |
|  Block 1:                                    |
|  +-- Conv2d(1, 32, kernel=(3,3), padding=1)  |
|  +-- BatchNorm2d(32)                         |
|  +-- ReLU                                    |
|  +-- MaxPool2d((2,2))                        |
|                                               |
|  Block 2:                                    |
|  +-- Conv2d(32, 64, kernel=(3,3), padding=1) |
|  +-- BatchNorm2d(64)                         |
|  +-- ReLU                                    |
|  +-- MaxPool2d((2,2))                        |
|                                               |
|  Block 3:                                    |
|  +-- Conv2d(64, 128, kernel=(3,3), padding=1)|
|  +-- BatchNorm2d(128)                        |
|  +-- ReLU                                    |
|  +-- AdaptiveAvgPool2d((1, None))            |
|  |   -> squeeze height dim -> (batch, 128, T') |
|                                               |
|  Reshape: (batch, T', 128)                   |
+-------------------+---------------------------+
                    |
                    v
+----------------------------------------------+
|  Bi-LSTM Sequence Modeling                    |
|                                               |
|  +-- BiLSTM(input=128, hidden=256,           |
|  |         num_layers=2, dropout=0.3)         |
|  |   -> output: (batch, T', 512)             |
|  |   -> (256 forward + 256 backward)         |
|  +-- Linear(512, 256)                         |
|  +-- ReLU                                    |
|       -> output: (batch, T', 256)            |
+-------------------+---------------------------+
                    |
                    v
+----------------------------------------------+
|  Attention Pooling                            |
|                                               |
|  +-- Attention weights:                       |
|  |   alpha = softmax(W_a . tanh(h_t))        |
|  |   where h_t = each frame's BiLSTM output   |
|  |   W_a: (256, 256) learned weight matrix   |
|  |                                               |
|  +-- Context vector:                           |
|  |   c = sum(alpha_t * h_t)  (weighted sum)   |
|  |   -> output: (batch, 256)                  |
|  |                                               |
|  +-- Projection:                              |
|  |   Linear(256, embedding_dim=256)           |
|  |                                               |
|  +-- L2 Normalize:                            |
|      embedding = c / ||c||_2                  |
|      -> output: (batch, 256)                  |
+----------------------------------------------+
```

### 6.3 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Input representation | Raw mel-spectrogram (80 mels) | Let CNN learn optimal features |
| CNN depth | 3 blocks | Sufficient for speech spectral patterns |
| LSTM layers | 2-layer Bi-LSTM | Captures temporal dependencies in both directions |
| Hidden size | 256 per direction (512 total) | Balance between capacity and inference speed |
| Attention mechanism | Additive (Bahdanau-style) | Proven for variable-length utterances |
| Embedding dim | 256 | Upgrade from current 128 |
| Max pooling | Replaced by attention | More flexible for variable-length input |
| Dropout | 0.3 on LSTM, 0.2 on CNN | Regularization for limited enrollment data |

### 6.4 Migration from Current Model

| Current `VoiceEmbeddingNet` | V2 `CNNLSTMAttention` |
|----------------------------|----------------------|
| Input: fixed 256-dim vector | Input: variable-length mel-spectrogram (Tx80) |
| 3 FC layers (256->512->256->128) | CNN blocks -> Bi-LSTM -> Attention -> FC -> 256-dim |
| Output: 128-dim L2-normalized | Output: 256-dim L2-normalized |
| Random weights, never trained | Trained on speaker verification loss |
| No temporal modeling | Full temporal modeling via LSTM + attention |
| No spectral pattern learning | CNN learns spectral patterns from raw mel |

### 6.5 Backward Compatibility During Migration

- Keep `VoiceEmbeddingNet` as `LegacyEmbeddingNet` during development
- `voice/verification.py` accepts either model via configuration
- Default to new model once trained weights are available
- Old model remains available as `--model legacy` flag for comparison

---

## 7. Audio Preprocessing Pipeline

### 7.1 Current State

Located in `VoiceFeatureExtractor` (`app.py:109-218`):

```python
def preprocess_audio(self, audio_data):
    audio_data = audio_data / np.max(np.abs(audio_data))   # normalize
    audio_data = self.spectral_subtraction(audio_data)      # naive noise reduction
    audio_data = self.apply_vad(audio_data)                 # energy-only VAD
    return audio_data
```

Problems identified:

- **Normalization**: Divides by max amplitude -- simple but crashes on empty data
- **Noise reduction**: First 10% of frames as noise estimate -- assumes stationary noise only at start
- **VAD**: Energy-only threshold (0.02) -- fails with quiet speakers, non-stationary noise

### 7.2 Target Preprocessing Pipeline

```
Raw Audio (16kHz, mono, float32)
    |
    v
+-------------------------------------+
|  Step 1: Validation                  |
|  +-- Check sample rate == 16000     |
|  +-- Check mono channel             |
|  +-- Check non-empty                |
|  +-- Check float32 range [-1, 1]   |
+-----------------+-------------------+
                  |
                  v
+-------------------------------------+
|  Step 2: Amplitude Normalization    |
|  +-- Peak normalization to -1 dBFS |
|  +-- With clipping guard           |
|  +-- Handle near-silence edge case  |
+-----------------+-------------------+
                  |
                  v
+-------------------------------------+
|  Step 3: Noise Reduction            |
|  Option A: Adaptive Wiener filter   |
|  Option B: Spectral gating          |
|    +-- Estimate noise profile from  |
|    |   VAD-detected silence frames  |
|    |   (uses actual silence,        |
|    |    not just first 10%)         |
|    +-- Spectral subtraction with    |
|    |   oversubtraction factor 2.0   |
|    +-- Spectral floor (0.01)       |
+-----------------+-------------------+
                  |
                  v
+-------------------------------------+
|  Step 4: Voice Activity Detection   |
|  Multi-feature VAD:                 |
|  +-- Short-time energy              |
|  +-- Zero-crossing rate             |
|  +-- Spectral flux                  |
|  +-- Combined threshold decision    |
|  +-- Morphological smoothing        |
|      (fill gaps < 0.15s,            |
|       trim edges < 0.05s)          |
|                                     |
|  Alternative: WebRTC VAD wrapper    |
+-----------------+-------------------+
                  |
                  v
+-------------------------------------+
|  Step 5: Silence Trimming           |
|  +-- Trim leading/trailing silence  |
|  +-- Preserve 50ms margin          |
|  +-- Reject if voice < 0.5s        |
|      after trimming                 |
+-----------------+-------------------+
                  |
                  v
+-------------------------------------+
|  Step 6: Pre-emphasis               |
|  y[n] = x[n] - 0.97 * x[n-1]     |
|  (Apply only for LPC/formant       |
|   analysis; not for mel-spectrogram)|
+-------------------------------------+
```

### 7.3 Migration Mapping

| Current Step | Code Location | V2 Action |
|-------------|---------------|-----------|
| Amplitude normalization | `app.py:164` | **Improve** -- add clipping guard, edge case handling |
| Spectral subtraction | `app.py:174-192` | **Replace** -- noise profile from VAD-detected silence |
| Energy VAD | `app.py:194-202` | **Replace** -- multi-feature VAD or WebRTC VAD |
| Pre-emphasis | `app.py:207` | **Reuse** in formant estimation path only |
| Formant estimation | `app.py:204-218` | **Reuse** as-is, move to secondary feature path |

---

## 8. Feature Extraction Pipeline

### 8.1 Current State

`VoiceFeatureExtractor.extract_features()` (`app.py:117-159`) produces a dict. Then `_prepare_feature_vector()` concatenates only 4 fields (mfcc_mean, mfcc_std, mfcc_delta, mel_mean) and pads/truncates to 256. Spectral, prosodic, energy, and formant features are discarded.

### 8.2 Target Feature Extraction

**Path A: Primary (for CNN-LSTM-Attention)**

```
Input: preprocessed audio (16kHz, trimmed)

Mel-Spectrogram:
    +-- n_mels = 80
    +-- n_fft = 512
    +-- hop_length = 160 (10ms)
    +-- f_min = 50
    +-- f_max = 8000
    +-- Log-scale: log(mel_spec + 1e-6)
    +-- Output shape: (80, T) where T ~ duration_seconds * 100
    +-- No averaging -- preserve temporal dimension for CNN
```

**Path B: Secondary (for GMM scoring + anti-spoofing)**

```
Statistical feature vector:
    +-- MFCC (40) + delta (40) + delta-delta (40)
    |   -> mean + std per coefficient: 120 * 2 = 240 features
    +-- Spectral centroid, rolloff, bandwidth, contrast (4)
    +-- ZCR (mean + std = 2)
    +-- Pitch (mean, std, range, voiced_fraction = 4)
    +-- RMS energy (mean, std = 2)
    +-- Formants F1, F2, F3 (mean + std = 6)
    +-- Spectral flatness (mean = 1)
    +-- Total: ~261 features
    +-- Output: 1D numpy array (no truncation/padding)
```

**Path C: Anti-Spoofing Features (new)**

```
Anti-spoofing feature vector:
    +-- High-frequency energy ratio (band > 4kHz / total)
    +-- Spectral envelope smoothness
    +-- Cepstral variance (MFCC variance across time)
    +-- Phase spectrum statistics
    +-- Fundamental frequency contour stability
    +-- Jitter and shimmer (cycle-to-cycle variation)
    +-- Spectral tilt (low-freq / high-freq energy ratio)
    +-- Group delay features
```

### 8.3 Migration Mapping

| Current Feature | Code Location | V2 Action |
|----------------|---------------|-----------|
| MFCC + delta | `app.py:125-129` | **Reuse** with delta-delta added |
| Mel-spectrogram | `app.py:132-134` | **Reuse** for GMM path; **new** for CNN path (variable-length, log-scale) |
| Spectral centroid/rolloff/ZCR | `app.py:137-142` | **Reuse** -- add bandwidth, contrast |
| Pitch (pyin) | `app.py:145-149` | **Reuse** -- add voiced_fraction, range |
| Energy (rms) | `app.py:152-154` | **Reuse** |
| Formants (LPC) | `app.py:157, 204-218` | **Reuse** -- add mean+std per formant |
| `_prepare_feature_vector` | `app.py:469-486` | **Delete** -- CNN takes raw mel; GMM gets full vector |

---

## 9. Anti-Spoofing Architecture

### 9.1 Current State

`AntiSpoofingDetector` (`app.py:242-341`) implements audio quality checks, replay detection via cosine similarity, and challenge phrase generation (never validated). The `OneClassSVM` is instantiated but never trained or used.

### 9.2 Target Anti-Spoofing Architecture

```
                    Incoming Audio
                         |
                         v
            +---------------------------+
            |   Quality Gate             |
            |   (from current code)      |
            |   +-- SNR check            |
            |   +-- Clipping check       |
            |   +-- Silence ratio        |
            |   +-- Duration check       |
            +------------+--------------+
                         | pass
                         v
            +---------------------------+
            |   Liveness Detector        |
            |   (3 parallel checks)      |
            |                            |
            |  +--------------------+    |
            |  | Challenge-Response |    | <-- Section 13
            |  | Speech-to-text     |    |
            |  +--------------------+    |
            |  +--------------------+    |
            |  | Replay Detector    |    | <-- Section 10
            |  | Feature + ML       |    |
            |  +--------------------+    |
            |  +--------------------+    |
            |  | Deepfake Detect    |    | <-- Section 11
            |  | Neural classifier  |    |
            |  +--------------------+    |
            +------------+--------------+
                         |
                         v
            +---------------------------+
            |  Composite Anti-Spoof     |
            |  Score                     |
            |                            |
            |  score = w1*challenge     |
            |        + w2*replay        |
            |        + w3*deepfake      |
            |                            |
            |  Decision:                 |
            |  score > threshold         |
            |    -> proceed to           |
            |       speaker verification |
            |  score <= threshold        |
            |    -> REJECT               |
            +---------------------------+
```

### 9.3 What Can Be Reused

| Component | Source | V2 Action |
|-----------|--------|-----------|
| `check_audio_quality()` | `app.py:308-329` | **Reuse** with tightened thresholds |
| `_calculate_snr()` | `app.py:331-341` | **Reuse** |
| `detect_replay()` heuristic | `app.py:260-296` | **Extend** -- use as one signal, not sole detector |
| `generate_challenge()` | `app.py:253-258` | **Replace** -- expand vocabulary, add challenge_id |
| `_features_to_vector()` | `app.py:298-306` | **Delete** -- use structured feature paths |

### 9.4 What Must Be Newly Built

- Challenge-response validation via speech-to-text
- Neural replay detection model (trained on ASVspoof)
- Deepfake/synthetic speech detection classifier
- Voice conversion detection module
- Composite anti-spoof scoring with configurable weights

---

## 10. Replay Attack Detection

### 10.1 Current State

`AntiSpoofingDetector.detect_replay()` (`app.py:260-296`) computes cosine similarity between current feature vector and last 10 historical feature vectors. Requires 5+ historical samples (inactive during initial use). Cannot detect replay through a different speaker or with room convolution.

### 10.2 Target Replay Detection

**Tier 1: Heuristic (Fast, No Training Required)**

```
Replay Heuristic Features:
    +-- Spectral bandwidth consistency
    |   (live speech has natural bandwidth variation)
    +-- Background noise profile match
    |   (replay introduces speaker's environment noise)
    +-- Channel frequency response signature
    |   (microphone -> speaker -> microphone adds coloration)
    +-- Temporal fine structure consistency
    |   (replay may have quantization artifacts)
    +-- Amplitude envelope naturalness
        (replay may have flat or unnatural envelopes)
```

**Tier 2: ML-Based (Trained Classifier)**

```
Trained Replay Detector:
    +-- Input: mel-spectrogram + anti-spoof features
    +-- Architecture: 1D CNN (3 blocks) + FC classifier
    +-- Output: binary (live vs. replay) + confidence score
    +-- Training data: ASVspoof 2019/2021 logical/physical access
    +-- Inference: ~10ms on CPU
```

### 10.3 What Can Be Reused

The current cosine-similarity heuristic (`detect_replay()`) can serve as Tier 1 during development, with the understanding that it requires historical data and only catches exact or near-exact replays.

### 10.4 What Must Be Newly Built

- Heuristic replay features (spectral bandwidth, channel signature, temporal structure)
- ML replay detection model architecture
- Training pipeline using ASVspoof datasets
- Integration with composite anti-spoof score

---

## 11. Deepfake/Synthetic Speech Detection

### 11.1 Current State

**Not implemented.** No component in the current codebase detects AI-generated synthetic speech.

### 11.2 Target Architecture

```
Input: audio mel-spectrogram + anti-spoof feature vector
    |
    v
+-------------------------------------------+
|  Deepfake Detection Classifier            |
|                                            |
|  Option A: CNN-based                       |
|  +-- Conv1D blocks on mel-spectrogram      |
|  +-- Global average pooling                |
|  +-- FC layers -> binary output            |
|  +-- ~500K parameters                      |
|                                            |
|  Option B: wav2vec2-based (if GPU avail.)  |
|  +-- Pretrained wav2vec2 feature extractor |
|  +-- Fine-tuned classifier head            |
|  +-- Higher accuracy, higher latency       |
|                                            |
|  Output:                                   |
|  +-- probability: float [0, 1]             |
|  |   0.0 = likely live                     |
|  |   1.0 = likely synthetic                |
|  +-- confidence: float [0, 1]              |
+-------------------------------------------+
```

### 11.3 Training Data Sources

| Dataset | Content | Use |
|---------|---------|-----|
| ASVspoof 2021 | Logical access (TTS/VC) + physical access | Main training set |
| In-the-Wild | Real-world deepfake audio | Test set / fine-tuning |
| Wavefake | Multiple TTS system outputs | Synthetic positive samples |
| Custom collected | Live speech recordings | Genuine positive samples |

### 11.4 Key Detection Signals

- Spectral discontinuities at frame boundaries (common in neural vocoders)
- Unnatural periodicity in high-frequency bands
- Phase spectrum anomalies
- Cepstral variance patterns inconsistent with natural speech
- Absence of natural breathing artifacts
- Prosodic flatness (monotone delivery typical of TTS)

---

## 12. Voice Conversion/Mimicry Detection

### 12.1 Current State

**Not implemented.** No component detects voice conversion or speaker mimicry.

### 12.2 Target Architecture

```
Input: audio + speaker embedding from CNN-LSTM-Attention
    |
    v
+-----------------------------------------------+
|  Voice Conversion Detection                    |
|                                                |
|  Check 1: Embedding Consistency                |
|  +-- Compare CNN-LSTM embedding with           |
|  |   parallel traditional feature vector       |
|  |   (GMM-based i-vector or x-vector)         |
|  +-- If embeddings disagree significantly,     |
|  |   one may be spoofed                        |
|  +-- Threshold: embedding distance > tau       |
|                                                |
|  Check 2: Prosodic Naturalness                 |
|  +-- Pitch contour smoothness                  |
|  +-- Energy contour naturalness                |
|  +-- Speaking rate consistency                 |
|  +-- Unnatural prosody -> flag as converted    |
|                                                |
|  Check 3: Spectral-Temporal Coherence          |
|  +-- Joint analysis of spectral envelope       |
|  |   and temporal fine structure               |
|  +-- Converted speech shows inconsistencies    |
|  +-- Convolutional artifact detection          |
|                                                |
|  Composite score -> feeds into anti-spoof      |
+-----------------------------------------------+
```

### 12.3 What Must Be Newly Built

All components -- this is entirely new functionality:

- Embedding consistency checking
- Prosodic naturalness scoring
- Spectral-temporal coherence analysis
- Training data collection pipeline (VC system outputs vs. genuine)

---

## 13. Dynamic Challenge-Response Liveness

### 13.1 Current State

`AntiSpoofingDetector.generate_challenge()` (`app.py:253-258`) generates a phrase + 4-digit number displayed to the user. The `challenge_text` parameter passed to `authenticate()` (`app.py:392-393`) is never read or validated. The user can say anything and still pass.

### 13.2 Target Architecture

```
Server-Side:
    |
    +-- Generate challenge:
    |   +-- Random phrase from large vocabulary (100+ phrases)
    |   +-- Optional: random digits / letters
    |   +-- Store challenge_id + expected_text + expiry (30s)
    |   +-- Send challenge_id + display_text to client

Client-Side:
    |
    +-- Display challenge text to user
    +-- User speaks challenge phrase
    +-- Upload audio + challenge_id to server

Server-Side:
    |
    +-- Retrieve expected_text for challenge_id
    +-- Run speech-to-text on user audio:
    |   +-- Option A: Whisper (local, higher accuracy)
    |   +-- Option B: Whisper API (external, easier)
    |   +-- Option C: Simpler ASR model (lighter)
    |
    +-- Compare ASR output with expected_text:
    |   +-- Word Error Rate (WER) calculation
    |   +-- WER < 0.2 -> challenge passed
    |   +-- WER 0.2-0.4 -> challenge marginal (weighted)
    |   +-- WER > 0.4 -> challenge failed
    |
    +-- Challenge score feeds into composite anti-spoof score
```

### 13.3 Challenge Vocabulary

Replace the current 5 hardcoded phrases:

```
Challenge Types:
    +-- Phrase challenges:
    |   +-- 100+ common phrases
    |   +-- Randomly selected per attempt
    |   +-- Phrase changes every authentication
    |
    +-- Number challenges:
    |   +-- "Say the numbers: 7 3 9 2"
    |   +-- Random length (3-6 digits)
    |   +-- Digit verification via ASR
    |
    +-- Combined challenges:
    |   +-- "Say 'pay' followed by 3 8 1"
    |   +-- "Repeat: merchant verification active"
    |
    +-- Time-based:
        +-- Challenge expires after 30 seconds
        +-- Same challenge cannot be reused
        +-- Rate limit: 3 challenges per minute
```

### 13.4 What Can Be Reused

| Component | Source | V2 Action |
|-----------|--------|-----------|
| `generate_challenge()` | `app.py:253-258` | **Replace** -- expand vocabulary, add expiry, add challenge_id |
| Challenge display on PaymentScreen | `app.py:1296-1298` | **Reuse** display pattern, **extend** with challenge_id |

### 13.5 What Must Be Newly Built

- Challenge generation service with expiry and uniqueness
- Speech-to-text integration (Whisper or alternative)
- Word Error Rate calculation
- Challenge-response score integration with anti-spoof pipeline

---

## 14. Biometric Template Protection

### 14.1 Current State

`SecureDatabase` (`app.py:59-106`) encrypts user data with Fernet symmetric encryption. Key is generated fresh each run. All data in-memory. No transformation of biometric data before storage.

### 14.2 Target Architecture

```
Enrollment:
    |
    +-- Raw audio -> features -> embedding (256-dim float32)
    |
    +-- Apply cancelable transform:
    |   +-- User-specific random projection matrix R_u
    |   |   (derived from user's secret key + salt)
    |   +-- transformed_embedding = normalize(R_u . embedding)
    |   +-- R_u derived from: PBKDF2(user_password, salt)
    |
    +-- Store:
    |   +-- transformed_embedding (for matching)
    |   +-- user_id
    |   +-- salt (per-user, random)
    |   +-- enrollment_timestamp
    |   +-- model_version
    |
    +-- NEVER store: raw audio, raw embeddings, feature vectors

Authentication:
    |
    +-- Raw audio -> features -> embedding
    +-- Derive R_u from user's key + stored salt
    +-- transformed_embedding = normalize(R_u . embedding)
    +-- Compare transformed_embedding against stored template
```

### 14.3 Encryption Layers

```
Layer 1: Transport (TLS 1.3)
    +-- All client-server communication encrypted in transit

Layer 2: Database Encryption (at rest)
    +-- Column-level encryption for biometric data
    +-- AES-256-GCM for encrypted columns
    +-- Key managed externally (env var / Vault)
    +-- Key rotation policy (90-day)

Layer 3: Template Protection (cancelable biometrics)
    +-- Per-user random projection for embeddings
    +-- Derived from user credential + salt
    +-- Enables re-enrollment with new projection if compromised

Layer 4: GMM Model Protection
    +-- GMM parameters stored encrypted
    +-- Model file encrypted with server key
    +-- Decrypted only in memory during inference
```

### 14.4 What Can Be Reused

| Component | Source | V2 Action |
|-----------|--------|-----------|
| Fernet encryption | `app.py:68-81` | **Reuse** for general data; **add** AES-256-GCM for biometric columns |
| PBKDF2 key derivation | `app.py:50-51` (imported, unused) | **Activate** for cancelable transform derivation |

### 14.5 What Must Be Newly Built

- Cancelable biometric transform (random projection with per-user salt)
- Column-level database encryption
- Key management integration
- Template lifecycle management (enrollment, update, revocation)
- Biometric data retention policy enforcement

---

## 15. User Authentication/Session Architecture

### 15.1 Current State

- Login requires only username + voice recording (`app.py:747-778`)
- No password, no PIN, no multi-factor
- No session tokens, no session timeout
- `pyotp` imported (`app.py:52`) but never used
- Navigation between screens unrestricted

### 15.2 Target Architecture

```
Registration Flow:
    |
    +-- 1. User provides: username, email, password
    |   (password hashed with Argon2id, stored in DB)
    |
    +-- 2. User records N voice samples (N >= 5)
    |   +-- Quality-gated
    |   +-- Feature extraction -> embedding generation
    |   +-- Cancelable transform applied
    |   +-- Template stored
    |
    +-- 3. Account activated

Login Flow:
    |
    +-- 1. User provides: username + password
    |   +-- Password verified against Argon2id hash
    |   +-- If password fails -> reject
    |
    +-- 2. If password correct -> voice authentication
    |   +-- Challenge-response phrase displayed
    |   +-- User speaks phrase
    |   +-- Anti-spoofing checks pass
    |   +-- Speaker verification passes
    |   +-- Challenge-response validation passes
    |
    +-- 3. Issue JWT access token (15 min expiry)
    |   +-- Issue refresh token (7 day expiry, stored in DB)
    |
    +-- 4. Return tokens to client

Session Management:
    |
    +-- Access token: short-lived JWT, in memory on client
    +-- Refresh token: long-lived, stored HTTP-only cookie
    +-- Token refresh: automatic before expiry
    +-- Session invalidation: on logout, password change, compromise
    +-- Concurrent session limit: configurable (default: 3)

Transaction Authorization (within active session):
    |
    +-- User already authenticated (has valid JWT)
    +-- For transactions above threshold:
    |   +-- Re-authenticate with voice (different challenge)
    |   +-- Anti-spoofing + speaker verification
    |
    +-- For transactions below threshold:
        +-- PIN or biometric confirmation (future)
```

### 15.3 What Can Be Reused

| Component | Source | V2 Action |
|-----------|--------|-----------|
| Auth attempt tracking | `app.py:88-105` | **Reuse** logic, move to Redis-backed |
| `pyotp` import | `app.py:52` | **Activate** for TOTP backup MFA |
| Enrollment flow (5 samples) | `app.py:356-390` | **Reuse** core, **add** password step |

### 15.4 What Must Be Newly Built

- Password-based authentication (Argon2id hashing)
- JWT access/refresh token system
- Session management (creation, refresh, invalidation)
- Multi-factor authentication flow (voice + password)
- TOTP backup authentication
- Rate limiting per-user and per-IP
- Account lockout after N failed attempts
- Session timeout enforcement

---

## 16. Transaction Authorization Architecture

### 16.1 Current State

`PaymentScreen.process_payment()` (`app.py:1354-1413`): validates form, calls `auth_engine.authenticate()`, deducts from local `self.user_balance` variable, logs to in-memory list. No atomic operations, no server validation, no transaction IDs.

### 16.2 Target Architecture

```
Transaction Request:
    |
    +-- Client sends:
    |   +-- recipient_id, amount, description
    |   +-- audio recording (WAV/FLAC bytes)
    |   +-- challenge_id (from current challenge)
    |   +-- request_id (client-generated UUID, for idempotency)
    |   +-- JWT access token
    |
    +-- Server validates:
    |   +-- JWT is valid and not expired
    |   +-- Request idempotency (check request_id not seen)
    |   +-- Rate limit check (max 10 transactions/hour)
    |   +-- Amount within user's daily limit
    |
    +-- Server processes:
    |   +-- Voice pipeline: preprocess -> extract -> embed -> anti-spoof -> verify
    |   +-- If voice verification fails -> reject, log attempt
    |   +-- If anti-spoofing fails -> reject, flag account
    |   |
    |   +-- If all checks pass:
    |   |   +-- BEGIN TRANSACTION (PostgreSQL)
    |   |   +-- Debit sender: UPDATE users SET balance = balance - amount
    |   |   |   WHERE balance >= amount AND id = user_id
    |   |   +-- Credit recipient: UPDATE users SET balance = balance + amount
    |   |   +-- INSERT INTO transactions
    |   |   +-- COMMIT
    |   |
    |   +-- Return:
    |       +-- transaction_id
    |       +-- new_balance
    |       +-- voice_score
    |       +-- status: "completed" | "declined"
    |
    +-- Client displays result
```

### 16.3 What Can Be Reused

| Component | Source | V2 Action |
|-----------|--------|-----------|
| Form validation | `app.py:1415-1433` | **Reuse** -- move to Pydantic schemas |
| Balance display format | `app.py:1149, 1287` | **Reuse** formatting |
| Transaction log fields | `app.py:1376-1383` | **Reuse** fields, map to DB schema |
| Success/decline popups | `app.py:1459-1600` | **Reuse** UI pattern |
| Transaction history | `app.py:1448-1457` | **Reuse** display, paginate |

### 16.4 What Must Be Newly Built

- Server-side transaction processing with ACID guarantees
- Idempotency (prevent duplicate transactions)
- Rate limiting per-user
- Daily transaction limits
- Transaction ID generation (UUID v4)
- Balance atomic operations (database-level)
- Audit logging for every transaction attempt

---

## 17. Database Schema

### 17.1 Current State

No database. All data in Python dicts. Lost on every restart.

### 17.2 Target PostgreSQL Schema

```sql
-- Users table
CREATE TABLE users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username        VARCHAR(32) UNIQUE NOT NULL,
    email           VARCHAR(255) UNIQUE NOT NULL,
    password_hash   VARCHAR(255) NOT NULL,
    display_name    VARCHAR(64),
    is_active       BOOLEAN DEFAULT true,
    is_locked       BOOLEAN DEFAULT false,
    daily_limit     DECIMAL(12,2) DEFAULT 50000.00,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

-- Voice templates (biometric data)
CREATE TABLE voice_templates (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID NOT NULL REFERENCES users(id),
    model_version       VARCHAR(32) NOT NULL,
    template_data       BYTEA NOT NULL,
    enrollment_samples  INT NOT NULL,
    quality_scores      JSONB,
    salt                BYTEA NOT NULL,
    is_active           BOOLEAN DEFAULT true,
    created_at          TIMESTAMPTZ DEFAULT now()
);

-- Voice models (GMM parameters)
CREATE TABLE voice_models (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id),
    template_id     UUID NOT NULL REFERENCES voice_templates(id),
    model_type      VARCHAR(32) NOT NULL,
    model_data      BYTEA NOT NULL,
    parameters      JSONB,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Sessions
CREATE TABLE sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id),
    refresh_token   VARCHAR(512) UNIQUE NOT NULL,
    user_agent      TEXT,
    ip_address      INET,
    expires_at      TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now(),
    revoked_at      TIMESTAMPTZ
);

-- Transactions
CREATE TABLE transactions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sender_id       UUID NOT NULL REFERENCES users(id),
    recipient_id    UUID,
    recipient_name  VARCHAR(128) NOT NULL,
    amount          DECIMAL(12,2) NOT NULL CHECK (amount > 0),
    currency        CHAR(3) DEFAULT 'INR',
    description     TEXT,
    status          VARCHAR(16) NOT NULL,
    voice_score     DECIMAL(5,4),
    challenge_id    UUID,
    request_id      UUID UNIQUE,
    decline_reason  TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Audit log
CREATE TABLE audit_log (
    id              BIGSERIAL PRIMARY KEY,
    user_id         UUID REFERENCES users(id),
    event_type      VARCHAR(64) NOT NULL,
    event_detail    JSONB,
    ip_address      INET,
    user_agent      TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Challenges
CREATE TABLE challenges (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id),
    challenge_text  VARCHAR(128) NOT NULL,
    challenge_type  VARCHAR(32) NOT NULL,
    is_used         BOOLEAN DEFAULT false,
    expires_at      TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Auth attempts
CREATE TABLE auth_attempts (
    id              BIGSERIAL PRIMARY KEY,
    user_id         UUID REFERENCES users(id),
    attempt_type    VARCHAR(16) NOT NULL,
    success         BOOLEAN NOT NULL,
    failure_reason  TEXT,
    ip_address      INET,
    created_at      TIMESTAMPTZ DEFAULT now()
);
```

### 17.3 Redis Schema

```
Session cache:
    session:{user_id}:{session_id} -> JWT payload (TTL: 15 min)

Rate limiting:
    ratelimit:{user_id}:voice -> counter (TTL: 1 hour)
    ratelimit:{user_id}:transaction -> counter (TTL: 1 hour)
    ratelimit:{ip}:login -> counter (TTL: 15 min)

Attempt tracking:
    attempts:{user_id}:failed -> counter (TTL: 30 min)

Transaction cache:
    tx-history:{user_id} -> JSON array (TTL: 60 sec)

Challenge nonce:
    challenge:{challenge_id} -> {text, user_id, expires_at} (TTL: 30 sec)
```

---

## 18. Model Storage/Versioning

### 18.1 Current State

Models exist only in memory. `VoiceEmbeddingNet` initialized at startup (random weights). `GaussianMixture` trained per-user at enrollment. `OneClassSVM` initialized but never used. No model files, no versioning, no persistence.

### 18.2 Target Architecture

```
Model Registry:
    |
    +-- Global models (shared across all users):
    |   +-- speaker_embedding_net.pt      (CNN-LSTM-Attention weights)
    |   +-- anti_spoof_replay.pt          (replay detection model)
    |   +-- anti_spoof_deepfake.pt        (deepfake detection model)
    |   +-- background_gmm.pt             (universal background model)
    |
    +-- Per-user models (stored encrypted):
    |   +-- user_{id}_gmm.pkl             (GMM parameters)
    |   +-- user_{id}_template.npy        (transformed enrollment embedding)
    |
    +-- Versioning:
        +-- Each model tagged with version (v1.0, v1.1, ...)
        +-- Model metadata in DB: version, training date, metrics
        +-- A/B testing: serve two versions simultaneously
        +-- Rollback: switch to previous version
```

### 18.3 Model File Storage

```
Object Storage (MinIO / S3):
    voiceguard-models/
    +-- global/
    |   +-- speaker_embedding/
    |   |   +-- v1.0/model.pt, config.json, metrics.json
    |   |   +-- v1.1/model.pt, config.json, metrics.json
    |   +-- anti_spoof/
    |   |   +-- replay/v1.0/...
    |   |   +-- deepfake/v1.0/...
    |   +-- background_gmm/v1.0/...
    +-- users/
        +-- {user_id}/
            +-- gmm_v1.0.pkl.enc
            +-- template_v1.0.npy.enc
```

### 18.4 What Must Be Newly Built

- Model registry (DB table + object storage)
- Model loading/caching service
- Version management (promotion, rollback)
- Model artifact encryption at rest
- Model performance monitoring
- Automated retraining triggers

---

## 19. API Boundaries

### 19.1 Current State

No API layer exists. All logic is in-process.

### 19.2 Target REST API

```
Authentication:
    POST   /api/v1/auth/register          -> Register new user
    POST   /api/v1/auth/login-password    -> Password login (step 1)
    POST   /api/v1/auth/login-voice       -> Voice auth (step 2)
    POST   /api/v1/auth/refresh           -> Refresh access token
    POST   /api/v1/auth/logout            -> Invalidate session

Voice Enrollment:
    POST   /api/v1/voice/enroll           -> Upload N audio samples
    GET    /api/v1/voice/status           -> Get enrollment status
    POST   /api/v1/voice/re-enroll        -> Re-enroll

Transactions:
    POST   /api/v1/transactions           -> Create (with voice auth)
    GET    /api/v1/transactions           -> List (paginated)
    GET    /api/v1/transactions/{id}      -> Get detail
    GET    /api/v1/transactions/balance   -> Get balance

Users:
    GET    /api/v1/users/me               -> Get profile
    PUT    /api/v1/users/me               -> Update profile
    PUT    /api/v1/users/me/password      -> Change password

Challenges:
    POST   /api/v1/challenges/generate    -> Generate challenge
    GET    /api/v1/challenges/{id}        -> Get challenge

Health:
    GET    /api/v1/health                 -> Health check
    GET    /api/v1/health/ready           -> Readiness probe
```

### 19.3 Request/Response Examples

#### POST /api/v1/auth/login-voice

```json
// Request (multipart/form-data)
{
    "user_id": "uuid-or-username",
    "audio": "<binary WAV/FLAC data>",
    "challenge_id": "uuid",
    "device_id": "optional-device-identifier"
}

// Response (200 OK)
{
    "status": "authenticated",
    "access_token": "eyJhbGciOi...",
    "refresh_token": "eyJhbGciOi...",
    "expires_in": 900,
    "voice_score": 0.87,
    "anti_spoof_score": 0.95
}

// Response (401 Unauthorized)
{
    "status": "rejected",
    "reason": "voice_match_insufficient",
    "detail": "Voice match score 0.61 below threshold 0.82",
    "voice_score": 0.61,
    "attempts_remaining": 3
}
```

### 19.4 API Design Principles

| Principle | Implementation |
|-----------|---------------|
| Idempotency | `request_id` in POST requests; duplicate IDs return same result |
| Pagination | `?page=1&limit=20`; response includes `total_count`, `has_more` |
| Error format | `{"error": "code", "detail": "message", "field": "optional_field"}` |
| Versioning | URL prefix `/api/v1/` |
| Rate limiting | Headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining` |
| Audio upload | Multipart; max 10MB; WAV/FLAC/OGG; max 30 seconds |

---

## 20. Security Boundaries

### 20.1 Current State

Security weaknesses: no password authentication, challenge-response not validated, encryption key not persisted, no rate limiting, no session management, thread safety issues, no input sanitization.

### 20.2 Target Security Architecture

```
+-------------------------------------------------------+
|                    SECURITY LAYERS                      |
|                                                        |
|  Layer 1: Transport Security                           |
|  +-- TLS 1.3 for all communication                    |
|  +-- Certificate pinning on mobile                     |
|  +-- HSTS headers                                      |
|                                                        |
|  Layer 2: Authentication                               |
|  +-- Multi-factor: password + voice biometric          |
|  +-- JWT access tokens (short-lived, signed)           |
|  +-- Refresh tokens (long-lived, stored in DB)         |
|  +-- Account lockout after 5 failed attempts           |
|                                                        |
|  Layer 3: Authorization                                |
|  +-- Users can only access own data                    |
|  +-- Admin endpoints require admin role                |
|  +-- CORS restricted to known origins                  |
|                                                        |
|  Layer 4: Input Validation                             |
|  +-- Pydantic schemas for all request bodies           |
|  +-- Audio format/size/duration validation             |
|  +-- SQL injection prevention (parameterized queries)  |
|                                                        |
|  Layer 5: Rate Limiting                                |
|  +-- Global: 100 req/min/IP                            |
|  +-- Auth: 5 login attempts/min/user                   |
|  +-- Voice: 10 verifications/min/user                  |
|  +-- Transactions: 10/hour/user                        |
|                                                        |
|  Layer 6: Data Protection                              |
|  +-- Biometric templates encrypted at rest (AES-256)   |
|  +-- Cancelable biometric transform                    |
|  +-- Passwords hashed with Argon2id                    |
|  +-- Secrets in environment variables / Vault          |
|  +-- No raw audio stored after processing              |
|                                                        |
|  Layer 7: Audit and Monitoring                         |
|  +-- Every auth attempt logged                         |
|  +-- Every transaction logged                          |
|  +-- Security events flagged                           |
|  +-- Anomaly detection on access patterns              |
|                                                        |
|  Layer 8: Anti-Spoofing                                |
|  +-- Challenge-response liveness verification          |
|  +-- Replay attack detection (heuristic + ML)          |
|  +-- Deepfake/synthetic speech detection               |
|  +-- Voice conversion detection                        |
+-------------------------------------------------------+
```

### 20.3 Threat Model

| Threat | Mitigation |
|--------|-----------|
| Stolen voice recording (replay) | Challenge-response (unique per session), replay detection ML |
| AI-generated voice (deepfake) | Deepfake detection classifier, anti-spoofing features |
| Voice conversion (impersonation) | VC detection, embedding consistency check |
| Database breach | Encrypted biometric templates, Argon2id hashes, cancelable transforms |
| Session hijacking | Short-lived JWT, refresh token rotation, device binding |
| Brute force login | Rate limiting, account lockout, progressive cooldown |
| Man-in-the-middle | TLS 1.3, certificate pinning |
| Audio interception | End-to-end encrypted audio upload |

---

## 21. Configuration/Secrets Management

### 21.1 Current State

All configuration hardcoded as literals throughout `app.py`: sample rate, feature dimensions, thresholds, phrases, balance amounts, encryption key generated fresh each run.

### 21.2 Target Configuration

```python
class Settings(BaseSettings):
    # Application
    APP_NAME: str = "VoiceGuard"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4

    # Database
    DATABASE_URL: str  # Required
    REDIS_URL: str = "redis://localhost:6379/0"

    # Security
    JWT_SECRET_KEY: str  # Required
    ENCRYPTION_KEY: str  # Required: base64 Fernet key

    # Voice Processing
    AUDIO_SAMPLE_RATE: int = 16000
    N_MFCC: int = 40
    N_MELS: int = 80
    HOP_LENGTH: int = 160
    N_FFT: int = 512

    # Speaker Verification
    EMBEDDING_DIM: int = 256
    ENROLLMENT_MIN_SAMPLES: int = 5
    VERIFICATION_THRESHOLD: float = 0.82
    GMM_N_COMPONENTS: int = 8

    # Anti-Spoofing
    CHALLENGE_EXPIRY_SECONDS: int = 30
    ANTI_SPOOF_THRESHOLD: float = 0.70

    # Rate Limiting
    RATE_LIMIT_AUTH_PER_MINUTE: int = 5
    RATE_LIMIT_TRANSACTION_PER_HOUR: int = 10

    # Account
    MAX_FAILED_ATTEMPTS: int = 5
    DAILY_TRANSACTION_LIMIT: float = 50000.00
    DEFAULT_BALANCE: float = 10000.00

    class Config:
        env_prefix = "VG_"
        env_file = ".env"
```

### 21.3 Secrets Management

| Secret | Storage | Rotation |
|--------|---------|----------|
| Database password | Environment variable | Every 90 days |
| JWT secret key | Environment variable | Every 90 days |
| Encryption key | Environment variable | Every 180 days |
| Redis password | Environment variable | Every 90 days |

Production: HashiCorp Vault or cloud secrets manager. Development: `.env` file (gitignored).

---

## 22. Logging/Audit Architecture

### 22.1 Current State

**No logging exists.** No `logging` import. No audit trail. Transaction logs are in a Python list lost on restart.

### 22.2 Target Architecture

```
Structured Logging (JSON format):

Application Log:
    +-- Timestamp
    +-- Level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    +-- Module / function name
    +-- Request ID (UUID, propagated through middleware)
    +-- User ID (if authenticated)
    +-- Message
    +-- Structured fields

Audit Log (separate stream):
    +-- Event type:
    |   +-- auth.login.success
    |   +-- auth.login.failure
    |   +-- auth.register.success
    |   +-- auth.voice.enroll
    |   +-- auth.voice.verify.success
    |   +-- auth.voice.verify.failure
    |   +-- auth.anti_spoof.replay_detected
    |   +-- auth.anti_spoof.deepfake_detected
    |   +-- auth.lockout.activated
    |   +-- transaction.created
    |   +-- transaction.declined
    |   +-- security.suspicious_activity
    |
    +-- Actor: user_id
    +-- IP address
    +-- User agent
    +-- Timestamp
    +-- Event detail (JSON)
    +-- Outcome (success/failure)

Voice Processing Log:
    +-- Request ID
    +-- Preprocessing duration (ms)
    +-- Feature extraction duration (ms)
    +-- Embedding generation duration (ms)
    +-- Anti-spoof scoring duration (ms)
    +-- Verification duration (ms)
    +-- Total pipeline duration (ms)
    +-- Audio quality metrics
    +-- Anti-spoof sub-scores
    +-- Final decision + score
```

### 22.3 What Must Be Newly Built

- All logging infrastructure (currently nonexistent)
- Structured logging framework
- Request ID middleware
- Audit event classification
- Log rotation and retention policy
- Alert rules for security events

---

## 23. Testing Strategy

### 23.1 Current State

**Zero tests exist.** No test files, no test configuration, no test framework in dependencies.

### 23.2 Target Testing Pyramid

```
                    +---------+
                    |  E2E    |  <- Few: critical user flows
                    | Tests   |
                    +---------+
                    |Integration| <- Moderate: API + DB + ML pipeline
                    | Tests    |
                    +---------+
                    |  Unit   |  <- Many: individual functions
                    | Tests   |
                    +---------+
```

### 23.3 Test Categories

| Category | Scope | Count Target |
|----------|-------|-------------|
| Feature Extraction | MFCC, mel, formants | 20+ |
| Preprocessing | VAD, normalization, noise reduction | 15+ |
| Speaker Embedding | CNN-LSTM-Attention forward pass | 10+ |
| Verification | Scoring logic, threshold decisions | 15+ |
| Anti-Spoofing | Replay, challenge-response, quality | 20+ |
| User Service | Registration, password, profile | 15+ |
| Auth Service | Login, session, JWT, rate limiting | 20+ |
| Transaction Service | Create, balance, idempotency | 15+ |
| API Routes | Validation, response format, status codes | 30+ |
| Database | Migrations, queries, constraints | 10+ |
| Security | Encryption, tokens, sanitization | 15+ |
| ML Training | Training loop, checkpointing, metrics | 10+ |
| Client | Screen navigation, recording, forms | 15+ |
| E2E | Full enrollment -> login -> payment | 5+ |

### 23.4 Test Data Strategy

```
Test Audio:
    fixtures/
    +-- enrollment/
    |   +-- user_a_sample_1.wav ... user_a_sample_5.wav
    |   +-- user_b_sample_1.wav ... user_b_sample_5.wav
    +-- authentication/
    |   +-- user_a_genuine.wav
    |   +-- user_a_impostor.wav
    |   +-- replay_of_user_a.wav
    +-- anti_spoofing/
        +-- genuine_samples/
        +-- replay_samples/
        +-- deepfake_samples/
        +-- low_quality_samples/
```

---

## 24. ML Training/Evaluation Strategy

### 24.1 Current State

`VoiceEmbeddingNet`: random weights, never trained. No training data, loss function, optimizer, or training scripts. `OneClassSVM`: instantiated, never trained.

### 24.2 Training Pipeline Architecture

```
+----------------------------------------------+
|  Training Pipeline                             |
|                                                |
|  +----------------+                           |
|  | Data Pipeline   |                           |
|  | +-- Dataset     | VoxCeleb1/2, LibriSpeech |
|  | +-- Sampler     | Speaker-balanced          |
|  | +-- Augment     | Noise, speed, reverb     |
|  | +-- Split       | Train / Val / Test       |
|  +-------+--------+                           |
|          |                                     |
|  +-------v--------+                           |
|  | Model           |                           |
|  | CNN-LSTM-Attn   |                           |
|  +-------+--------+                           |
|          |                                     |
|  +-------v--------+                           |
|  | Loss Function   |                           |
|  | Triplet Loss    | margin=0.2               |
|  | + ArcFace Loss  | s=30, m=0.3             |
|  +-------+--------+                           |
|          |                                     |
|  +-------v--------+                           |
|  | Optimizer        |                           |
|  | Adam             | lr=1e-3, wd=1e-5        |
|  | CosineAnnealing  | T_max=50 epochs         |
|  +-------+--------+                           |
|          |                                     |
|  +-------v--------+                           |
|  | Checkpointing    |                           |
|  | +-- Best model   | (by validation EER)      |
|  | +-- Final model  |                           |
|  +----------------+                           |
+----------------------------------------------+
```

### 24.3 Training Phases

**Phase 1: Speaker Embedding Training (Global Model)**

```
Dataset: VoxCeleb1 + VoxCeleb2 (~7000 speakers, ~1M utterances)
Loss: Triplet loss (semi-hard) + ArcFace classification
Batch size: 64
Epochs: 100
LR: 1e-3 -> 1e-5 (cosine annealing)
Augmentation: noise (0-20dB), speed (0.9-1.1x), reverb, time/freq masking
Target: EER < 3% on VoxCeleb1 test
Hardware: single GPU (RTX 3080 or equivalent)
```

**Phase 2: Anti-Spoofing Model Training**

```
Replay Detection:
    Dataset: ASVspoof 2019/2021
    Architecture: 1D CNN (3 blocks) + FC
    Loss: Binary cross-entropy

Deepfake Detection:
    Dataset: ASVspoof 2021 (logical access) + Wavefake
    Architecture: 1D CNN or fine-tuned wav2vec2
    Target: AUC-ROC > 0.95
```

**Phase 3: Per-User Enrollment**

```
At enrollment time:
    +-- Collect N voice samples (N >= 5)
    +-- Extract features -> generate embeddings
    +-- Apply cancelable transform
    +-- Store transformed template
    +-- Train per-user GMM (8 components)
    +-- Compute intra-class statistics
    +-- Compute per-user adaptive threshold

Ongoing:
    +-- After each successful auth, optionally update enrollment
    +-- Periodic re-enrollment prompt (every 6 months)
    +-- Anomaly detection: trending scores -> prompt re-enrollment
```

### 24.4 Evaluation Metrics

| Domain | Metrics |
|--------|---------|
| Speaker Verification | EER, TAR@FAR=0.01, TAR@FAR=0.001, DET curve |
| Anti-Spoofing | AUC-ROC, ACER, BPCER, APCER |
| System | End-to-end accuracy, FAR, FRR, pipeline latency (p50/p95/p99), throughput |

---

## 25. Deployment Architecture

### 25.1 Current State

`.devcontainer/devcontainer.json` runs `streamlit run app.py` -- wrong framework (app is Kivy, not Streamlit). No Dockerfile, no docker-compose, no CI/CD, no environment handling.

### 25.2 Target Deployment

```
Development:
    docker-compose.yml (local dev)
    +-- voiceguard-api (FastAPI)
    +-- postgres:16
    +-- redis:7
    +-- minio (S3-compatible)

    Updated DevContainer:
    +-- Python 3.11 + PostgreSQL client + Redis client
    +-- Run: uvicorn voiceguard.main:app

    Local Desktop Client:
    +-- python -m voiceguard.app
    +-- Connects to localhost:8000

Staging:
    Docker containers on single VM
    +-- API server (2 workers)
    +-- PostgreSQL
    +-- Redis
    +-- Nginx reverse proxy
    +-- SSL: Let's Encrypt

Production:
    Option A: Cloud-Managed
    +-- API: ECS Fargate / Cloud Run / AKS
    +-- DB: RDS PostgreSQL / Cloud SQL
    +-- Cache: ElastiCache Redis / Memorystore
    +-- Storage: S3 / GCS

    Option B: Self-Hosted
    +-- Docker Swarm or Kubernetes
    +-- PostgreSQL (replicated)
    +-- Redis (sentinel)
    +-- MinIO cluster
```

### 25.3 CI/CD Pipeline

```
Push to main:
    +-- Lint (ruff) -> Type check (mypy) -> Security scan (bandit)
    +-- Unit tests -> Integration tests
    +-- Build Docker image -> Push to registry
    +-- Deploy to staging (automatic)

Tag release (v*):
    +-- All staging checks pass
    +-- Manual approval gate
    +-- Deploy to production (blue-green)
    +-- Smoke tests

Pull request:
    +-- All tests -> Coverage > 80% -> Code review -> Merge
```

### 25.4 DevContainer Update

```json
{
    "name": "VoiceGuard V2",
    "image": "mcr.microsoft.com/devcontainers/python:1-3.11-bookworm",
    "features": {
        "ghcr.io/devcontainers/features/docker-in-docker:2": {}
    },
    "forwardPorts": [8000, 5432, 6379],
    "postCreateCommand": "pip install -e '.[dev]' && docker compose up -d && alembic upgrade head"
}
```

---

## 26. Android/Mobile Strategy

### 26.1 Current State

No mobile application. The desktop Kivy app cannot run on mobile without significant rework. Kivy has an Android target but is not production-grade for mobile.

### 26.2 Target Strategy

**Phase 1: Desktop First (Current Priority)**

- Stabilize desktop client with backend integration
- Prove the voice biometric pipeline works end-to-end
- Collect real-world audio data for model improvement

**Phase 2: Android Native Client**

```
Android App (Kotlin):
    +-- Audio recording via Android AudioRecord API
    +-- On-device preprocessing (noise reduction, VAD)
    +-- Optional on-device inference (ONNX Runtime Mobile)
    +-- REST API communication with backend
    +-- BiometricPrompt integration for device-level biometrics
    +-- Secure enclave for key storage (Android Keystore)
    +-- Certificate pinning
```

**Phase 3: On-Device Inference**

```
Model Export:
    +-- PyTorch model -> ONNX -> TFLite / ONNX Runtime Mobile
    +-- Quantize to INT8 for mobile inference
    +-- Target: < 50ms inference on mid-range Android device
    +-- Model bundled with app (no network dependency for inference)

On-Device Pipeline:
    +-- Audio -> preprocess -> mel-spectrogram -> CNN-LSTM-Attention -> embedding
    +-- Embedding sent to server for matching (not raw audio)
    +-- Anti-spoofing: lightweight models on-device, full models on-server
    +-- Challenge-response: server-validated
```

### 26.3 Mobile-Specific Considerations

| Concern | Approach |
|---------|----------|
| Audio quality (varied microphones) | Adaptive preprocessing, quality feedback to user |
| Background noise | Enhanced VAD + noise reduction on-device |
| Battery usage | Efficient model architecture, batch inference |
| Network latency | On-device embedding, send 256-dim vector (1KB) instead of audio (100KB+) |
| Security | Android Keystore for keys, no raw audio stored, encrypted at rest |
| Fragmentation | Test on top 20 Android devices by market share |

---

## 27. Future Scalability

### 27.1 Horizontal Scaling

```
API Servers:
    +-- Stateless FastAPI instances behind load balancer
    +-- Scale: add instances as needed
    +-- Session state in Redis (not in-process)
    +-- ML inference: CPU-based, scales with instances

Database:
    +-- Read replicas for query scaling
    +-- Connection pooling (asyncpg + SQLAlchemy)
    +-- Partitioning: transactions by month, audit_log by year

Redis:
    +-- Redis Cluster for horizontal scaling
    +-- Sharding by user_id hash

Object Storage:
    +-- S3/MinIO scales automatically
    +-- CDN for model file distribution
```

### 27.2 ML Model Scaling

```
Model Serving Options:
    +-- Current: CPU inference in FastAPI process (sufficient for < 100 req/s)
    +-- Scale: dedicated inference servers (ONNX Runtime, Triton)
    +-- GPU inference if throughput demands > 1000 req/s
    +-- Edge deployment for mobile (see Section 26)

Model Update Strategy:
    +-- Blue-green model deployment
    +-- Shadow mode: run new model alongside old, compare results
    +-- Canary: route 5% traffic to new model, monitor metrics
    +-- Full rollout: after validation period
```

### 27.3 Geographic Scaling

```
Multi-Region:
    +-- Database replication across regions
    +-- Redis per-region (local sessions)
    +-- CDN for static assets
    +-- Model files replicated to each region
    +-- Regional API endpoints for low latency
```

### 27.4 Feature Roadmap

| Phase | Features | Dependencies |
|-------|----------|-------------|
| V2.0 | Desktop client, full backend, trained CNN-LSTM-Attention, anti-spoofing | This architecture |
| V2.1 | Web client (React), admin dashboard | V2.0 stable |
| V2.2 | Android native client | V2.0 stable |
| V2.3 | On-device inference (ONNX), offline mode | Android client |
| V2.4 | Continuous authentication, background voice monitoring | Model improvements |
| V2.5 | Multilingual voice models, transformer-based verification | Training infrastructure |
| V3.0 | Cloud-native deployment, multi-tenant, API marketplace | Production workload |

---

## Appendix A: Complete Migration Summary

### What Exists Today

| Component | File Location | Lines | Status |
|-----------|---------------|-------|--------|
| `SecureDatabase` | `app.py` | 59-106 | In-memory only, Fernet encryption (key not persisted) |
| `VoiceFeatureExtractor` | `app.py` | 109-218 | Functional but primitive preprocessing |
| `VoiceEmbeddingNet` | `app.py` | 222-238 | Architecture defined, random weights, never trained |
| `AntiSpoofingDetector` | `app.py` | 242-341 | Partial: quality checks + cosine replay detection; SVM dead code |
| `VoiceAuthenticationEngine` | `app.py` | 344-503 | Functional: enroll + authenticate with combined scoring |
| `PaytmButton` / `PaytmCard` / `VoiceWaveform` | `app.py` | 507-564 | Functional Kivy widgets |
| `LoginScreen` | `app.py` | 567-792 | Functional UI, no validation, duplicated recording |
| `RegistrationScreen` | `app.py` | 794-1105 | Functional 5-sample enrollment UI |
| `PaymentScreen` | `app.py` | 1108-1615 | Functional payment UI, challenge not validated |
| `VoiceGuardApp` | `app.py` | 1619-1646 | App entry point |

### What Must Be Replaced

- In-memory storage -> PostgreSQL + Redis
- Fernet key per run -> externally managed encryption key
- Hardcoded config -> Pydantic BaseSettings
- `VoiceEmbeddingNet` (random weights) -> trained CNN-LSTM-Attention
- Energy-only VAD -> multi-feature VAD
- Naive spectral subtraction -> adaptive noise reduction
- Graduated thresholds (lower for first attempt) -> single adaptive threshold
- Username-only login -> password + voice MFA
- Unvalidated challenge-response -> ASR-validated challenge-response
- Unstructured error handling -> structured HTTP error responses
- No logging -> structured logging + audit trail
- No tests -> comprehensive test suite
- Kivy-only desktop -> desktop + Android + web (future)
- Broken devcontainer -> working development environment

### What Can Be Reused (with modifications)

- Feature extraction logic (MFCC, mel-spectrogram, spectral, prosodic)
- Per-user GMM training and scoring
- Cosine similarity computation
- Challenge phrase generation (expanded)
- Audio quality checks (tightened thresholds)
- Replay detection heuristic (as Tier 1)
- Fernet encryption pattern (extended)
- Kivy widgets (renamed, extracted)
- Enrollment workflow (password-enhanced)
- Payment flow (server-validated)

### What Must Be Newly Built

- FastAPI backend with full middleware stack
- PostgreSQL schema + Alembic migrations
- JWT authentication + session management
- Password hashing (Argon2id) + password verification
- Cancelable biometric template protection
- CNN-LSTM-Attention model architecture
- Speaker embedding training pipeline
- Anti-spoofing ML models (replay, deepfake, voice conversion)
- Challenge-response validation via ASR
- Composite anti-spoof scoring system
- Model registry + version management
- Structured logging + audit framework
- Comprehensive test suite
- CI/CD pipeline
- Working Docker/DevContainer configuration
- Android client (future phase)
- On-device inference pipeline (future phase)
