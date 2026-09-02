# VoiceGuard V2 — Implementation Roadmap

**Date:** 2026-09-02
**Status:** Planning — not yet implemented
**Prerequisites:** [VOICEGUARD_CODEBASE_AUDIT.md](./VOICEGUARD_CODEBASE_AUDIT.md), [VOICEGUARD_V2_ARCHITECTURE.md](./VOICEGUARD_V2_ARCHITECTURE.md)

---

## Critical Path

The critical path through the implementation is:

```
Phase 1 (Project Structure)
  → Phase 2 (Database & Persistence)
    → Phase 3 (Backend API Foundation)
      → Phase 5 (Voice Preprocessing & Features)
        → Phase 6 (CNN-LSTM-Attention & Speaker Verification ML)
          → Phase 11 (Composite Anti-Spoofing)
            → Phase 12 (Client Refactoring)
              → Phase 13 (Integration & E2E)
                → Phase 14 (Deployment & CI/CD)
```

**The first phase to implement is Phase 1: Project Structure & Configuration Foundation.** Nothing else can begin until the monolith is decomposed into a multi-package project with a shared configuration system. Phase 2 (database) and Phase 5 (voice preprocessing) can begin in parallel once Phase 1 is complete, as they share no code dependencies.

Phases 7, 8, 9, and 10 (ML anti-spoofing modules) can proceed in parallel with each other once Phase 5 is complete, but all must finish before Phase 11 can integrate them.

---

## Phase 1: Project Structure & Configuration Foundation

### Objective
Decompose the single-file `app.py` monolith into a multi-package Python project and establish a centralized configuration system.

### Current Problem
The entire application (1,646 lines) lives in one file. All configuration is hardcoded as literals (sample rates, thresholds, dimensions, phrases, encryption settings). There is no `.gitignore`, no `pyproject.toml`, no package structure, and no environment variable handling. The `requirements.txt` is incorrect (lists `streamlit` which is unused; omits `kivy`, `sounddevice`, `soundfile`, `pyotp` which are required).

### Work to Perform
1. Create top-level project layout with `backend/` and `client/` packages, each with their own `pyproject.toml`.
2. Create `shared/` or `common/` package for code shared between backend and client (feature extraction, preprocessing utilities).
3. Extract all hardcoded constants into a Pydantic `BaseSettings` class with `VG_` env-prefix and `.env` file support.
4. Fix `requirements.txt` — remove phantom deps (`streamlit`, `noisereduce`), add missing deps (`kivy`, `sounddevice`, `soundfile`, `pyotp`). Split into `backend/requirements.txt` and `client/requirements.txt` or use `pyproject.toml` extras.
5. Create `.gitignore` for Python bytecode, IDE files, virtual environments, recordings, `.env`, `__pycache__`, `*.pyc`.
6. Move Fernet key generation to an externally managed secret (env var `VG_ENCRYPTION_KEY`), loaded at startup, with a fallback that raises a clear error in production.
7. Create `LICENSE` file (MIT, matching README claim).

### Files/Modules Affected
- New: `backend/pyproject.toml`
- New: `client/pyproject.toml`
- New: `backend/voiceguard/__init__.py`
- New: `backend/voiceguard/config.py` (Pydantic BaseSettings)
- New: `client/voiceguard/__init__.py`
- New: `client/voiceguard/config.py`
- New: `shared/__init__.py` or `backend/voiceguard/shared/__init__.py`
- New: `.gitignore`
- New: `LICENSE`
- Modified: `requirements.txt` (or removed in favor of `pyproject.toml`)
- Existing: `app.py` (read-only reference, not modified yet)

### Dependencies
- None (this is the starting phase).

### Tests
- Unit test: `Settings` loads from environment variables with correct defaults.
- Unit test: `Settings` raises `ValidationError` when required fields (`DATABASE_URL`, `JWT_SECRET_KEY`, `ENCRYPTION_KEY`) are missing.
- Unit test: `.gitignore` contains essential patterns.
- Lint: `ruff check` passes on all new files.
- Type check: `mypy` passes on config modules.

### Acceptance Criteria
1. `backend/` and `client/` are independent Python packages installable via `pip install -e .`.
2. All hardcoded constants from `app.py` (lines 134-135 for thresholds, 222-238 for network dims, etc.) are centralized in `config.py`.
3. `ruff check .` and `mypy backend/voiceguard/config.py` pass with zero errors.
4. `.gitignore` prevents common artifacts from being tracked.
5. `LICENSE` file exists and matches MIT.

### Definition of Done
- [ ] Multi-package project structure is created and documented.
- [ ] Configuration system loads from env vars with sane defaults.
- [ ] All new files pass lint and type checks.
- [ ] `.gitignore` and `LICENSE` are in place.
- [ ] No application source code in `app.py` has been modified.

### Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| Over-engineering package boundaries too early | Wasted effort if structure needs refactoring | Keep boundaries minimal; extract only config first |
| Pydantic BaseSettings version incompatibility | Import errors at startup | Pin pydantic version in `pyproject.toml` |
| Circular imports between packages | Build failures | Enforce strict dependency direction: shared → backend, shared → client |

---

## Phase 2: Database Layer & Persistence

### Objective
Replace in-memory Python dicts with PostgreSQL for durable storage and Redis for session/rate-limit caching.

### Current Problem
`SecureDatabase` (`app.py:59-106`) stores everything in Python dicts. All user data, voice models, and transaction history vanish on application restart. The Fernet encryption key is generated fresh each run (`app.py:68`), making encrypted data from previous runs unrecoverable. This is the most critical functional gap in the application.

### Work to Perform
1. Define SQLAlchemy async models matching the target schema (users, voice_templates, voice_models, sessions, transactions, audit_log, challenges, auth_attempts — see Architecture §17.2).
2. Set up Alembic migration pipeline with initial migration.
3. Implement `db/session.py` — async SQLAlchemy session factory with connection pooling.
4. Implement column-level encryption for biometric data (`voice_templates.template_data`, `voice_models.model_data`) using AES-256-GCM with the externally managed encryption key.
5. Implement Redis connection manager for session cache, rate limiting, attempt tracking, and challenge nonces.
6. Create data access layer (repository pattern) for each entity.
7. Write a `docker-compose.yml` with PostgreSQL 16, Redis 7, and MinIO for local development.

### Files/Modules Affected
- New: `backend/voiceguard/db/__init__.py`
- New: `backend/voiceguard/db/session.py`
- New: `backend/voiceguard/db/migrations/` (Alembic)
- New: `backend/voiceguard/models/__init__.py`
- New: `backend/voiceguard/models/user.py`
- New: `backend/voiceguard/models/voice_model.py`
- New: `backend/voiceguard/models/transaction.py`
- New: `backend/voiceguard/models/session.py`
- New: `backend/voiceguard/models/audit_log.py`
- New: `backend/voiceguard/models/challenge.py`
- New: `backend/voiceguard/models/auth_attempt.py`
- New: `backend/voiceguard/security/crypto.py` (AES-256-GCM for biometric columns)
- New: `docker-compose.yml`
- Modified: `backend/pyproject.toml` (add sqlalchemy, asyncpg, alembic, redis deps)

### Dependencies
- Phase 1 (config for `DATABASE_URL`, `REDIS_URL`, `ENCRYPTION_KEY`).

### Tests
- Unit test: Each SQLAlchemy model creates correctly with expected columns and constraints.
- Unit test: AES-256-GCM encrypt/decrypt round-trip for biometric data.
- Integration test: Alembic migration applies cleanly to a test database.
- Integration test: Repository CRUD operations (create user, read user, update balance) against a test database.
- Integration test: Redis set/get with TTL for session and rate-limit keys.

### Acceptance Criteria
1. `alembic upgrade head` creates all 8 tables without errors.
2. User enrollment data persists across application restarts.
3. Biometric templates are encrypted at rest and decrypt correctly during inference.
4. Redis TTL-based keys expire as expected.
5. `docker-compose up` starts PostgreSQL, Redis, and MinIO without errors.

### Definition of Done
- [ ] All 8 database tables created and migrated.
- [ ] Repository layer provides CRUD for all entities.
- [ ] Biometric column encryption/decryption verified.
- [ ] Docker Compose starts all infrastructure services.
- [ ] Integration tests pass against a real database.

### Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| Async SQLAlchemy complexity | Session management bugs | Use well-tested patterns from FastAPI + SQLAlchemy docs |
| Redis connection failures at startup | Application crashes | Implement retry with backoff; graceful degradation for non-critical caching |
| Alembic migration conflicts during parallel development | Broken migrations | Use `--autogenerate` carefully; review all migrations before merge |
| Biometric encryption key rotation breaks existing templates | Data loss | Implement key versioning; store key ID with encrypted data |

---

## Phase 3: Backend API Foundation (FastAPI)

### Objective
Build the FastAPI application with middleware stack, route structure, dependency injection, and Pydantic request/response schemas.

### Current Problem
No API layer exists. All logic is in-process inside the Kivy application. There is no separation between business logic and presentation. No request validation, no structured error responses, no rate limiting, no CORS configuration.

### Work to Perform
1. Create FastAPI application factory in `backend/voiceguard/main.py` with middleware: CORS, request-ID propagation, structured logging, rate limiting.
2. Implement Pydantic request/response schemas for all endpoints (auth, voice, transactions, users, challenges).
3. Implement dependency injection for DB sessions, service instances, auth context.
4. Implement route modules: `routes/auth.py`, `routes/users.py`, `routes/transactions.py`, `routes/voice.py`.
5. Implement service layer: `services/user_service.py`, `services/auth_service.py`, `services/transaction_service.py`, `services/voice_service.py`.
6. Implement health check endpoints (`/api/v1/health`, `/api/v1/health/ready`).
7. Implement structured error handling with HTTPException and consistent error response format.
8. Implement rate limiting middleware using Redis counters.

### Files/Modules Affected
- New: `backend/voiceguard/main.py`
- New: `backend/voiceguard/dependencies.py`
- New: `backend/voiceguard/routes/__init__.py`
- New: `backend/voiceguard/routes/auth.py`
- New: `backend/voiceguard/routes/users.py`
- New: `backend/voiceguard/routes/transactions.py`
- New: `backend/voiceguard/routes/voice.py`
- New: `backend/voiceguard/services/__init__.py`
- New: `backend/voiceguard/services/user_service.py`
- New: `backend/voiceguard/services/auth_service.py`
- New: `backend/voiceguard/services/transaction_service.py`
- New: `backend/voiceguard/services/voice_service.py`
- New: `backend/voiceguard/schemas/__init__.py`
- New: `backend/voiceguard/schemas/user.py`
- New: `backend/voiceguard/schemas/voice.py`
- New: `backend/voiceguard/schemas/transaction.py`
- New: `backend/voiceguard/security/__init__.py`
- New: `backend/voiceguard/security/rate_limit.py`
- Modified: `backend/pyproject.toml` (add fastapi, uvicorn, python-multipart deps)

### Dependencies
- Phase 1 (config, package structure).
- Phase 2 (database models, session management).
- Note: Voice routes will initially be stubs until Phase 5–6 provide the pipeline.

### Tests
- Unit test: Pydantic schemas reject invalid input (malformed email, negative amounts, missing fields).
- Unit test: Rate limiter increments and blocks after threshold.
- Integration test: `POST /api/v1/auth/register` creates a user (using a mock voice service).
- Integration test: `GET /api/v1/health` returns 200 with correct JSON body.
- Integration test: Structured error responses have consistent format (`error`, `detail`, optional `field`).
- Integration test: Request ID is propagated in response headers.

### Acceptance Criteria
1. `uvicorn voiceguard.main:app` starts without errors.
2. OpenAPI docs are accessible at `/docs`.
3. All route stubs accept requests and return appropriate status codes.
4. Rate limiting blocks requests exceeding configured thresholds.
5. Error responses follow the format `{"error": "code", "detail": "message"}`.
6. Request ID appears in every response header.

### Definition of Done
- [ ] FastAPI app starts and serves OpenAPI documentation.
- [ ] All route modules are wired to service layer.
- [ ] Dependency injection provides DB sessions and auth context.
- [ ] Rate limiting is functional via Redis.
- [ ] Structured error handling produces consistent responses.
- [ ] All unit and integration tests pass.

### Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| FastAPI async session lifecycle bugs | Database connection leaks | Use `async with` patterns; test with `pytest-asyncio` |
| Overly broad rate limits block legitimate traffic | Poor user experience | Start with generous limits; tune based on testing |
| Pydantic V2 migration differences | Schema validation bugs | Pin Pydantic version; use V2 syntax consistently |
| Voice route stubs create false confidence | Integration gaps | Mark stub routes clearly; add TODO comments with phase references |

---

## Phase 4: Authentication & Session Management

### Objective
Implement password-based authentication, JWT session tokens, and multi-factor auth flow (password + voice).

### Current Problem
Login requires only a username and voice recording (`app.py:747-778`). There is no password, no PIN, no multi-factor credential. `pyotp` is imported but never used (`app.py:52`). No session tokens exist — navigation between screens is unrestricted. The graduated threshold (0.72 first attempt, 0.80 subsequent) weakens security by making the first attempt the easiest to spoof.

### Work to Perform
1. Implement password hashing with Argon2id (via `argon2-cffi`).
2. Implement registration flow: username + email + password + voice enrollment.
3. Implement two-step login: (a) password verification, (b) voice authentication.
4. Implement JWT access tokens (15-min expiry) and refresh tokens (7-day expiry, stored in DB).
5. Implement session management: creation, refresh, invalidation, concurrent session limiting.
6. Implement account lockout after 5 failed attempts with escalating cooldown.
7. Implement TOTP backup authentication via `pyotp` (currently imported, unused).
8. Replace graduated thresholds with single adaptive threshold + confidence band.
9. Implement attempt limiting: max 5 per session, cooldown escalation (30s → 1m → 5m).

### Files/Modules Affected
- New: `backend/voiceguard/security/tokens.py` (JWT access/refresh)
- New: `backend/voiceguard/security/password.py` (Argon2id hashing)
- New: `backend/voiceguard/security/totp.py` (TOTP backup MFA)
- Modified: `backend/voiceguard/routes/auth.py` (implement registration + login endpoints)
- Modified: `backend/voiceguard/services/auth_service.py` (full auth logic)
- Modified: `backend/voiceguard/services/user_service.py` (password management)
- Modified: `backend/voiceguard/models/session.py` (refresh token storage)
- Modified: `backend/voiceguard/models/auth_attempt.py` (lockout tracking)

### Dependencies
- Phase 2 (database for users, sessions, auth_attempts tables).
- Phase 3 (FastAPI routes, service layer, dependency injection).
- Voice enrollment endpoint is a stub until Phase 6 provides the pipeline; password auth can be tested independently.

### Tests
- Unit test: Argon2id hash/verify round-trip.
- Unit test: JWT access token creation, validation, expiry detection.
- Unit test: Refresh token rotation (old token invalidated after use).
- Unit test: TOTP generation and verification.
- Unit test: Account lockout triggers after 5 failed attempts.
- Unit test: Adaptive threshold computes correctly from enrollment variance.
- Integration test: Full registration → password login → voice login → token refresh flow (voice step mocked initially).
- Integration test: Concurrent session limit enforced (4th session revokes oldest).

### Acceptance Criteria
1. Registration requires username, email, password, and 5+ voice samples.
2. Login requires correct password before voice authentication is attempted.
3. JWT access tokens expire after 15 minutes; refresh tokens expire after 7 days.
4. Account locks after 5 failed attempts; cooldown escalates.
5. TOTP backup codes work for authentication.
6. Adaptive threshold replaces graduated thresholds.

### Definition of Done
- [ ] Password-based authentication works end-to-end.
- [ ] JWT access/refresh token lifecycle is correct.
- [ ] Account lockout and cooldown function as specified.
- [ ] TOTP backup authentication works.
- [ ] Graduated thresholds are removed; adaptive threshold is active.
- [ ] All unit and integration tests pass.

### Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| Argon2id too slow on low-end hardware | Registration/login feels sluggish | Tune memory/time cost parameters; benchmark on target hardware |
| JWT secret key rotation invalidates sessions | Users logged out unexpectedly | Implement overlapping key validity; document rotation procedure |
| TOTP clock skew causes false rejections | Users cannot authenticate via backup | Allow ±1 time-step tolerance (30-second window) |
| Adaptive threshold too strict/lenient without calibration data | High FAR or FRR | Start with conservative threshold; calibrate after Phase 6 training data is available |

---

## Phase 5: Voice Preprocessing & Feature Extraction

### Objective
Replace the primitive preprocessing pipeline with robust noise reduction, multi-feature VAD, and structured feature extraction supporting three output paths (CNN mel-spectrogram, statistical feature vector, anti-spoofing features).

### Current Problem
`VoiceFeatureExtractor` (`app.py:109-218`) uses naive spectral subtraction (first 10% of frames as noise estimate), energy-only VAD (threshold 0.02), and discards most extracted features. `_prepare_feature_vector()` (`app.py:469-486`) concatenates only 4 fields and truncates/pads to 256 dimensions, losing spectral, prosodic, energy, and formant information. No anti-spoofing features are extracted.

### Work to Perform
1. Implement audio validation (sample rate, mono, non-empty, float32 range).
2. Implement amplitude normalization with clipping guard and near-silence handling.
3. Replace spectral subtraction with adaptive Wiener filter or spectral gating (noise profile from VAD-detected silence, not first 10%).
4. Implement multi-feature VAD: short-time energy + zero-crossing rate + spectral flux + combined threshold + morphological smoothing.
5. Implement silence trimming with 50ms margin and minimum voice duration check (0.5s).
6. Implement Path A feature extraction: variable-length mel-spectrogram (80 mels, log-scaled, n_fft=512, hop=160, f_min=50, f_max=8000).
7. Implement Path B feature extraction: statistical vector (MFCC 40 + delta + delta-delta → 240, spectral 4+ features, prosodic 4+ features, formants 6+ features, ~261 total).
8. Implement Path C anti-spoofing features: high-frequency energy ratio, spectral envelope smoothness, cepstral variance, phase spectrum statistics, jitter, shimmer, spectral tilt, group delay.
9. Delete `_prepare_feature_vector` pattern — CNN takes raw mel; GMM gets full statistical vector.

### Files/Modules Affected
- New: `backend/voiceguard/voice/__init__.py`
- New: `backend/voiceguard/voice/preprocessing.py` (validation, normalization, noise reduction, VAD, trimming)
- New: `backend/voiceguard/voice/features.py` (three extraction paths)
- New: `shared/voiceguard/__init__.py` (or `backend/voiceguard/shared/`)
- New: `shared/voiceguard/preprocessing.py` (shared preprocessing for client)
- New: `shared/voiceguard/features.py` (shared features for client)
- Modified: `backend/pyproject.toml` (add `librosa`, `scipy`, `numpy` deps if not already present)

### Dependencies
- Phase 1 (config for audio parameters: sample rate, n_mels, n_fft, hop_length, f_min, f_max).
- Independent of Phases 2–4 (no database or API dependency for preprocessing logic).

### Tests
- Unit test: Audio validation rejects wrong sample rate, stereo, empty, out-of-range inputs.
- Unit test: Amplitude normalization peaks at target dBFS without clipping.
- Unit test: Noise reduction improves SNR on synthetic noisy audio.
- Unit test: Multi-feature VAD detects speech segments correctly on clean and noisy audio.
- Unit test: Mel-spectrogram output shape is (80, T) for variable-length input.
- Unit test: Statistical feature vector has ~261 dimensions, no truncation/padding.
- Unit test: Anti-spoofing feature vector has expected dimensionality.
- Unit test: Pre-emphasis filter applied only in formant estimation path.
- Integration test: End-to-end preprocessing of a 5-second WAV file produces valid output for all three paths.

### Acceptance Criteria
1. Preprocessing handles audio at 16kHz mono float32 input.
2. Multi-feature VAD correctly segments speech from silence on test audio.
3. Mel-spectrogram output is variable-length, log-scaled, 80 mels.
4. Statistical feature vector contains all specified features without truncation.
5. Anti-spoofing feature vector includes all 8+ specified features.
6. No information loss from truncation or padding (unlike current 256-dim approach).

### Definition of Done
- [ ] All preprocessing steps implemented and tested.
- [ ] Three feature extraction paths produce correct output shapes.
- [ ] Shared preprocessing module is usable by both backend and client.
- [ ] All unit and integration tests pass.
- [ ] Processing time for 5-second audio is < 100ms on CPU.

### Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| WebRTC VAD dependency installation fails on some platforms | VAD unavailable | Implement multi-feature VAD as primary; WebRTC as optional fallback |
| librosa version incompatibility | Feature extraction breaks | Pin librosa version; test against pinned version |
| Anti-spoofing features insufficient for trained detectors | Phase 7–9 models underperform | Design features based on published literature; validate with domain experts |
| Shared preprocessing module creates tight coupling | Changes to backend break client | Use interface contracts; test shared module independently |

---

## Phase 6: ML — CNN-LSTM-Attention Model & Speaker Verification

### Objective
Implement, train, and deploy the CNN-LSTM-Attention speaker embedding model, and build the complete speaker verification scoring pipeline.

### Current Problem
`VoiceEmbeddingNet` (`app.py:222-238`) is a 3-layer FC network with random PyTorch weights that was never trained. Speaker embeddings are semantically meaningless. The scoring formula (`app.py:447`) combines cosine similarity and GMM score, but with random embeddings the system works (or fails) by accident. The GMM uses only 2 diagonal components (`app.py:488-495`), likely insufficient for speaker modeling. Graduated thresholds (0.72 first, 0.80 subsequent) weaken security.

### Work to Perform

#### Speaker Verification ML

1. Implement `CNNLSTMAttention` model architecture per Architecture §6.2:
   - CNN front-end: 3 Conv2d blocks (1→32→64→128) with BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d.
   - Bi-LSTM: 2-layer, input=128, hidden=256, dropout=0.3.
   - Attention pooling: additive (Bahdanau-style) with learned weight matrix.
   - Projection: Linear(256, 256) + L2 normalization.
2. Keep `VoiceEmbeddingNet` as `LegacyEmbeddingNet` for backward compatibility and comparison.
3. Implement training pipeline:
   - Dataset loader for VoxCeleb1/2 (speaker-balanced sampling).
   - Data augmentation: noise (0–20dB SNR), speed perturbation (0.9–1.1x), reverberation, time/freq masking.
   - Loss: Triplet loss (semi-hard, margin=0.2) + ArcFace classification loss (s=30, m=0.3).
   - Optimizer: Adam (lr=1e-3, weight_decay=1e-5) with cosine annealing (T_max=50 epochs).
   - Checkpointing: best model by validation EER, final model.
   - Target: EER < 3% on VoxCeleb1 test set.
4. Implement scoring pipeline in `voice/verification.py`:
   - Centroid + covariance computation at enrollment.
   - Cosine similarity to centroid, max similarity, min similarity.
   - Mahalanobis distance using enrollment covariance.
   - GMM log-likelihood ratio (increase to 8-component diagonal).
   - Configurable composite score weights.
   - Single adaptive threshold (replace graduated thresholds).
   - Confidence-band soft accept flow.
   - Attempt limiting with cooldown escalation (move to Redis-backed).
5. Implement model registry: version management, metadata storage, loading/caching.
6. Implement cancelable biometric transform (random projection with per-user salt derived from PBKDF2).

### Files/Modules Affected
- New: `backend/voiceguard/ml/__init__.py`
- New: `backend/voiceguard/ml/models/__init__.py`
- New: `backend/voiceguard/ml/models/cnn_lstm_attention.py`
- New: `backend/voiceguard/ml/models/embedding_net.py` (migrated `VoiceEmbeddingNet` as `LegacyEmbeddingNet`)
- New: `backend/voiceguard/ml/training/__init__.py`
- New: `backend/voiceguard/ml/training/train_speaker.py`
- New: `backend/voiceguard/ml/training/evaluate.py`
- New: `backend/voiceguard/ml/registry.py`
- New: `backend/voiceguard/voice/embedding.py`
- New: `backend/voiceguard/voice/verification.py`
- Modified: `backend/voiceguard/voice/features.py` (ensure mel-spectrogram path matches model input)
- Modified: `backend/voiceguard/services/voice_service.py` (wire up full pipeline)

### Dependencies
- Phase 1 (config for model hyperparameters: embedding_dim, n_mels, thresholds, GMM components).
- Phase 5 (mel-spectrogram extraction for CNN input; statistical features for GMM).
- Phase 2 (database for storing trained models, enrollment templates).
- Phase 3 (API routes for enrollment and verification endpoints).
- Training requires external dataset (VoxCeleb1/2) — must be obtained separately.

### Tests
- Unit test: `CNNLSTMAttention` forward pass produces (batch, 256) L2-normalized output.
- Unit test: `LegacyEmbeddingNet` forward pass produces (batch, 128) L2-normalized output.
- Unit test: Training loop runs for 1 epoch without error on synthetic data.
- Unit test: Triplet loss decreases with easy positive, hard negative.
- Unit test: ArcFace loss produces correct gradients.
- Unit test: Cosine similarity computation matches `scipy.spatial.distance.cosine`.
- Unit test: Mahalanobis distance computation is correct against manual calculation.
- Unit test: GMM training and scoring produce valid log-likelihood values.
- Unit test: Adaptive threshold computation from enrollment variance is correct.
- Unit test: Cancelable transform is deterministic given same user key + salt.
- Unit test: Cancelable transform with different keys produces different templates.
- Integration test: Enrollment → authentication round-trip with synthetic embeddings.
- Integration test: Impostor rejection (different user embeddings fail verification).
- Integration test: Model registry loads and serves model with version metadata.

### Acceptance Criteria
1. `CNNLSTMAttention` forward pass runs in < 50ms on CPU for a 3-second utterance.
2. Trained model achieves EER < 3% on VoxCeleb1 test set.
3. Enrollment stores transformed templates (not raw embeddings) in the database.
4. Verification correctly accepts genuine users and rejects impostors on test data.
5. GMM uses 8 diagonal components (up from 2).
6. Single adaptive threshold replaces graduated thresholds.
7. Model versioning allows rollback to previous model version.

### Definition of Done
- [ ] CNN-LSTM-Attention model implemented and architecture-verified.
- [ ] Training pipeline produces a model meeting EER < 3% target.
- [ ] Speaker verification scoring pipeline is complete with all scoring methods.
- [ ] Cancelable biometric transform is applied at enrollment and verification.
- [ ] Model registry supports versioning and rollback.
- [ ] All unit and integration tests pass.

### Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| VoxCeleb dataset too large to download in dev environment | Training blocked | Provide synthetic test data for CI; document dataset download steps |
| GPU not available for training | Training too slow | Support CPU training with smaller batch size; document GPU requirements |
| EER target not achievable with current architecture | Model inadequate | Tune hyperparameters; consider adding spectral attention or depth |
| Cancelable transform reduces matching accuracy | Higher FRR | Evaluate with/without transform; tune projection dimensions |
| Legacy model removal breaks existing test fixtures | Test failures | Keep legacy model as `--model legacy` option throughout V2 lifecycle |

---

## Phase 7: ML — Replay Attack Detection

### Objective
Implement a two-tier replay detection system: fast heuristic checks and a trained ML classifier.

### Current Problem
`AntiSpoofingDetector.detect_replay()` (`app.py:260-296`) computes cosine similarity between current and historical feature vectors. It requires 5+ historical samples (inactive during initial use), cannot detect replay through a different speaker or with room convolution, and is the sole anti-replay mechanism. The `OneClassSVM` (`app.py:242`) is instantiated but `.fit()` and `.predict()` are never called — dead code.

### Work to Perform
1. Implement Tier 1 heuristic replay features:
   - Spectral bandwidth consistency (live speech has natural variation).
   - Background noise profile match (replay introduces speaker's environment noise).
   - Channel frequency response signature (mic→speaker→mic adds coloration).
   - Temporal fine structure consistency (replay may have quantization artifacts).
   - Amplitude envelope naturalness.
2. Implement Tier 2 ML-based replay detector:
   - Architecture: 1D CNN (3 blocks) + FC binary classifier.
   - Input: mel-spectrogram + anti-spoof feature vector.
   - Output: binary (live vs. replay) + confidence score.
   - Training data: ASVspoof 2019/2021 logical and physical access.
   - Target inference: ~10ms on CPU.
3. Remove dead `OneClassSVM` code.
4. Integrate replay detector into the voice pipeline (called after feature extraction, before verification).

### Files/Modules Affected
- New: `backend/voiceguard/ml/models/anti_spoofing_replay.py`
- New: `backend/voiceguard/ml/training/train_replay.py`
- New: `backend/voiceguard/voice/anti_spoofing/__init__.py`
- New: `backend/voiceguard/voice/anti_spoofing/replay.py`
- Modified: `backend/voiceguard/voice/pipeline.py` (add replay detection stage)
- Removed: Dead `OneClassSVM` instantiation (from `app.py:242-250` pattern).

### Dependencies
- Phase 5 (anti-spoofing feature extraction, mel-spectrogram).
- Phase 6 (model architecture patterns, training pipeline structure).
- Training data: ASVspoof 2019/2021 dataset.

### Tests
- Unit test: Heuristic replay features are computed correctly for synthetic audio.
- Unit test: ML replay detector forward pass produces probability in [0, 1].
- Unit test: ML replay detector training runs for 1 epoch on synthetic data.
- Unit test: Known replay samples are classified with higher confidence than live samples.
- Integration test: Replay detector integrated into voice pipeline returns anti-spoof score.

### Acceptance Criteria
1. Tier 1 heuristic runs in < 5ms on CPU.
2. Tier 2 ML classifier runs in < 10ms on CPU.
3. Trained replay detector achieves AUC-ROC > 0.90 on ASVspoof test set.
4. Dead `OneClassSVM` code is removed.
5. Replay detection score is available as input to composite anti-spoof scoring.

### Definition of Done
- [ ] Heuristic replay features implemented and tested.
- [ ] ML replay detector trained and meeting AUC-ROC target.
- [ ] Replay detection integrated into voice pipeline.
- [ ] Dead code removed.
- [ ] All unit and integration tests pass.

### Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| ASVspoof dataset requires registration/download | Training blocked | Provide synthetic test data; document download instructions |
| Heuristic features insufficient for sophisticated replays | Low Tier 1 detection rate | ML tier compensates; tune heuristic weights empirically |
| Physical access replay (re-recorded) harder to detect | Lower accuracy on physical access attacks | Prioritize physical access in training data; augment with simulated room impulse responses |

---

## Phase 8: ML — Deepfake/Synthetic Speech Detection

### Objective
Build a neural classifier that detects AI-generated synthetic speech (TTS, neural vocoder output).

### Current Problem
**Not implemented.** No component in the current codebase detects AI-generated synthetic speech. A user could play a deepfake audio clip and bypass voice authentication entirely.

### Work to Perform
1. Implement deepfake detection classifier:
   - Option A (default): CNN-based — Conv1D blocks on mel-spectrogram, global average pooling, FC layers → binary output. ~500K parameters.
   - Option B (GPU optional): wav2vec2-based — pretrained feature extractor + fine-tuned classifier head. Higher accuracy, higher latency.
2. Training pipeline using ASVspoof 2021 (logical access — TTS/VC), Wavefake (multiple TTS systems), and custom collected genuine speech.
3. Key detection signals to leverage: spectral discontinuities at frame boundaries, unnatural high-frequency periodicity, phase spectrum anomalies, cepstral variance patterns, absence of breathing artifacts, prosodic flatness.
4. Output: probability [0, 1] (0 = likely live, 1 = likely synthetic) + confidence [0, 1].

### Files/Modules Affected
- New: `backend/voiceguard/ml/models/anti_spoofing_deepfake.py`
- New: `backend/voiceguard/ml/training/train_deepfake.py`
- New: `backend/voiceguard/voice/anti_spoofing/deepfake.py`
- Modified: `backend/voiceguard/voice/pipeline.py` (add deepfake detection stage)

### Dependencies
- Phase 5 (anti-spoofing feature extraction).
- Phase 6 (training pipeline patterns).
- Training data: ASVspoof 2021 logical access, Wavefake, custom genuine recordings.

### Tests
- Unit test: Deepfake classifier forward pass produces probability in [0, 1].
- Unit test: Training runs for 1 epoch on synthetic data.
- Unit test: Known synthetic samples (TTS output) are flagged with higher probability.
- Integration test: Deepfake detector integrated into voice pipeline returns anti-spoof score.

### Acceptance Criteria
1. CNN-based detector runs in < 15ms on CPU.
2. Trained detector achieves AUC-ROC > 0.95 on ASVspoof 2021 logical access test set.
3. Detector handles multiple TTS systems (not just one).
4. Probability output is calibrated (not overconfident).

### Definition of Done
- [ ] Deepfake detection classifier implemented.
- [ ] Model trained and meeting AUC-ROC target.
- [ ] Detector integrated into voice pipeline.
- [ ] All unit and integration tests pass.

### Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| New TTS systems not represented in training data | Lower detection rate on unseen synthesizers | Include diverse TTS systems in training; plan for periodic retraining |
| wav2vec2 model too large for CPU inference | High latency in production | Default to CNN-based detector; wav2vec2 as optional enhancement |
| Genuine speech misclassified as deepfake (false positive) | Legitimate users rejected | Tune decision threshold for low FPR; combine with other anti-spoof signals |

---

## Phase 9: ML — Voice Conversion Detection

### Objective
Detect voice conversion and speaker mimicry attacks where an attacker transforms their voice to sound like the target user.

### Current Problem
**Not implemented.** No component detects voice conversion or speaker mimicry. An attacker with access to a VC system could impersonate a target speaker.

### Work to Perform
1. Implement embedding consistency checking:
   - Compare CNN-LSTM-Attention embedding with a parallel traditional feature vector (e.g., i-vector or x-vector from a separate model).
   - If embeddings disagree significantly, flag as potential conversion.
   - Threshold: embedding distance > τ.
2. Implement prosodic naturalness scoring:
   - Pitch contour smoothness analysis.
   - Energy contour naturalness.
   - Speaking rate consistency.
   - Unnatural prosody → flag as converted.
3. Implement spectral-temporal coherence analysis:
   - Joint analysis of spectral envelope and temporal fine structure.
   - Converted speech shows inconsistencies between these domains.
   - Convolutional artifact detection.
4. Composite score feeds into anti-spoof pipeline.

### Files/Modules Affected
- New: `backend/voiceguard/voice/anti_spoofing/voice_conversion.py`
- New: `backend/voiceguard/ml/training/train_voice_conversion.py` (if ML-based approach is used)
- Modified: `backend/voiceguard/voice/pipeline.py` (add VC detection stage)

### Dependencies
- Phase 5 (anti-spoofing feature extraction, mel-spectrogram).
- Phase 6 (speaker embedding model for consistency checking).
- Training data: VC system outputs vs. genuine speech (may need to be synthesized or sourced from public datasets).

### Tests
- Unit test: Embedding consistency check flags dissimilar embeddings.
- Unit test: Prosodic naturalness scores are in valid range.
- Unit test: Spectral-temporal coherence detects known conversion artifacts.
- Integration test: VC detector integrated into voice pipeline returns composite score.

### Acceptance Criteria
1. VC detection runs in < 20ms on CPU.
2. Detection works against at least 2 VC systems (e.g., StarGAN-VC, OpenVoice).
3. Genuine speech is not flagged as converted (FPR < 5%).
4. Composite score integrates cleanly with other anti-spoof signals.

### Definition of Done
- [ ] All three VC detection checks implemented.
- [ ] Detection tested against known VC system outputs.
- [ ] Integrated into voice pipeline.
- [ ] All unit and integration tests pass.

### Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| VC detection without dedicated training data is unreliable | High false positive/negative rate | Start with heuristic checks; add ML training as data becomes available |
| Embedding consistency check requires a second model | Increased inference cost | Use lightweight model (e.g., GMM-based i-vector) for the parallel path |
| New VC techniques bypass detection | Security gap | Design for periodic model updates; monitor detection rates in production |

---

## Phase 10: ML — Challenge-Response Liveness

### Objective
Implement server-side challenge-response validation using speech-to-text, replacing the current display-only challenge that is never validated.

### Current Problem
`AntiSpoofingDetector.generate_challenge()` (`app.py:253-258`) generates a phrase + 4-digit number displayed to the user. The `challenge_text` parameter in `authenticate()` (`app.py:392-393`) is never read or validated. The user can say anything and still pass. Only 5 hardcoded phrases exist. There is no expiry, no uniqueness, and no server-side validation. This is the single most exploitable gap in the current anti-spoofing system.

### Work to Perform
1. Implement challenge generation service:
   - Random phrase from large vocabulary (100+ phrases).
   - Number challenges: "Say the numbers: 7 3 9 2" (random length 3–6).
   - Combined challenges: "Say 'pay' followed by 3 8 1".
   - Store challenge_id + expected_text + expiry (30 seconds) in Redis.
   - Rate limit: 3 challenges per minute per user.
2. Implement speech-to-text integration:
   - Option A: Whisper (local, higher accuracy, ~1s latency).
   - Option B: Whisper API (external, easier setup, network dependency).
   - Option C: Simpler ASR model (lighter, lower accuracy).
3. Implement Word Error Rate (WER) calculation:
   - WER < 0.2 → challenge passed (score = 1.0).
   - WER 0.2–0.4 → challenge marginal (weighted score, e.g., 0.5).
   - WER > 0.4 → challenge failed (score = 0.0).
4. Integrate challenge score into composite anti-spoof scoring.
5. Expand challenge vocabulary from 5 phrases to 100+.

### Files/Modules Affected
- New: `backend/voiceguard/voice/challenge.py` (challenge generation, validation, WER)
- New: `backend/voiceguard/services/challenge_service.py`
- New: `backend/voiceguard/routes/challenges.py`
- Modified: `backend/voiceguard/voice/pipeline.py` (add challenge validation stage)
- Modified: `backend/voiceguard/voice/anti_spoofing/__init__.py` (wire challenge score)

### Dependencies
- Phase 2 (Redis for challenge nonce storage with TTL).
- Phase 3 (FastAPI routes for challenge generation/retrieval).
- Phase 5 (audio preprocessing for the recorded challenge response).
- ASR model or API must be available (Whisper or alternative).

### Tests
- Unit test: Challenge generation produces unique challenge_ids.
- Unit test: Challenge expires after 30 seconds.
- Unit test: Same challenge cannot be reused.
- Unit test: Rate limiting blocks after 3 challenges/minute.
- Unit test: WER calculation produces correct values for known inputs.
- Unit test: Challenge score mapping (WER < 0.2 → 1.0, etc.) is correct.
- Integration test: Full challenge flow: generate → user records → ASR → WER → score.
- Integration test: Expired challenge is rejected.

### Acceptance Criteria
1. Challenge vocabulary has 100+ phrases.
2. Challenges expire after 30 seconds and cannot be reused.
3. ASR correctly transcribes challenge responses with WER < 0.15 on clean speech.
4. Challenge-response score feeds into composite anti-spoof score.
5. Rate limiting prevents challenge flooding.

### Definition of Done
- [ ] Challenge generation service with large vocabulary implemented.
- [ ] ASR integration (Whisper or alternative) functional.
- [ ] WER-based scoring is correct and configurable.
- [ ] Challenge validation is integrated into voice pipeline.
- [ ] All unit and integration tests pass.

### Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| Whisper model too large for deployment | High memory usage | Use Whisper tiny/base model; or use external API |
| ASR accuracy degrades in noisy environments | Legitimate users fail challenge | Adjust WER threshold; combine with audio quality gate |
| Challenge phrases may be unintelligible on small speakers | User confusion | Test phrase vocabulary with real users; avoid similar-sounding phrases |
| Network-dependent ASR (API option) creates availability risk | Challenge validation fails when API is down | Implement local Whisper fallback; cache ASR model locally |

---

## Phase 11: Composite Anti-Spoofing System

### Objective
Integrate all anti-spoofing detectors (replay, deepfake, voice conversion, challenge-response) into a single composite scoring system with configurable weights and threshold.

### Current Problem
The current system has disjointed anti-spoofing: audio quality checks work, challenge-response is displayed but never validated, OneClassSVM is dead code, and replay detection is a simple heuristic. There is no composite score, no weighting, and no single decision point.

### Work to Perform
1. Implement composite anti-spoof orchestrator in `voice/anti_spoofing/__init__.py`:
   - Collect scores from all detectors: quality gate, challenge-response, replay, deepfake, voice conversion.
   - Compute weighted composite score: `score = w1*challenge + w2*replay + w3*deepfake + w4*vc_detection`.
   - Weights are configurable via `Settings`.
   - Decision: `score > threshold → proceed to speaker verification`; `score <= threshold → REJECT`.
2. Implement quality gate as first checkpoint (from current `check_audio_quality()`, tightened thresholds).
3. Implement parallel execution of liveness detectors (challenge-response, replay, deepfake, VC) where possible.
4. Add anti-spoof score to audit log and API response.
5. Implement anti-spoof-specific event logging: `auth.anti_spoof.replay_detected`, `auth.anti_spoof.deepfake_detected`, etc.

### Files/Modules Affected
- Modified: `backend/voiceguard/voice/anti_spoofing/__init__.py` (composite orchestrator)
- Modified: `backend/voiceguard/voice/pipeline.py` (anti-spoof → verification ordering)
- Modified: `backend/voiceguard/services/voice_service.py` (log anti-spoof scores)
- Modified: `backend/voiceguard/schemas/voice.py` (add anti_spoof_score to response)

### Dependencies
- Phase 7 (replay detection).
- Phase 8 (deepfake detection).
- Phase 9 (voice conversion detection).
- Phase 10 (challenge-response validation).
- Phase 5 (quality gate and anti-spoofing features).

### Tests
- Unit test: Composite score is weighted sum of individual scores.
- Unit test: Threshold decision correctly accepts/rejects.
- Unit test: Missing detector (e.g., VC not available) gracefully degrades composite score.
- Unit test: Quality gate rejects audio below SNR/clipping/silence thresholds.
- Integration test: Full anti-spoof pipeline with all detectors returns composite score.
- Integration test: Known spoofed audio is rejected by composite system.
- Integration test: Genuine audio passes composite system.

### Acceptance Criteria
1. Composite score aggregates all available detector scores with configurable weights.
2. Quality gate is the first checkpoint (fast reject for garbage audio).
3. Anti-spoof decision is logged with all sub-scores.
4. Missing detectors are handled gracefully (score defaults to neutral).
5. Full anti-spoof pipeline (quality → challenge + replay + deepfake + VC) runs in < 100ms on CPU.

### Definition of Done
- [ ] Composite anti-spoof orchestrator implemented.
- [ ] All four detectors integrated and weighted.
- [ ] Quality gate rejects poor audio before expensive ML inference.
- [ ] Anti-spoof scores are logged and returned in API responses.
- [ ] All unit and integration tests pass.

### Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| Weight tuning without validation data produces suboptimal decisions | Poor FAR/FRR trade-off | Start with equal weights; tune on validation set post-integration |
| Parallel detector execution increases latency | Slow authentication | Profile bottleneck detectors; run serially if needed |
| One weak detector drags down composite score | Legitimate users rejected | Allow per-detector enable/disable; graceful degradation |
| Over-reliance on challenge-response (strongest signal) ignores other detectors | Replay/deepfake attacks bypass if challenge is compromised | Ensure weights prevent single-detector dominance |

---

## Phase 12: Client Refactoring

### Objective
Refactor the Kivy desktop client to communicate with the backend API, extract shared components, eliminate code duplication, and implement proper state management.

### Current Problem
The client is the same `app.py` monolith. Audio recording logic is triplicated across three screens (`LoginScreen`, `RegistrationScreen`, `PaymentScreen`). Challenge phrases are displayed but never sent to the backend. Balance is stored in a local float variable. No API client, no session management, no input validation, no error handling framework.

### Work to Perform
1. Create client package structure per Architecture §2.5:
   - `client/voiceguard/screens/` — login, registration, payment, settings.
   - `client/voiceguard/widgets/` — extracted `VGButton`, `VGCard`, `VoiceWaveform`.
   - `client/voiceguard/services/` — `AudioRecorder`, `ApiClient`, `AuthManager`.
   - `client/voiceguard/config.py`.
2. Extract `AudioRecorder` service class — single shared class replacing triplicated `_record_audio`/`_audio_callback`.
3. Implement `ApiClient` — HTTP client for backend REST endpoints with:
   - JWT token management (access + refresh).
   - Automatic token refresh before expiry.
   - Structured error handling from backend responses.
   - Audio upload (multipart/form-data).
4. Implement `AuthManager` — session state (tokens, user info, login status).
5. Refactor screens to use `ApiClient` instead of in-process logic.
6. Add form validation (username format, email format, amount bounds).
7. Implement settings screen for microphone selection.
8. Add on-device preprocessing (quality checks before upload).
9. Rename `PaytmButton` → `VGButton`, `PaytmCard` → `VGCard`.

### Files/Modules Affected
- New: `client/voiceguard/__init__.py`
- New: `client/voiceguard/app.py`
- New: `client/voiceguard/screens/__init__.py`
- New: `client/voiceguard/screens/login.py`
- New: `client/voiceguard/screens/registration.py`
- New: `client/voiceguard/screens/payment.py`
- New: `client/voiceguard/screens/settings.py`
- New: `client/voiceguard/widgets/__init__.py`
- New: `client/voiceguard/widgets/waveform.py`
- New: `client/voiceguard/widgets/vg_button.py`
- New: `client/voiceguard/widgets/vg_card.py`
- New: `client/voiceguard/services/__init__.py`
- New: `client/voiceguard/services/audio_recorder.py`
- New: `client/voiceguard/services/api_client.py`
- New: `client/voiceguard/services/auth_manager.py`
- Modified: `client/pyproject.toml` (add `kivy`, `httpx` deps)
- Removed: In-process ML/database logic from screens (moved to backend).

### Dependencies
- Phase 3 (backend API must be running for client to communicate with).
- Phase 4 (authentication endpoints for login/registration).
- Phase 11 (anti-spoofing pipeline must be functional on backend for voice auth).
- Phase 5 (shared preprocessing module for on-device quality checks).

### Tests
- Unit test: `AudioRecorder` starts/stops recording, handles errors.
- Unit test: `ApiClient` sends correct request format for each endpoint.
- Unit test: `ApiClient` handles 401 response by refreshing token.
- Unit test: `AuthManager` stores and clears session state.
- Unit test: Form validation rejects invalid inputs.
- Unit test: On-device preprocessing quality checks reject poor audio before upload.
- Integration test: Registration flow with backend (create user, upload voice samples).
- Integration test: Login flow with backend (password step, voice step).
- Integration test: Payment flow with backend (submit transaction, receive result).

### Acceptance Criteria
1. Client communicates with backend via REST API (no in-process ML or database).
2. Audio recording is shared across all screens via `AudioRecorder` service.
3. Challenge-response audio is sent to backend for validation.
4. Balance is fetched from server (not stored locally).
5. Form validation prevents invalid submissions.
6. Token refresh happens automatically before expiry.
7. `PaytmButton`/`PaytmCard` are renamed.

### Definition of Done
- [ ] Client package structure is complete and documented.
- [ ] All screens use `ApiClient` for backend communication.
- [ ] `AudioRecorder` is shared and triplicated code is eliminated.
- [ ] Form validation is implemented on all forms.
- [ ] Token management (access + refresh) works correctly.
- [ ] All unit and integration tests pass.

### Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| Kivy framework limitations block advanced UI features | Reduced UX quality | Identify Kivy limitations early; plan workarounds |
| Backend API changes during client development | Client-server contract mismatch | Use OpenAPI contract-first; generate client stubs |
| Audio upload over slow network times out | Failed voice authentication | Implement chunked upload; show progress indicator; set generous timeout |
| On-device preprocessing rejects valid audio | User frustration | Make preprocessing checks configurable; allow override in development mode |

---

## Phase 13: Integration & End-to-End Testing

### Objective
Verify the complete system works end-to-end: registration, authentication, payment, anti-spoofing, and audit logging.

### Current Problem
Zero tests exist in the repository. No automated verification that any component works correctly, let alone the integrated system. The codebase audit identified 10+ untested areas.

### Work to Perform
1. Implement E2E test suite covering critical user flows:
   - Full enrollment: register user → upload 5 voice samples → template stored.
   - Login: enter username → password → voice authentication → session created.
   - Payment: authenticated user → enter recipient/amount → voice authorize → transaction completed.
   - Anti-spoof: replay audio → rejected; deepfake audio → rejected; wrong challenge response → rejected.
   - Session lifecycle: login → token refresh → logout → old token rejected.
   - Account lockout: 5 failed attempts → locked → cooldown → unlocked.
2. Create test audio fixtures:
   - Genuine enrollment samples (5 per user).
   - Genuine authentication samples.
   - Impostor samples (different speaker).
   - Replay samples (recorded playback).
   - Deepfake samples (TTS-generated).
   - Low-quality samples (noisy, clipped, too short).
3. Implement test infrastructure:
   - Docker Compose test profile with test database and Redis.
   - Test fixtures for database seeding.
   - Mock ASR for challenge-response tests (or use local Whisper).
   - Test audio file generation scripts.
4. Run full lint (`ruff check`), type check (`mypy`), and security scan (`bandit`) on entire codebase.

### Files/Modules Affected
- New: `tests/` directory (top-level or per-package).
- New: `tests/conftest.py` (shared fixtures).
- New: `tests/e2e/` (end-to-end tests).
- New: `tests/fixtures/audio/` (test audio files).
- New: `tests/integration/` (API + DB + ML pipeline tests).
- New: `tests/unit/` (individual function tests).
- Modified: `docker-compose.yml` (add test profile).
- Modified: `pyproject.toml` (add pytest, pytest-asyncio, httpx test deps).

### Dependencies
- All previous phases (1–12) must be complete.

### Tests
(This phase is about *running* all tests and filling gaps.)
- Verify all unit tests from Phases 1–12 pass.
- Verify all integration tests from Phases 1–12 pass.
- E2E tests for each critical flow listed above.
- Performance tests: voice pipeline latency < 200ms p95.
- Security scan: no hardcoded secrets, no SQL injection vectors.

### Acceptance Criteria
1. All unit tests pass (target: 200+ tests).
2. All integration tests pass.
3. All E2E tests pass for critical flows.
4. `ruff check .` passes with zero warnings.
5. `mypy backend/ client/ shared/` passes with zero errors.
6. `bandit -r backend/` reports no high-severity issues.
7. Voice pipeline latency is < 200ms at p95 on CPU.
8. Code coverage is > 80%.

### Definition of Done
- [ ] Full test suite is green.
- [ ] Lint, type check, and security scan pass clean.
- [ ] E2E flows are verified against running backend + database.
- [ ] Performance benchmarks meet latency targets.
- [ ] Coverage report shows > 80% line coverage.

### Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| Test audio fixtures are not representative of real speech | Tests pass but production fails | Include diverse speakers, accents, recording conditions |
| E2E tests are flaky due to timing/async issues | Unreliable CI | Use explicit waits, retry logic, and deterministic test data |
| Coverage target forces testing of unimportant code paths | Wasted effort | Focus on critical paths; accept lower coverage for UI code |
| ML model performance degrades between training and integration | Tests fail unexpectedly | Pin model versions; use deterministic inference (set seeds) |

---

## Phase 14: Deployment & CI/CD

### Objective
Establish a working development environment, CI/CD pipeline, and deployment configuration.

### Current Problem
The `.devcontainer/devcontainer.json` runs `streamlit run app.py` — the app is Kivy, not Streamlit, so this fails immediately. No Dockerfile, no docker-compose, no CI/CD, no environment handling, no GitHub Actions.

### Work to Perform
1. Update DevContainer configuration per Architecture §25.4:
   - Python 3.11 + PostgreSQL client + Redis client.
   - Post-create: `pip install -e '.[dev]' && docker compose up -d && alembic upgrade head`.
   - Forward ports: 8000 (API), 5432 (PostgreSQL), 6379 (Redis).
2. Implement CI/CD pipeline (GitHub Actions):
   - On push to main: lint → type check → security scan → unit tests → integration tests → build Docker image → push to registry → deploy to staging.
   - On tag release (v*): all staging checks → manual approval → deploy to production (blue-green) → smoke tests.
   - On pull request: all tests → coverage > 80% → code review → merge.
3. Implement Dockerfile for backend:
   - Multi-stage build: builder (install deps) → runtime (minimal image).
   - Non-root user, health check, graceful shutdown.
4. Implement deployment configurations:
   - Development: `docker-compose.yml` (API, PostgreSQL, Redis, MinIO).
   - Staging: Docker containers on single VM + Nginx reverse proxy + Let's Encrypt SSL.
   - Production (Option A): Cloud-managed (ECS/Cloud Run/AKS + RDS/Cloud SQL + ElastiCache/Memorystore).
5. Implement environment-specific configuration via `.env` files (gitignored).
6. Implement structured logging (JSON format) with request-ID propagation.
7. Implement audit log table and logging for all security events.

### Files/Modules Affected
- Modified: `.devcontainer/devcontainer.json` (replace Streamlit with FastAPI).
- New: `Dockerfile` (backend).
- New: `.github/workflows/ci.yml`
- New: `.github/workflows/deploy.yml`
- New: `nginx/` (reverse proxy config for staging).
- New: `.env.example` (template for required environment variables).
- New: `scripts/deploy.sh` (deployment helper).
- Modified: `backend/voiceguard/main.py` (structured logging, request-ID middleware).
- Modified: `docker-compose.yml` (add dev/test/prod profiles).

### Dependencies
- All previous phases (1–13) must be complete.

### Tests
- CI pipeline runs all existing tests.
- Docker build succeeds and image starts.
- `docker-compose up` starts all services in correct order.
- Health check endpoint returns healthy after startup.
- Smoke tests pass after deployment.
- Structured logs are produced in JSON format.

### Acceptance Criteria
1. `devcontainer` opens in Codespaces and starts all services.
2. CI pipeline runs on every push and PR.
3. Docker image builds and starts the API server.
4. Health check endpoint returns 200.
5. Structured logs are produced in JSON format.
6. Deployment to staging is automated on merge to main.
7. `.env.example` documents all required variables.

### Definition of Done
- [ ] DevContainer works correctly in GitHub Codespaces.
- [ ] CI pipeline runs lint, type check, security scan, and all tests.
- [ ] Docker image builds and starts successfully.
- [ ] Structured logging is operational.
- [ ] Staging deployment is automated.
- [ ] Documentation is updated with setup and deployment instructions.

### Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| DevContainer resource limits slow down services | Poor developer experience | Document minimum Codespace specs; provide local Docker alternative |
| CI pipeline is slow (>10 min) | Developer frustration | Parallelize test stages; cache dependencies; use matrix builds |
| Blue-green deployment requires duplicate infrastructure | Higher cost | Start with single-instance staging; add blue-green for production |
| Secrets management in CI/CD is complex | Security risk | Use GitHub Secrets; document rotation procedure; audit access |

---

## Phase Summary & Dependency Graph

| Phase | Name | Parallelizable With | Blocked By |
|-------|------|---------------------|------------|
| 1 | Project Structure & Config | — | — |
| 2 | Database & Persistence | 5 | 1 |
| 3 | Backend API Foundation | 5 | 1, 2 |
| 4 | Auth & Session Management | 5, 7, 8, 9, 10 | 2, 3 |
| 5 | Voice Preprocessing & Features | 2, 3, 4 | 1 |
| 6 | CNN-LSTM-Attention & Speaker Verification | 4, 7, 8, 9, 10 | 2, 3, 5 |
| 7 | Replay Detection | 4, 6, 8, 9, 10 | 5, 6 (patterns) |
| 8 | Deepfake Detection | 4, 6, 7, 9, 10 | 5, 6 (patterns) |
| 9 | Voice Conversion Detection | 4, 6, 7, 8, 10 | 5, 6 |
| 10 | Challenge-Response Liveness | 4, 6, 7, 8, 9 | 2, 3, 5 |
| 11 | Composite Anti-Spoofing | — | 7, 8, 9, 10 |
| 12 | Client Refactoring | 11 | 3, 4, 5, 11 |
| 13 | Integration & E2E | — | 1–12 |
| 14 | Deployment & CI/CD | — | 1–13 |

### Critical Path (Longest Dependency Chain)

```
Phase 1 → Phase 2 → Phase 3 → Phase 5 → Phase 6 → Phase 11 → Phase 12 → Phase 13 → Phase 14
```

**Estimated critical path duration: 14 sequential phases.** Phases 7–10 can be done in parallel with Phases 4 and 6, reducing total wall-clock time.

### Recommended Implementation Order

1. **Phase 1** — Project structure (blocks everything)
2. **Phase 2** — Database (blocks API and auth)
3. **Phase 5** — Voice preprocessing (can start in parallel with Phase 2)
4. **Phase 3** — Backend API (blocks auth and client)
5. **Phase 4** — Auth & sessions (blocks client)
6. **Phase 6** — Speaker verification ML (blocks composite anti-spoof)
7. **Phase 7, 8, 9, 10** — Anti-spoofing ML (parallel with each other)
8. **Phase 11** — Composite anti-spoofing (integrates 7–10)
9. **Phase 12** — Client refactoring (needs API + anti-spoofing)
10. **Phase 13** — Integration testing (needs everything)
11. **Phase 14** — Deployment (final)
