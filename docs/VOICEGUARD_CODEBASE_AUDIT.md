# VoiceGuard Codebase Audit

**Date:** 2026-09-02
**Branch:** `voiceguard-v2`
**Auditor:** Automated read-only inspection
**Scope:** Complete repository at commit `defb452`

---

## 1. Repository Overview

VoiceGuard is a **voice-authenticated payment application**. It allows users to register a voice profile, log in via voice recognition, and authorize monetary transactions by speaking a challenge phrase. The entire application is a **single Python file** (`app.py`, ~1,646 lines) using the **Kivy** GUI framework.

### File Inventory

| File | Size | Purpose |
|------|------|---------|
| `app.py` | 64 KB, 1,646 lines | Entire application: GUI, ML, DB, auth, payments |
| `requirements.txt` | 8 lines | Python dependencies (partially incorrect) |
| `README.md` | 355 lines | Project documentation |
| `CNAME` | 1 line | `voiceguardbtrial.com` |
| `.devcontainer/devcontainer.json` | 33 lines | GitHub Codespaces config |
| `screenshots/*.png` | 5 files | UI screenshots |

**Total: 10 files.** No tests, no `.gitignore`, no LICENSE file, no templates, no static assets, no model files, no database files, no scripts.

---

## 2. Architecture

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────┐
│               VoiceGuardApp (Kivy)              │
│                                                 │
│  ┌──────────┐ ┌──────────────┐ ┌─────────────┐ │
│  │  Login   │ │ Registration │ │   Payment   │ │
│  │  Screen  │ │   Screen     │ │   Screen    │ │
│  └────┬─────┘ └──────┬───────┘ └──────┬──────┘ │
│       │              │                │         │
│  ┌────▼──────────────▼────────────────▼──────┐  │
│  │      VoiceAuthenticationEngine           │  │
│  │  ┌────────────────┐  ┌─────────────────┐  │  │
│  │  │VoiceFeatureExtr│  │AntiSpoofDetector│  │  │
│  │  └────────────────┘  └─────────────────┘  │  │
│  │  ┌────────────────┐  ┌─────────────────┐  │  │
│  │  │VoiceEmbedding  │  │  GMM (sklearn)  │  │  │
│  │  │   Net (PyTorch)│  │                 │  │  │
│  │  └────────────────┘  └─────────────────┘  │  │
│  └─────────────────────┬─────────────────────┘  │
│                        │                        │
│  ┌─────────────────────▼─────────────────────┐  │
│  │          SecureDatabase (In-Memory)       │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### 2.2 Application Flow

1. **Startup:** `VoiceGuardApp.build()` creates a `VoiceAuthenticationEngine` and three screens: Login, Registration, Payment.
2. **Registration:** User enters username/email, records 5 voice samples (each with a prompted phrase). Features are extracted, a GMM is trained per user, embeddings are generated via the neural net. User data is stored in-memory with a starting balance of 10,000.00 rupees.
3. **Login:** User enters username, holds a button to record audio. The engine extracts features, computes cosine similarity against stored embeddings, scores via GMM, and applies a combined weighted threshold. Graduated thresholds: 0.72 for first attempt, 0.80 for subsequent.
4. **Payment:** Authenticated user sees a dashboard with balance, send-money form, challenge phrase, and transaction history. User fills in recipient/amount, presses "HOLD TO AUTHORIZE", which triggers voice recording and a second authentication gate before the transaction executes.

### 2.3 Modules Inside `app.py`

| Class | Lines | Responsibility |
|-------|-------|---------------|
| `SecureDatabase` | 59-106 | In-memory encrypted user/model/transaction storage |
| `VoiceFeatureExtractor` | 109-218 | MFCC, mel-spectrogram, spectral, prosodic, formant extraction |
| `VoiceEmbeddingNet` | 222-238 | PyTorch 3-layer FC network for 128-dim speaker embeddings |
| `AntiSpoofingDetector` | 242-341 | Replay detection via cosine similarity, audio quality checks |
| `VoiceAuthenticationEngine` | 344-503 | Orchestrates enrollment + authentication |
| `PaytmButton` / `PaytmCard` / `VoiceWaveform` | 507-564 | Custom Kivy widgets |
| `LoginScreen` | 567-792 | Login UI + audio recording thread |
| `RegistrationScreen` | 794-1105 | Registration UI + multi-sample collection |
| `PaymentScreen` | 1108-1615 | Payment dashboard UI + full transaction flow |
| `VoiceGuardApp` | 1619-1645 | App entry point |

---

## 3. Technologies Used

### 3.1 Actual Imports in `app.py`

| Category | Libraries |
|----------|-----------|
| **GUI** | `kivy` (App, ScreenManager, Screen, widgets, properties, graphics, animation, clock, window) |
| **Audio** | `sounddevice`, `soundfile`, `scipy.io.wavfile`, `scipy.signal`, `librosa` |
| **ML/Scientific** | `numpy`, `scipy.spatial.distance.cosine`, `sklearn.preprocessing.StandardScaler`, `sklearn.mixture.GaussianMixture`, `sklearn.svm.OneClassSVM`, `torch`, `torch.nn`, `torch.nn.functional` |
| **Security** | `cryptography.fernet.Fernet`, `cryptography.hazmat.primitives` (PBKDF2, hashes), `pyotp`, `base64` |
| **Standard Lib** | `os`, `json`, `hashlib`, `random`, `string`, `datetime`, `timedelta`, `threading`, `queue`, `typing` |

### 3.2 Listed in `requirements.txt`

```
streamlit>=1.28.0
numpy>=1.24.0
scipy>=1.10.0
librosa>=0.10.0
noisereduce>=3.0.0
scikit-learn>=1.3.0
torch>=2.0.0
cryptography>=41.0.0
```

### 3.3 Discrepancies

| Issue | Detail |
|-------|--------|
| **Missing from requirements.txt** | `kivy`, `sounddevice`, `soundfile`, `pyotp` — all imported and actively used |
| **Phantom in requirements.txt** | `streamlit` — not imported anywhere. `noisereduce` — imported nowhere |
| **Unused stdlib imports** | `hashlib`, `queue`, `string`, `timedelta`, `Optional`, `GridLayout`, `SlideTransition`, `Image` (Kivy), `wavfile`, `sf`, `StandardScaler` |

---

## 4. What Is Implemented

### 4.1 Voice Feature Extraction (Implemented)
- MFCC (40 coefficients) with delta
- Mel-spectrogram (128 mels)
- Spectral centroid, rolloff, zero-crossing rate
- Pitch estimation via `librosa.pyin`
- RMS energy (mean, std)
- Formant estimation via LPC
- Audio preprocessing: normalization, spectral subtraction noise reduction, energy-based VAD

### 4.2 Speaker Embedding Neural Network (Implemented, Untrained)
- 3-layer fully connected: 256 → 512 → 256 → 128
- BatchNorm1d + Dropout(0.3) on hidden layers
- L2-normalized output
- **Randomly initialized weights** — never trained on any dataset

### 4.3 User Enrollment (Implemented)
- Collects 5 voice samples with prompted phrases
- Quality-gated: rejects samples with quality score < 0.45
- Stores embeddings, raw features, trained GMM (2-component diagonal) per user
- Starting balance of 10,000.00

### 4.4 Voice Authentication (Implemented)
- Combined scoring: 0.6 × max cosine similarity + 0.3 × avg cosine similarity + 0.1 × GMM log-likelihood score
- Graduated thresholds: 0.72 (first attempt) → 0.80 (subsequent)
- Attempt tracking with 15-minute session reset
- Post-auth feature accumulation (up to 20 vectors)

### 4.5 Anti-Spoofing (Partially Implemented)
- Audio quality scoring: SNR, clipping ratio, silence ratio, duration checks
- Replay detection: cosine similarity against historical features (not trained OneClassSVM)
- Challenge-response phrases displayed to user (**but never validated**)

### 4.6 Payment Flow (Implemented)
- Send-money form with recipient, amount, description
- Challenge phrase generation and display
- Voice authorization gate before transaction execution
- Balance deduction and transaction logging
- Success/decline popup dialogs
- Transaction history display

### 4.7 GUI (Implemented)
- Three screens with slide transitions: Login, Registration, Payment
- Custom widgets: `PaytmButton`, `PaytmCard`, `VoiceWaveform`
- Real-time waveform visualization during recording
- Background audio recording via `sounddevice.InputStream` in threads

### 4.8 Encryption (Implemented)
- Fernet symmetric encryption for user data at rest (in-memory)
- Encryption key generated fresh each run (not persisted)

---

## 5. What Is Incomplete or Missing

| Component | Status |
|-----------|--------|
| **ML model training** | `VoiceEmbeddingNet` uses random weights. No training pipeline exists. No pre-trained weights are loaded. |
| **OneClassSVM anti-spoofing** | Instantiated in `AntiSpoofingDetector.__init__` but never `.fit()` or `.predict()` called. Dead code. |
| **Challenge-response validation** | Challenge phrase is displayed to the user but `challenge_text` parameter in `authenticate()` is never read or validated. The user can say anything. |
| **pyotp (TOTP)** | Imported but never used anywhere. MFA/2FA is not implemented. |
| **Data persistence** | All data lives in Python dicts/lists. Lost on every restart. No file, SQLite, or external DB. |
| **Model persistence** | Trained GMMs and neural net weights are never saved to disk. |
| **User session management** | No session tokens, no timeout enforcement, no secure session storage. |
| **Input validation/sanitization** | No validation on username format, email, recipient, amount (beyond basic empty checks). |
| **Error handling** | Minimal. No try/except around audio recording, ML inference, or encryption. Crashes are unhandled. |
| **Configuration** | No config file. All constants hardcoded (thresholds, sample rates, dimensions, phrases, etc.). |
| **Logging** | No application logging. No audit trail beyond in-memory transaction list. |
| **`.gitignore`** | Absent. Risk of committing Python bytecode, virtual envs, IDE files, recordings. |
| **LICENSE** | Absent. README claims MIT but no LICENSE file exists. |
| **Tests** | Zero test files anywhere in the repository. |

---

## 6. Architectural Weaknesses

### 6.1 Single-File Monolith
The entire application (ML pipeline, database, GUI, business logic, security) is in one 1,646-line file. This makes the codebase difficult to maintain, test, review, or extend.

### 6.2 In-Memory-Only Storage
`SecureDatabase` stores everything in Python dicts. All user data, voice models, and transaction history vanish on application restart. This makes the application non-functional for any real use.

### 6.3 Tight Coupling
Screens directly instantiate and manage background threads, audio streams, ML models, and the database. There is no service layer, no dependency injection, and no separation of concerns.

### 6.4 No Persistence Layer
No database engine, no file-based storage, no serialization to disk. The README claims a `database/` directory that does not exist.

### 6.5 Framework Mismatch
The `.devcontainer` configuration runs `streamlit run app.py`, but the application is built with Kivy (a desktop GUI framework). This would fail immediately. The `requirements.txt` lists `streamlit` but it is not imported.

### 6.6 CNAME vs. Application Type
A `CNAME` file points to `voiceguardbtrial.com` (GitHub Pages), but the application is a desktop GUI, not a web application. This serves no purpose.

---

## 7. Security Weaknesses

| # | Severity | Issue |
|---|----------|-------|
| 1 | **Critical** | **No persistence** — encryption is moot when all data is lost on restart. The Fernet encryption protects nothing in practice. |
| 2 | **Critical** | **Encryption key not persisted** — `Fernet.generate_key()` is called fresh each run, making all encrypted data from previous runs unrecoverable. |
| 3 | **High** | **Challenge-response not validated** — The system generates a challenge phrase ("Verify 3847") and displays it, but never checks if the user actually said it. Anti-spoofing via challenge-response is completely bypassed. |
| 4 | **High** | **No password/auth for login** — Username alone grants access. There is no password, PIN, or any non-voice credential. If voice auth is bypassed, account takeover is trivial. |
| 5 | **High** | **Trained model not used** — `OneClassSVM` is instantiated but never trained, leaving the anti-spoofing system weaker than intended. |
| 6 | **High** | **Graduated threshold weakens security** — First attempt uses a *lower* threshold (0.72 vs 0.80), meaning the easiest time to spoof is the first try. |
| 7 | **Medium** | **Thread safety** — Audio recording runs in background threads that share mutable state (`is_recording`, `audio_queue`, `recorded_audio`) with the main Kivy thread without locks. |
| 8 | **Medium** | **No rate limiting** — Authentication attempts are tracked but never blocked. An attacker can try indefinitely. |
| 9 | **Medium** | **Balance manipulation risk** — Balance is stored in-memory in a plain dict. No atomic operations, no locks on balance reads/writes. |
| 10 | **Medium** | **No input sanitization** — Username, recipient, description fields accept arbitrary strings with no validation. |
| 11 | **Low** | **pyotp imported but unused** — Suggests planned MFA that was never implemented. |
| 12 | **Low** | **No TLS/HTTPS consideration** — While the app is desktop-only, there is no mention of transport security for any future network features. |

---

## 8. ML / Voice Processing Weaknesses

| # | Issue | Impact |
|---|-------|--------|
| 1 | **Untrained embedding network** | `VoiceEmbeddingNet` is initialized with random PyTorch weights. Speaker embeddings are essentially random vectors. Authentication accuracy is driven entirely by feature-level cosine similarity and GMM scoring. |
| 2 | **No training pipeline** | There is no code, script, or configuration to train the neural network on any dataset (e.g., VoxCeleb). |
| 3 | **OneClassSVM never used** | Declared but `.fit()` is never called. The replay detection fallback is a simpler cosine-similarity heuristic. |
| 4 | **Fixed feature vector size** | `_prepare_feature_vector` truncates/pads to exactly 256 dimensions regardless of actual feature dimensionality. Information loss on both ends. |
| 5 | **Minimal GMM** | Only 2 Gaussian components with diagonal covariance. May be insufficient to model speaker variability. |
| 6 | **Primitive VAD** | Energy-based VAD with a fixed threshold (0.02). Fails in noisy environments, quiet speakers, or non-stationary noise. |
| 7 | **Naive spectral subtraction** | Uses the first 10% of frames as noise estimate. Fragile assumption that noise is stationary and present only at the start. |
| 8 | **No data augmentation** | Enrollment uses raw audio only. No noise augmentation, speed perturbation, or other techniques to improve robustness. |
| 9 | **Replay detection is feature-level** | Compares extracted feature vectors to historical ones. A sophisticated replay (re-recorded with different mic, different room) may bypass this. |
| 10 | **No speaker verification baselines** | No comparison against established speaker verification approaches (x-vectors, ECAPA-TDNN, etc.). |
| 11 | **In-memory feature accumulation** | Post-auth, new features are appended to the user model (up to 20 vectors), progressively contaminating the reference set. |

---

## 9. Technical Debt and Code Quality

### 9.1 Unused Imports
14 imports are never referenced in the code: `hashlib`, `queue`, `string`, `timedelta`, `Optional`, `GridLayout`, `SlideTransition`, `Image`, `StandardScaler`, `wavfile`, `sf`, `pyotp`, `noisereduce` (not even imported but in requirements), `streamlit` (in requirements but not imported).

### 9.2 No `.gitignore`
Python bytecode (`__pycache__/`, `*.pyc`), IDE files (`.vscode/`, `.idea/`), virtual environments, and recording files can all be accidentally committed.

### 9.3 Hardcoded Constants
All configuration is embedded as literals: sample rates, feature dimensions, thresholds, phrase lists, audio parameters, neural network architecture, encryption settings. No environment variables, no config files, no command-line arguments.

### 9.4 Naming and Branding Inconsistencies
- Remote URL typo: `VoiceGurad` instead of `VoiceGuard`
- Widget named `PaytmButton`/`PaytmCard` (Paytm is a separate company)
- README references `main.py` but the file is `app.py`

### 9.5 README Documentation Mismatches
- Lists project directories (`assets/`, `models/`, `recordings/`, `database/`) that do not exist
- Claims MIT License but no `LICENSE` file is present
- States `python main.py` for installation but the entry point is `app.py`
- Mentions performance metrics (V1-V3, Latest: 94.6%/94%) with no evidence or test methodology

### 9.6 Duplicated Patterns
- Audio recording logic (thread creation, queue management, `sounddevice.InputStream` setup) is duplicated across `LoginScreen`, `RegistrationScreen`, and `PaymentScreen` with no shared abstraction.
- Waveform visualization is duplicated between screens.
- Authentication call patterns are repeated with minor variations.

---

## 10. Testing

**There are zero test files in the entire repository.**

Not tested:
- Voice feature extraction correctness
- Embedding network output consistency
- Authentication scoring logic and thresholds
- Replay detection heuristic
- Audio quality assessment
- Database encryption/decryption round-trip
- Enrollment workflow (5-sample collection, quality gating, GMM training)
- Payment flow (validation, balance deduction, transaction logging)
- GUI screens (navigation, recording, state transitions)
- Edge cases (empty audio, zero-length features, missing user, double-submit)

---

## 11. Deployment

### 11.1 DevContainer Configuration
- **Image:** `mcr.microsoft.com/devcontainers/python:1-3.11-bookworm`
- **Extensions:** ms-python.python, ms-python.vscode-pylance
- **Run command:** `streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false`
- **Port:** 8501

### 11.2 Deployment Issues

| Issue | Detail |
|-------|--------|
| **Wrong framework** | The app uses Kivy (desktop GUI), not Streamlit (web framework). `streamlit run app.py` will fail because `app.py` contains no Streamlit code. |
| **Missing system dependencies** | Kivy requires system-level libraries (e.g., SDL2, GStreamer) not installed by the devcontainer setup. `sounddevice` requires PortAudio. |
| **No Dockerfile** | No containerization for deployment. |
| **No CI/CD** | No GitHub Actions, no test pipeline, no build pipeline. |
| **No environment config** | No `.env` files, no secrets management, no environment variable handling. |

---

## 12. Risks and Assumptions

### 12.1 Critical Risks

| Risk | Impact |
|------|--------|
| **Application is non-functional on restart** | All enrolled users, voice models, balances, and transaction history are lost every time the application closes. |
| **Voice authentication with random-weight neural net** | Speaker embeddings are semantically meaningless. The system works (or doesn't) by accident of feature-level statistics, not learned speaker representations. |
| **Anti-spoofing is largely cosmetic** | Challenge-response is displayed but not validated. OneClassSVM is dead code. Replay detection is a simple heuristic. A recording attack has a reasonable chance of success. |
| **No tests mean no confidence** | Zero automated verification that any component works correctly. |
| **Framework mismatch blocks deployment** | The Codespaces/devcontainer configuration is incompatible with the actual application framework. |

### 12.2 Assumptions

| Assumption | Risk |
|------------|------|
| User speaks clearly in a quiet environment | VAD and feature extraction assume clean audio |
| 5 enrollment samples are sufficient | No validation of enrollment quality beyond score threshold |
| Cosine similarity of features is a valid speaker discriminator | Without a trained model, this is an untested assumption |
| Kivy is the right framework | Desktop GUI limits deployment options; no web/mobile path |
| In-memory storage is acceptable for a prototype | Even for prototyping, losing all data on restart is disruptive |
| The CNAME / GitHub Pages setup is intentional | The app cannot be served as a static site |

---

## 13. Summary

VoiceGuard is a **proof-of-concept prototype** that demonstrates the concept of voice-authenticated payments. The core idea — extracting voice features, building per-user profiles, and gating transactions behind voice verification — is present in skeleton form. However, the system has **critical gaps** that prevent it from being functional or secure:

1. **No data persistence** — the application forgets everything on restart
2. **Untrained ML models** — the neural network has random weights; no training pipeline exists
3. **Incomplete anti-spoofing** — challenge-response is not validated; SVM is dead code
4. **No tests** — zero automated verification of any component
5. **Broken deployment** — devcontainer runs Streamlit on a Kivy app
6. **Incorrect dependencies** — requirements.txt is wrong in both directions (has unused deps, missing required deps)
7. **Significant code quality issues** — single-file monolith, duplicated patterns, hardcoded constants, unused imports, no `.gitignore`

The codebase needs substantial restructuring, a proper persistence layer, trained ML models, real anti-spoofing, comprehensive tests, corrected dependencies, and a viable deployment strategy before it could be considered functional.
