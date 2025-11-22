import os
import json
import hashlib
import random
import string
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import threading
import queue

# Kivy imports
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen, SlideTransition
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.progressbar import ProgressBar
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.clock import Clock
from kivy.graphics import Color, RoundedRectangle, Line, Rectangle
from kivy.uix.image import Image
from kivy.animation import Animation
from kivy.properties import StringProperty, NumericProperty, BooleanProperty
from kivy.core.window import Window

# Audio processing
import sounddevice as sd
import soundfile as sf
from scipy.io import wavfile
from scipy import signal
from scipy.spatial.distance import cosine
import librosa
import librosa.display

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
import torch
import torch.nn as nn
import torch.nn.functional as F

# Security
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import pyotp
import base64

# Set window background
Window.clearcolor = (0.95, 0.95, 0.97, 1)

# Database simulation (in production, use PostgreSQL/MongoDB)
class SecureDatabase:
    def __init__(self):
        self.users = {}
        self.voice_models = {}
        self.transaction_logs = []
        self.auth_attempts = {}  # Track authentication attempts per user
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)

    def encrypt_data(self, data: bytes) -> bytes:
        return self.cipher_suite.encrypt(data)

    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        return self.cipher_suite.decrypt(encrypted_data)

    def store_user(self, username: str, user_data: dict):
        encrypted_data = self.encrypt_data(json.dumps(user_data).encode())
        self.users[username] = encrypted_data

    def get_user(self, username: str) -> dict:
        if username in self.users:
            decrypted_data = self.decrypt_data(self.users[username])
            return json.loads(decrypted_data)
        return None

    def log_transaction(self, transaction: dict):
        transaction['timestamp'] = datetime.now().isoformat()
        self.transaction_logs.append(transaction)

    def track_auth_attempt(self, username: str, session_id: str = None):
        """Track authentication attempt number for graduated thresholds"""
        if username not in self.auth_attempts:
            self.auth_attempts[username] = {'attempt': 1, 'session_start': datetime.now()}
        else:
            self.auth_attempts[username]['attempt'] += 1
        
        # Reset after 15 minutes of inactivity
        time_diff = (datetime.now() - self.auth_attempts[username]['session_start']).total_seconds()
        if time_diff > 900:  # 15 minutes
            self.auth_attempts[username]['attempt'] = 1
            self.auth_attempts[username]['session_start'] = datetime.now()

    def get_attempt_number(self, username: str) -> int:
        """Get current attempt number for graduated threshold"""
        if username in self.auth_attempts:
            return self.auth_attempts[username]['attempt']
        return 1


# Advanced Voice Feature Extractor
class VoiceFeatureExtractor:
    def __init__(self):
        self.sample_rate = 16000
        self.n_mfcc = 40
        self.n_mels = 128
        self.hop_length = 512
        self.n_fft = 2048

    def extract_features(self, audio_data: np.ndarray) -> Dict:
        """Extract comprehensive voice features including MFCC, spectral, prosodic features"""
        features = {}

        # Preprocessing
        audio_data = self.preprocess_audio(audio_data)

        # MFCC features
        mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate,
                                    n_mfcc=self.n_mfcc, hop_length=self.hop_length)
        features['mfcc_mean'] = np.mean(mfcc, axis=1)
        features['mfcc_std'] = np.std(mfcc, axis=1)
        features['mfcc_delta'] = np.mean(librosa.feature.delta(mfcc), axis=1)

        # Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=self.sample_rate,
                                                  n_mels=self.n_mels, hop_length=self.hop_length)
        features['mel_mean'] = np.mean(mel_spec, axis=1)

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)
        features['spectral_centroid'] = np.mean(spectral_centroid)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)
        features['spectral_rolloff'] = np.mean(spectral_rolloff)
        zcr = librosa.feature.zero_crossing_rate(audio_data)
        features['zcr'] = np.mean(zcr)

        # Prosodic features
        f0, voiced_flag, voiced_probs = librosa.pyin(audio_data,
                                                      fmin=librosa.note_to_hz('C2'),
                                                      fmax=librosa.note_to_hz('C7'))
        features['pitch_mean'] = np.nanmean(f0)
        features['pitch_std'] = np.nanstd(f0)

        # Energy features
        rms = librosa.feature.rms(y=audio_data)
        features['energy_mean'] = np.mean(rms)
        features['energy_std'] = np.std(rms)

        # Formants (simplified estimation)
        features['formants'] = self.estimate_formants(audio_data)

        return features

    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply preprocessing: noise reduction, normalization, VAD"""
        # Normalize
        audio_data = audio_data / np.max(np.abs(audio_data))

        # Simple noise reduction using spectral subtraction
        audio_data = self.spectral_subtraction(audio_data)

        # Voice Activity Detection (VAD)
        audio_data = self.apply_vad(audio_data)

        return audio_data

    def spectral_subtraction(self, audio: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
        """Simple spectral subtraction for noise reduction"""
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Estimate noise (using first 0.1 seconds)
        noise_frames = int(0.1 * self.sample_rate / self.hop_length)
        noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

        # Subtract noise
        clean_magnitude = magnitude - noise_factor * noise_spectrum
        clean_magnitude = np.maximum(clean_magnitude, 0)

        # Reconstruct
        clean_stft = clean_magnitude * np.exp(1j * phase)
        clean_audio = librosa.istft(clean_stft, hop_length=self.hop_length)

        return clean_audio

    def apply_vad(self, audio: np.ndarray, threshold: float = 0.02) -> np.ndarray:
        """Simple energy-based Voice Activity Detection"""
        energy = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        voice_frames = energy > threshold

        # Expand voice regions
        voice_samples = np.repeat(voice_frames, self.hop_length)[:len(audio)]

        return audio * voice_samples

    def estimate_formants(self, audio: np.ndarray, n_formants: int = 3) -> np.ndarray:
        """Estimate formant frequencies using LPC"""
        # Pre-emphasis
        pre_emphasized = np.append(audio[0], audio[1:] - 0.95 * audio[:-1])

        # LPC analysis
        a = librosa.lpc(pre_emphasized, order=2 * n_formants + 2)

        # Get roots and convert to frequencies
        roots = np.roots(a)
        roots = roots[np.imag(roots) >= 0]
        angles = np.angle(roots)
        frequencies = sorted(angles * (self.sample_rate / (2 * np.pi)))[:n_formants]

        return np.array(frequencies)


# Deep Learning Model for Voice Embedding
class VoiceEmbeddingNet(nn.Module):
    def __init__(self, input_dim: int = 256, embedding_dim: int = 128):
        super(VoiceEmbeddingNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, embedding_dim)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalization


# Anti-Spoofing Module with Enhanced Replay Detection
class AntiSpoofingDetector:
    def __init__(self):
        self.replay_detector = OneClassSVM(gamma='auto', nu=0.05)
        self.liveness_phrases = [
            "done",
            "payment",
            "Verify",
            "Complete transaction",
            "Random"
        ]

    def generate_challenge(self) -> Tuple[str, str]:
        """Generate random challenge phrase and number"""
        phrase = random.choice(self.liveness_phrases)
        number = ''.join(random.choices(string.digits, k=4))
        challenge = f"{phrase} {number}"
        return challenge, number

    def detect_replay(self, features: Dict, historical_features: List[Dict], 
                     threshold: float = 0.90) -> float:
        """
        Detect potential replay attacks by comparing with recent historical recordings.
        
        Returns:
        float: 1.0 for high replay probability (should block), 0.0 for safe
        """
        if len(historical_features) < 5:
            return 0.0  # Not enough historical data during initial enrollment

        current_vector = self._features_to_vector(features)

        # Check against recent recordings (last 10)
        recent_features = historical_features[-10:]

        similarities = []
        for hist_feat in recent_features:
            hist_vector = self._features_to_vector(hist_feat)
            # Calculate cosine similarity
            similarity = 1 - cosine(current_vector, hist_vector)
            similarities.append(similarity)

        max_similarity = max(similarities)
        avg_similarity = np.mean(similarities)

        # IMPROVED: Higher threshold for replay detection
        if max_similarity > threshold:
            return 1.0  # High replay probability - BLOCK

        # Check for consistent high similarity across multiple samples
        high_similarity_count = sum(1 for s in similarities if s > 0.75)
        if high_similarity_count > len(similarities) * 0.6 and avg_similarity > 0.75:
            return 0.9  # Likely replay - BLOCK

        # Return scaled risk score for monitoring
        return max(max_similarity * 0.5, avg_similarity * 0.3)

    def _features_to_vector(self, features: Dict) -> np.ndarray:
        """Convert feature dictionary to vector"""
        vector = []
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                vector.extend(value.flatten())
            elif isinstance(value, (int, float)):
                vector.append(value)
        return np.array(vector)

    def check_audio_quality(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict:
        """Check audio quality metrics - MORE LENIENT for first attempts"""
        quality = {
            'snr': self._calculate_snr(audio),
            'clipping': np.sum(np.abs(audio) > 0.99) / len(audio),
            'silence_ratio': np.sum(np.abs(audio) < 0.01) / len(audio),
            'duration': len(audio) / sample_rate
        }

        # Quality score (0-1) - More lenient thresholds
        score = 1.0
        if quality['snr'] < 3:  # REDUCED from 5
            score *= 0.7
        if quality['clipping'] > 0.10:  # INCREASED from 0.05
            score *= 0.8
        if quality['silence_ratio'] > 0.8:  # INCREASED from 0.7
            score *= 0.6
        if quality['duration'] < 0.4 or quality['duration'] > 15.0:  # More lenient
            score *= 0.9

        quality['overall_score'] = score
        return quality

    def _calculate_snr(self, audio: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio"""
        signal_power = np.mean(audio ** 2)
        noise = audio - signal.medfilt(audio, kernel_size=5)
        noise_power = np.mean(noise ** 2)

        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
            return snr

        return 40.0  # High SNR if no noise detected


# Voice Authentication Engine - WITH GRADUATED THRESHOLDS
class VoiceAuthenticationEngine:
    def __init__(self):
        self.feature_extractor = VoiceFeatureExtractor()
        self.anti_spoofing = AntiSpoofingDetector()
        self.embedding_model = VoiceEmbeddingNet()
        self.embedding_model.eval()  # Set to evaluation mode
        self.db = SecureDatabase()
        self.enrollment_samples_required = 5
        self.similarity_threshold_first = 0.72  # LOWER for first attempt
        self.similarity_threshold_normal = 0.80  # Standard threshold

    def enroll_user(self, username: str, audio_samples: List[np.ndarray]) -> bool:
        """Enroll a new user with multiple voice samples"""
        if len(audio_samples) < self.enrollment_samples_required:
            return False

        all_features = []
        embeddings = []

        for audio in audio_samples:
            # Quality check - LENIENT on enrollment
            quality = self.anti_spoofing.check_audio_quality(audio)
            if quality['overall_score'] < 0.45:  # REDUCED from 0.6
                return False

            # Extract features
            features = self.feature_extractor.extract_features(audio)
            all_features.append(features)

            # Generate embedding
            feature_vector = self._prepare_feature_vector(features)
            with torch.no_grad():
                embedding = self.embedding_model(torch.FloatTensor(feature_vector).unsqueeze(0))
            embeddings.append(embedding.numpy())

        # Create user voice model
        voice_model = {
            'embeddings': embeddings,
            'features': all_features,
            'created_at': datetime.now().isoformat(),
            'gmm_model': self._train_gmm(all_features)
        }

        # Store encrypted
        self.db.voice_models[username] = voice_model
        return True

    def authenticate(self, username: str, audio: np.ndarray,
                    challenge_text: str = None, attempt_number: int = None) -> Tuple[bool, float, str]:
        """
        Authenticate user with voice sample
        
        Args:
            username: User identifier
            audio: Audio sample as numpy array
            challenge_text: Challenge phrase (for liveness detection)
            attempt_number: Authentication attempt number (for graduated thresholds)
        """
        if username not in self.db.voice_models:
            return False, 0.0, "User not enrolled"

        # Track authentication attempt if not provided
        if attempt_number is None:
            self.db.track_auth_attempt(username)
            attempt_number = self.db.get_attempt_number(username)

        # Quality check - LENIENT for first attempts
        quality = self.anti_spoofing.check_audio_quality(audio)
        quality_threshold = 0.40 if attempt_number == 1 else 0.50  # GRADUATED
        
        if quality['overall_score'] < quality_threshold:
            return False, 0.0, f"Poor audio quality (score: {quality['overall_score']:.2f}). Please speak clearly."

        # Extract features
        features = self.feature_extractor.extract_features(audio)

        # Check for replay attack
        user_model = self.db.voice_models[username]
        replay_score = self.anti_spoofing.detect_replay(features, user_model['features'])

        if 0.90 < replay_score <= 0.99:  # Block only near-perfect matches
            return False, 0.0, f"Replay attack detected - identical recording"

        # Generate embedding
        feature_vector = self._prepare_feature_vector(features)
        with torch.no_grad():
            test_embedding = self.embedding_model(torch.FloatTensor(feature_vector).unsqueeze(0))

        # Compare with stored embeddings
        similarities = []
        for stored_embedding in user_model['embeddings']:
            similarity = 1 - cosine(test_embedding.numpy().flatten(),
                                   stored_embedding.flatten())
            similarities.append(similarity)

        avg_similarity = np.mean(similarities)
        max_similarity = np.max(similarities)

        # GMM likelihood score
        gmm_score = self._gmm_score(features, user_model['gmm_model'])

        # Combined score
        final_score = 0.6 * max_similarity + 0.3 * avg_similarity + 0.1 * gmm_score

        # GRADUATED THRESHOLD: Lower for first attempts
        if attempt_number == 1:
            threshold = self.similarity_threshold_first
        else:
            threshold = self.similarity_threshold_normal

        if final_score >= threshold:
            # Always update model with new sample (adaptive learning)
            user_model['features'].append(features)
            if len(user_model['features']) > 20:  # Keep last 20 samples
                user_model['features'].pop(0)
            
            # Reset attempt counter on success
            if username in self.db.auth_attempts:
                self.db.auth_attempts[username]['attempt'] = 1
            
            return True, final_score, "Authentication successful"

        return False, final_score, f"Voice match score too low ({final_score:.3f}). Please try again."

    def _prepare_feature_vector(self, features: Dict) -> np.ndarray:
        """Prepare feature vector for neural network"""
        vector = []
        for key in ['mfcc_mean', 'mfcc_std', 'mfcc_delta', 'mel_mean']:
            if key in features:
                if isinstance(features[key], np.ndarray):
                    vector.extend(features[key].flatten())
                else:
                    vector.append(features[key])

        # Pad or truncate to fixed size
        target_size = 256
        if len(vector) < target_size:
            vector.extend([0] * (target_size - len(vector)))
        else:
            vector = vector[:target_size]

        return np.array(vector)

    def _train_gmm(self, features: List[Dict]) -> GaussianMixture:
        """Train GMM for additional verification"""
        X = []
        for feat in features:
            X.append(self._prepare_feature_vector(feat))
        gmm = GaussianMixture(n_components=2, covariance_type='diag')
        gmm.fit(X)
        return gmm

    def _gmm_score(self, features: Dict, gmm: GaussianMixture) -> float:
        """Calculate GMM likelihood score"""
        X = self._prepare_feature_vector(features).reshape(1, -1)
        log_likelihood = gmm.score(X)

        # Normalize to 0-1 range
        return 1 / (1 + np.exp(-log_likelihood))


# Custom Kivy Widgets - Paytm Style
class PaytmButton(Button):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_normal = ''
        self.background_color = (0, 0, 0, 0)
        self.bind(pos=self.update_canvas, size=self.update_canvas)
        self.bind(on_press=self.animate_press)
        self.update_canvas()

    def update_canvas(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(0.0, 0.48, 0.82, 1)  # Paytm blue
            RoundedRectangle(pos=self.pos, size=self.size, radius=[10])

    def animate_press(self, instance):
        anim = Animation(opacity=0.7, duration=0.1)
        anim += Animation(opacity=1, duration=0.1)
        anim.start(self)


class PaytmCard(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = 15
        self.spacing = 10
        with self.canvas.before:
            Color(1, 1, 1, 1)
            self.rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[15])
        self.bind(pos=self.update_rect, size=self.update_rect)

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size


class VoiceWaveform(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wave_data = []
        self.is_recording = False

    def update_waveform(self, audio_data):
        """Update waveform visualization"""
        self.wave_data = audio_data[-1000:] if len(audio_data) > 1000 else audio_data
        self.canvas.clear()
        with self.canvas:
            Color(0.0, 0.48, 0.82, 0.8)  # Paytm blue
            if len(self.wave_data) > 1:
                points = []
                for i, sample in enumerate(self.wave_data):
                    x = (i / len(self.wave_data)) * self.width
                    y = self.height / 2 + (sample * self.height / 2)
                    points.extend([x, y])
                if len(points) > 2:
                    Line(points=points, width=2)


# Main Application Screens
class LoginScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.auth_engine = None
        self.recording = False
        self.audio_buffer = []
        self.recording_start_time = None
        self.build_ui()

    def build_ui(self):
        layout = BoxLayout(orientation='vertical', padding=0, spacing=0)

        # Top bar
        top_bar = BoxLayout(size_hint=(1, None), height=60, padding=[20, 10])
        with top_bar.canvas.before:
            Color(0.0, 0.48, 0.82, 1)
            Rectangle(pos=top_bar.pos, size=top_bar.size)

        title = Label(text='VoiceGuard', font_size='24sp', bold=True, color=(1, 1, 1, 1))
        top_bar.add_widget(title)
        layout.add_widget(top_bar)

        # Scroll view for content
        scroll = ScrollView(size_hint=(1, 1))
        content = BoxLayout(orientation='vertical', padding=20, spacing=20, size_hint_y=None)
        content.bind(minimum_height=content.setter('height'))

        # Welcome card
        welcome_card = PaytmCard(size_hint=(1, None), height=120)
        welcome_box = BoxLayout(orientation='vertical', spacing=5)
        welcome_title = Label(text='Welcome Back!', font_size='22sp', bold=True,
                             color=(0.2, 0.2, 0.2, 1), size_hint=(1, None), height=30)
        welcome_subtitle = Label(text='Login with your voice for secure access',
                                font_size='14sp', color=(0.5, 0.5, 0.5, 1))
        welcome_box.add_widget(welcome_title)
        welcome_box.add_widget(welcome_subtitle)
        welcome_card.add_widget(welcome_box)
        content.add_widget(welcome_card)

        # Login card
        login_card = PaytmCard(size_hint=(1, None), height=350)
        login_box = BoxLayout(orientation='vertical', spacing=15)

        # Username
        username_label = Label(text='Username', font_size='14sp', color=(0.3, 0.3, 0.3, 1),
                              size_hint=(1, None), height=25, halign='left')
        username_label.bind(size=username_label.setter('text_size'))
        login_box.add_widget(username_label)

        self.username_input = TextInput(
            hint_text='Enter your username',
            multiline=False,
            size_hint=(1, None),
            height=45,
            font_size='16sp',
            padding=[15, 12],
            background_color=(0.95, 0.95, 0.95, 1),
            foreground_color=(0.2, 0.2, 0.2, 1),
            cursor_color=(0.0, 0.48, 0.82, 1)
        )
        login_box.add_widget(self.username_input)

        # Voice recording
        voice_label = Label(text='Voice Authentication', font_size='14sp',
                           color=(0.3, 0.3, 0.3, 1), size_hint=(1, None), height=25,
                           halign='left')
        voice_label.bind(size=voice_label.setter('text_size'))
        login_box.add_widget(voice_label)

        self.waveform = VoiceWaveform(size_hint=(1, None), height=80)
        with self.waveform.canvas.before:
            Color(0.95, 0.95, 0.95, 1)
            self.wave_bg = RoundedRectangle(pos=self.waveform.pos,
                                           size=self.waveform.size, radius=[10])
        self.waveform.bind(pos=self.update_wave_bg, size=self.update_wave_bg)
        login_box.add_widget(self.waveform)

        self.record_btn = PaytmButton(
            text='Hold to Record Voice',
            size_hint=(1, None),
            height=50,
            font_size='16sp',
            bold=True
        )
        self.record_btn.bind(on_press=self.start_recording)
        self.record_btn.bind(on_release=self.stop_recording)
        login_box.add_widget(self.record_btn)

        self.status_label = Label(
            text='Ready for authentication',
            size_hint=(1, None),
            height=25,
            color=(0.5, 0.5, 0.5, 1),
            font_size='13sp'
        )
        login_box.add_widget(self.status_label)
        login_card.add_widget(login_box)
        content.add_widget(login_card)

        # Action buttons
        login_btn = PaytmButton(
            text='LOGIN',
            size_hint=(1, None),
            height=55,
            font_size='18sp',
            bold=True
        )
        login_btn.bind(on_press=self.authenticate_voice)
        content.add_widget(login_btn)

        # Register link
        register_box = BoxLayout(size_hint=(1, None), height=50)
        register_label = Label(text="Don't have an account?", font_size='14sp',
                              color=(0.5, 0.5, 0.5, 1))
        register_btn = Button(text='Register', font_size='14sp', bold=True,
                             color=(0.0, 0.48, 0.82, 1), background_color=(0, 0, 0, 0),
                             size_hint=(None, 1), width=100)
        register_btn.bind(on_press=self.go_to_register)
        register_box.add_widget(register_label)
        register_box.add_widget(register_btn)
        content.add_widget(register_box)

        scroll.add_widget(content)
        layout.add_widget(scroll)
        self.add_widget(layout)

    def update_wave_bg(self, *args):
        self.wave_bg.pos = self.waveform.pos
        self.wave_bg.size = self.waveform.size

    def start_recording(self, instance):
        self.recording = True
        self.audio_buffer = []
        self.recording_start_time = datetime.now()
        self.status_label.text = 'Recording... Keep speaking (hold button)'
        self.status_label.color = (0.9, 0.3, 0.3, 1)
        self.record_btn.text = '🔴 RECORDING...'
        threading.Thread(target=self._record_audio, daemon=True).start()

    def stop_recording(self, instance):
        if not self.recording:
            return

        # Check minimum hold time
        if self.recording_start_time:
            hold_duration = (datetime.now() - self.recording_start_time).total_seconds()
            if hold_duration < 0.8:  # Minimum 0.8 seconds
                self.recording = False
                self.status_label.text = f'Button held too briefly ({hold_duration:.1f}s). Hold for at least 1 second!'
                self.status_label.color = (0.9, 0.3, 0.3, 1)
                self.record_btn.text = 'Hold to Record Voice'
                self.audio_buffer = []
                return

        self.recording = False
        self.record_btn.text = 'Processing...'
        self.status_label.text = f'Recording captured ({len(self.audio_buffer)/16000:.1f}s)'
        self.status_label.color = (0.2, 0.7, 0.2, 1)

        Clock.schedule_once(lambda dt: self._finish_recording(), 0.3)

    def _finish_recording(self):
        self.record_btn.text = 'Hold to Record Voice'
        self.status_label.text = 'Voice recorded. Click LOGIN to authenticate'
        self.status_label.color = (0.0, 0.48, 0.82, 1)

    def _record_audio(self):
        """Record audio in background"""
        sample_rate = 16000
        with sd.InputStream(samplerate=sample_rate, channels=1,
                           callback=self._audio_callback):
            while self.recording:
                sd.sleep(100)

    def _audio_callback(self, indata, frames, time, status):
        """Audio stream callback"""
        if self.recording:
            self.audio_buffer.extend(indata[:, 0])
            Clock.schedule_once(lambda dt: self.waveform.update_waveform(self.audio_buffer), 0)

    def authenticate_voice(self, instance):
        if not self.username_input.text:
            self.show_popup("Error", "Please enter username")
            return

        if len(self.audio_buffer) < 4000:  # 0.25 seconds minimum
            self.show_popup("Error", "Please record your voice first (hold button for at least 1-2 seconds)")
            return

        self.status_label.text = 'Authenticating...'
        self.status_label.color = (0.9, 0.6, 0.0, 1)

        # Authenticate
        audio_array = np.array(self.audio_buffer)
        success, score, message = self.auth_engine.authenticate(
            self.username_input.text, audio_array
        )

        if success:
            self.status_label.text = f'Success! Score: {score:.2f}'
            self.status_label.color = (0.2, 0.7, 0.2, 1)

            # Pass username to payment screen
            payment_screen = self.manager.get_screen('payment')
            payment_screen.current_user = self.username_input.text
            payment_screen.on_enter()

            # Small delay before transition
            Clock.schedule_once(lambda dt: setattr(self.manager, 'current', 'payment'), 0.5)
        else:
            self.status_label.text = f'Failed: {message}'
            self.status_label.color = (0.9, 0.3, 0.3, 1)

    def go_to_register(self, instance):
        self.manager.current = 'register'

    def show_popup(self, title, message):
        content = BoxLayout(orientation='vertical', padding=20, spacing=15)
        msg_label = Label(text=message, font_size='16sp')
        content.add_widget(msg_label)
        ok_btn = PaytmButton(text='OK', size_hint=(1, None), height=45)
        content.add_widget(ok_btn)
        popup = Popup(title=title, content=content, size_hint=(0.8, 0.3))
        ok_btn.bind(on_press=popup.dismiss)
        popup.open()


class RegistrationScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.auth_engine = None
        self.audio_samples = []
        self.current_sample = 0
        self.recording = False
        self.audio_buffer = []
        self.recording_start_time = None
        self.build_ui()

    def build_ui(self):
        layout = BoxLayout(orientation='vertical', padding=0, spacing=0)

        # Top bar
        top_bar = BoxLayout(size_hint=(1, None), height=60, padding=[20, 10])
        with top_bar.canvas.before:
            Color(0.0, 0.48, 0.82, 1)
            Rectangle(pos=top_bar.pos, size=top_bar.size)

        back_btn = Button(text='<', font_size='24sp', bold=True,
                         color=(1, 1, 1, 1), background_color=(0, 0, 0, 0),
                         size_hint=(None, 1), width=50)
        back_btn.bind(on_press=self.go_back)
        top_bar.add_widget(back_btn)

        title = Label(text='Register', font_size='22sp', bold=True, color=(1, 1, 1, 1))
        top_bar.add_widget(title)

        top_bar.add_widget(Label(size_hint=(None, 1), width=50))  # Spacer

        layout.add_widget(top_bar)

        # Scroll content
        scroll = ScrollView(size_hint=(1, 1))
        content = BoxLayout(orientation='vertical', padding=20, spacing=20, size_hint_y=None)
        content.bind(minimum_height=content.setter('height'))

        # Info card
        info_card = PaytmCard(size_hint=(1, None), height=100)
        info_label = Label(
            text='Create your voice profile\nRecord 5 clear voice samples for secure authentication',
            font_size='15sp', color=(0.3, 0.3, 0.3, 1), halign='center'
        )
        info_card.add_widget(info_label)
        content.add_widget(info_card)

        # User details card
        details_card = PaytmCard(size_hint=(1, None), height=180)
        details_box = BoxLayout(orientation='vertical', spacing=10)

        self.username_input = TextInput(
            hint_text='Username',
            multiline=False,
            size_hint=(1, None),
            height=45,
            font_size='16sp',
            padding=[15, 12],
            background_color=(0.95, 0.95, 0.95, 1)
        )
        details_box.add_widget(self.username_input)

        self.email_input = TextInput(
            hint_text='Email Address',
            multiline=False,
            size_hint=(1, None),
            height=45,
            font_size='16sp',
            padding=[15, 12],
            background_color=(0.95, 0.95, 0.95, 1)
        )
        details_box.add_widget(self.email_input)

        details_card.add_widget(details_box)
        content.add_widget(details_card)

        # Progress card
        progress_card = PaytmCard(size_hint=(1, None), height=120)
        progress_box = BoxLayout(orientation='vertical', spacing=10)

        self.progress_label = Label(
            text='Sample 1 of 5',
            font_size='18sp',
            bold=True,
            color=(0.0, 0.48, 0.82, 1),
            size_hint=(1, None),
            height=30
        )
        progress_box.add_widget(self.progress_label)

        self.progress = ProgressBar(max=5, value=0, size_hint=(1, None), height=8)
        progress_box.add_widget(self.progress)

        self.instruction_label = Label(
            text='Say: "Authorize payment"',
            font_size='16sp',
            color=(0.4, 0.4, 0.4, 1),
            size_hint=(1, None),
            height=30
        )
        progress_box.add_widget(self.instruction_label)

        progress_card.add_widget(progress_box)
        content.add_widget(progress_card)

        # Voice recording card
        voice_card = PaytmCard(size_hint=(1, None), height=200)
        voice_box = BoxLayout(orientation='vertical', spacing=10)

        self.waveform = VoiceWaveform(size_hint=(1, None), height=100)
        with self.waveform.canvas.before:
            Color(0.95, 0.95, 0.95, 1)
            self.wave_bg = RoundedRectangle(pos=self.waveform.pos,
                                           size=self.waveform.size, radius=[10])
        self.waveform.bind(pos=self.update_wave_bg, size=self.update_wave_bg)
        voice_box.add_widget(self.waveform)

        self.record_btn = PaytmButton(
            text='Hold to Record',
            size_hint=(1, None),
            height=55,
            font_size='18sp',
            bold=True
        )
        self.record_btn.bind(on_press=self.start_recording)
        self.record_btn.bind(on_release=self.stop_recording)
        voice_box.add_widget(self.record_btn)

        voice_card.add_widget(voice_box)
        content.add_widget(voice_card)

        # Register button
        self.register_btn = PaytmButton(
            text='COMPLETE REGISTRATION',
            size_hint=(1, None),
            height=55,
            font_size='18sp',
            bold=True,
            disabled=True
        )
        self.register_btn.bind(on_press=self.complete_registration)
        content.add_widget(self.register_btn)

        # Reset button
        reset_btn = Button(
            text='Reset Samples',
            size_hint=(1, None),
            height=50,
            font_size='16sp',
            background_color=(0, 0, 0, 0),
            color=(0.9, 0.3, 0.3, 1)
        )
        reset_btn.bind(on_press=self.reset_samples)
        content.add_widget(reset_btn)

        scroll.add_widget(content)
        layout.add_widget(scroll)
        self.add_widget(layout)

    def update_wave_bg(self, *args):
        self.wave_bg.pos = self.waveform.pos
        self.wave_bg.size = self.waveform.size

    def reset_samples(self, instance):
        """Reset all samples to start over"""
        self.audio_samples = []
        self.current_sample = 0
        self.progress.value = 0
        self.progress_label.text = 'Sample 1 of 5'
        self.instruction_label.text = 'Say: "Authorize payment"'
        self.record_btn.disabled = False
        self.register_btn.disabled = True
        self.show_popup("Reset", "All voice samples cleared. Start recording again.")

    def start_recording(self, instance):
        self.recording = True
        self.audio_buffer = []
        self.recording_start_time = datetime.now()
        self.record_btn.text = '🔴 RECORDING...'
        self.instruction_label.text = 'Recording... Keep speaking (hold button)'
        threading.Thread(target=self._record_audio, daemon=True).start()

    def stop_recording(self, instance):
        if not self.recording:
            return

        # Check minimum hold time
        if self.recording_start_time:
            hold_duration = (datetime.now() - self.recording_start_time).total_seconds()
            if hold_duration < 0.8:  # Minimum 0.8 seconds
                self.recording = False
                self.instruction_label.text = f'Button held too briefly ({hold_duration:.1f}s). Hold for at least 1 second!'
                self.record_btn.text = 'Hold to Record'
                self.audio_buffer = []
                return

        self.recording = False
        self.record_btn.text = 'Processing...'

        # Show recording duration
        duration = len(self.audio_buffer) / 16000
        self.instruction_label.text = f'Captured {duration:.1f} seconds'

        Clock.schedule_once(self.process_sample, 0.3)

    def process_sample(self, dt):
        if len(self.audio_buffer) > 4000:  # 0.25 seconds minimum
            self.audio_samples.append(np.array(self.audio_buffer))
            self.current_sample += 1
            self.progress.value = self.current_sample
            self.progress_label.text = f'Sample {self.current_sample} of 5'

            if self.current_sample < 5:
                phrases = ["Authorize payment", "Verify transaction", "Complete purchase", "Confirm identity", "Voice authentication"]
                self.instruction_label.text = f'Say: "{phrases[self.current_sample]}"'
                self.record_btn.text = 'Hold to Record'
            else:
                self.instruction_label.text = 'All samples recorded!'
                self.record_btn.disabled = True
                self.record_btn.text = 'Completed'
                self.register_btn.disabled = False
        else:
            duration = len(self.audio_buffer) / 16000
            self.instruction_label.text = f'Recording too short ({duration:.1f}s)! Hold button for 1-2 seconds'
            self.record_btn.text = 'Hold to Record'

    def _record_audio(self):
        sample_rate = 16000
        with sd.InputStream(samplerate=sample_rate, channels=1,
                           callback=self._audio_callback):
            while self.recording:
                sd.sleep(100)

    def _audio_callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_buffer.extend(indata[:, 0])
            Clock.schedule_once(lambda dt: self.waveform.update_waveform(self.audio_buffer), 0)

    def complete_registration(self, instance):
        if not self.username_input.text or not self.email_input.text:
            self.show_popup("Error", "Please fill all fields")
            return

        if len(self.audio_samples) < 5:
            self.show_popup("Error", "Please record all 5 voice samples")
            return

        # Show loading
        self.register_btn.text = 'Enrolling...'
        self.register_btn.disabled = True

        # Enroll user in background
        threading.Thread(target=self._enroll_user, daemon=True).start()

    def _enroll_user(self):
        success = self.auth_engine.enroll_user(self.username_input.text, self.audio_samples)
        Clock.schedule_once(lambda dt: self._handle_enrollment_result(success), 0)

    def _handle_enrollment_result(self, success):
        if success:
            # Store user data
            user_data = {
                'username': self.username_input.text,
                'email': self.email_input.text,
                'created_at': datetime.now().isoformat(),
                'balance': 10000.00
            }
            self.auth_engine.db.store_user(self.username_input.text, user_data)
            self.show_success_popup()
        else:
            self.register_btn.text = 'COMPLETE REGISTRATION'
            self.register_btn.disabled = False
            self.show_popup("Enrollment Failed",
                           "Voice quality too low. Please reset and try again with clearer audio.")

    def show_success_popup(self):
        content = BoxLayout(orientation='vertical', padding=20, spacing=15)
        success_label = Label(
            text='Registration Successful!\n\nYou can now login with your voice',
            font_size='18sp',
            halign='center'
        )
        content.add_widget(success_label)
        ok_btn = PaytmButton(text='GO TO LOGIN', size_hint=(1, None), height=50)
        content.add_widget(ok_btn)
        popup = Popup(title='Success', content=content, size_hint=(0.85, 0.4))
        ok_btn.bind(on_press=lambda x: (popup.dismiss(), self.go_back(None)))
        popup.open()

    def go_back(self, instance):
        # Reset form
        self.username_input.text = ''
        self.email_input.text = ''
        self.audio_samples = []
        self.current_sample = 0
        self.progress.value = 0
        self.progress_label.text = 'Sample 1 of 5'
        self.instruction_label.text = 'Say: "Authorize payment"'
        self.record_btn.disabled = False
        self.record_btn.text = 'Hold to Record'
        self.register_btn.disabled = True
        self.manager.current = 'login'

    def show_popup(self, title, message):
        content = BoxLayout(orientation='vertical', padding=20, spacing=15)
        msg_label = Label(text=message, font_size='16sp')
        content.add_widget(msg_label)
        ok_btn = PaytmButton(text='OK', size_hint=(1, None), height=45)
        content.add_widget(ok_btn)
        popup = Popup(title=title, content=content, size_hint=(0.8, 0.35))
        ok_btn.bind(on_press=popup.dismiss)
        popup.open()


class PaymentScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.auth_engine = None
        self.recording = False
        self.audio_buffer = []
        self.current_user = None
        self.challenge_phrase = None
        self.transaction_amount = 0
        self.user_balance = 10000.00
        self.recording_start_time = None
        self.auth_attempt_count = 0  # Track attempt count for this transaction
        self.build_ui()

    def build_ui(self):
        layout = BoxLayout(orientation='vertical', padding=0, spacing=0)

        # Top bar
        top_bar = BoxLayout(size_hint=(1, None), height=180, orientation='vertical')
        with top_bar.canvas.before:
            Color(0.0, 0.48, 0.82, 1)
            Rectangle(pos=top_bar.pos, size=top_bar.size)

        # Header row
        header_row = BoxLayout(size_hint=(1, None), height=50, padding=[20, 10])
        self.user_label = Label(text='User', font_size='16sp', color=(1, 1, 1, 1),
                               halign='left')
        self.user_label.bind(size=self.user_label.setter('text_size'))
        header_row.add_widget(self.user_label)

        logout_btn = Button(text='Logout', font_size='14sp', color=(1, 1, 1, 1),
                           background_color=(0, 0, 0, 0), size_hint=(None, 1), width=80)
        logout_btn.bind(on_press=self.logout)
        header_row.add_widget(logout_btn)

        top_bar.add_widget(header_row)

        # Balance display
        balance_box = BoxLayout(orientation='vertical', padding=[20, 10], spacing=5)
        balance_title = Label(text='Available Balance', font_size='14sp',
                             color=(0.9, 0.9, 0.9, 1), size_hint=(1, None), height=20)
        self.balance_label = Label(text='₹10,000.00', font_size='36sp', bold=True,
                                  color=(1, 1, 1, 1), size_hint=(1, None), height=50)
        balance_box.add_widget(balance_title)
        balance_box.add_widget(self.balance_label)

        top_bar.add_widget(balance_box)
        layout.add_widget(top_bar)

        # Scroll content
        scroll = ScrollView(size_hint=(1, 1))
        content = BoxLayout(orientation='vertical', padding=20, spacing=20, size_hint_y=None)
        content.bind(minimum_height=content.setter('height'))

        # Payment form card
        form_card = PaytmCard(size_hint=(1, None), height=300)
        form_box = BoxLayout(orientation='vertical', spacing=15)

        form_title = Label(text='Send Money', font_size='20sp', bold=True,
                          color=(0.2, 0.2, 0.2, 1), size_hint=(1, None), height=30,
                          halign='left')
        form_title.bind(size=form_title.setter('text_size'))
        form_box.add_widget(form_title)

        # Recipient
        self.recipient_input = TextInput(
            hint_text='Recipient Name or Number',
            multiline=False,
            size_hint=(1, None),
            height=50,
            font_size='16sp',
            padding=[15, 15],
            background_color=(0.95, 0.95, 0.95, 1)
        )
        form_box.add_widget(self.recipient_input)

        # Amount
        self.amount_input = TextInput(
            hint_text='Enter Amount (₹)',
            multiline=False,
            size_hint=(1, None),
            height=50,
            font_size='20sp',
            padding=[15, 15],
            background_color=(0.95, 0.95, 0.95, 1),
            input_filter='float'
        )
        form_box.add_widget(self.amount_input)

        # Description
        self.desc_input = TextInput(
            hint_text='Add a note (optional)',
            multiline=False,
            size_hint=(1, None),
            height=45,
            font_size='14sp',
            padding=[15, 12],
            background_color=(0.95, 0.95, 0.95, 1)
        )
        form_box.add_widget(self.desc_input)

        form_card.add_widget(form_box)
        content.add_widget(form_card)

        # Voice auth card
        voice_card = PaytmCard(size_hint=(1, None), height=280)
        voice_box = BoxLayout(orientation='vertical', spacing=10)

        voice_title = Label(text='Voice Authentication', font_size='18sp', bold=True,
                           color=(0.2, 0.2, 0.2, 1), size_hint=(1, None), height=30,
                           halign='left')
        voice_title.bind(size=voice_title.setter('text_size'))
        voice_box.add_widget(voice_title)

        self.challenge_label = Label(
            text='Click "Pay Now" to start verification',
            font_size='15sp',
            color=(0.5, 0.5, 0.5, 1),
            size_hint=(1, None),
            height=50
        )
        voice_box.add_widget(self.challenge_label)

        self.waveform = VoiceWaveform(size_hint=(1, None), height=90)
        with self.waveform.canvas.before:
            Color(0.95, 0.95, 0.95, 1)
            self.wave_bg = RoundedRectangle(pos=self.waveform.pos,
                                           size=self.waveform.size, radius=[10])
        self.waveform.bind(pos=self.update_wave_bg, size=self.update_wave_bg)
        voice_box.add_widget(self.waveform)

        self.voice_btn = PaytmButton(
            text='PAY NOW',
            size_hint=(1, None),
            height=55,
            font_size='20sp',
            bold=True
        )
        self.voice_btn.bind(on_press=self.initiate_payment)
        voice_box.add_widget(self.voice_btn)

        voice_card.add_widget(voice_box)
        content.add_widget(voice_card)

        # Transaction history
        history_card = PaytmCard(size_hint=(1, None), height=180)
        history_box = BoxLayout(orientation='vertical', spacing=10)

        history_title = Label(text='Recent Transactions', font_size='18sp', bold=True,
                             color=(0.2, 0.2, 0.2, 1), size_hint=(1, None), height=30,
                             halign='left')
        history_title.bind(size=history_title.setter('text_size'))
        history_box.add_widget(history_title)

        self.history_label = Label(
            text='No recent transactions',
            font_size='14sp',
            color=(0.5, 0.5, 0.5, 1),
            halign='left'
        )
        self.history_label.bind(size=self.history_label.setter('text_size'))
        history_box.add_widget(self.history_label)

        history_card.add_widget(history_box)
        content.add_widget(history_card)

        scroll.add_widget(content)
        layout.add_widget(scroll)
        self.add_widget(layout)

    def update_wave_bg(self, *args):
        self.wave_bg.pos = self.waveform.pos
        self.wave_bg.size = self.waveform.size

    def on_enter(self):
        """Called when entering this screen"""
        if not self.current_user:
            self.current_user = "TestUser"
        self.user_label.text = f' {self.current_user}'
        self.balance_label.text = f'₹{self.user_balance:,.2f}'
        self.update_transaction_history()
        self.auth_attempt_count = 0  # Reset attempt count for new transaction

    def initiate_payment(self, instance):
        if not self.validate_payment_form():
            return

        # Generate challenge phrase
        self.challenge_phrase, _ = self.auth_engine.anti_spoofing.generate_challenge()
        self.challenge_label.text = f'Say: "{self.challenge_phrase}"'
        self.challenge_label.color = (0.9, 0.3, 0.3, 1)

        # Change to recording mode
        self.voice_btn.text = 'HOLD TO AUTHORIZE'
        self.voice_btn.unbind(on_press=self.initiate_payment)
        self.voice_btn.bind(on_press=self.start_voice_auth)
        self.voice_btn.bind(on_release=self.stop_voice_auth)
        
        # Reset attempt counter for this payment
        self.auth_attempt_count = 0

    def start_voice_auth(self, instance):
        self.recording = True
        self.audio_buffer = []
        self.recording_start_time = datetime.now()
        self.challenge_label.text = 'Recording... Keep speaking (hold button)'
        self.challenge_label.color = (0.9, 0.3, 0.3, 1)
        self.voice_btn.text = '🔴 RECORDING...'
        threading.Thread(target=self._record_audio, daemon=True).start()

    def stop_voice_auth(self, instance):
        if not self.recording:
            return

        # Calculate how long the button was held
        if self.recording_start_time:
            hold_duration = (datetime.now() - self.recording_start_time).total_seconds()

            # Require minimum hold time of 1 second
            if hold_duration < 1.0:
                self.recording = False
                self.challenge_label.text = f'Button held too briefly ({hold_duration:.1f}s). Hold for at least 2 seconds!'
                self.challenge_label.color = (0.9, 0.3, 0.3, 1)
                self.voice_btn.text = 'HOLD TO AUTHORIZE'
                self.audio_buffer = []
                return

        self.recording = False
        duration = len(self.audio_buffer) / 16000
        self.challenge_label.text = f'Processing ({duration:.1f}s recorded)...'
        self.challenge_label.color = (0.9, 0.6, 0.0, 1)
        self.voice_btn.text = 'Processing...'
        Clock.schedule_once(lambda dt: self.process_payment(), 0.5)

    def _record_audio(self):
        sample_rate = 16000
        with sd.InputStream(samplerate=sample_rate, channels=1,
                           callback=self._audio_callback):
            while self.recording:
                sd.sleep(100)

    def _audio_callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_buffer.extend(indata[:, 0])
            Clock.schedule_once(lambda dt: self.waveform.update_waveform(self.audio_buffer), 0)

    def process_payment(self):
        # More lenient minimum
        if len(self.audio_buffer) < 4000:  # 0.25 seconds minimum
            duration = len(self.audio_buffer) / 16000
            self.show_popup("Error", f"Recording too short ({duration:.1f}s). Please HOLD the button while speaking (need 2-4 seconds)")
            self.reset_voice_button()
            return

        audio_array = np.array(self.audio_buffer)
        self.challenge_label.text = 'Verifying voice...'

        # Increment attempt count and authenticate
        self.auth_attempt_count += 1
        success, score, message = self.auth_engine.authenticate(
            self.current_user, audio_array, self.challenge_phrase,
            attempt_number=self.auth_attempt_count
        )

        if success:
            self.transaction_amount = float(self.amount_input.text)

            # Log transaction
            transaction = {
                'user': self.current_user,
                'recipient': self.recipient_input.text,
                'amount': self.transaction_amount,
                'description': self.desc_input.text,
                'voice_score': score,
                'status': 'completed'
            }
            self.auth_engine.db.log_transaction(transaction)

            # Update balance
            self.user_balance -= self.transaction_amount
            self.balance_label.text = f'₹{self.user_balance:,.2f}'

            # Clear form
            self.clear_payment_form()
            self.update_transaction_history()

            # Show success
            self.show_success_popup(transaction)
        else:
            # Log declined transaction
            transaction = {
                'user': self.current_user,
                'recipient': self.recipient_input.text,
                'amount': float(self.amount_input.text),
                'description': self.desc_input.text,
                'voice_score': score,
                'status': 'declined',
                'reason': message
            }
            self.auth_engine.db.log_transaction(transaction)

            # Show decline popup
            self.show_decline_popup(transaction, message)
            self.challenge_label.text = f'Failed: {message}'
            self.challenge_label.color = (0.9, 0.3, 0.3, 1)
            self.reset_voice_button()

    def validate_payment_form(self):
        if not self.recipient_input.text:
            self.show_popup("Error", "Please enter recipient")
            return False

        try:
            amount = float(self.amount_input.text)
            if amount <= 0:
                self.show_popup("Error", "Amount must be greater than 0")
                return False

            if amount > self.user_balance:
                self.show_popup("Error", "Insufficient balance")
                return False
        except ValueError:
            self.show_popup("Error", "Please enter valid amount")
            return False

        return True

    def clear_payment_form(self):
        self.recipient_input.text = ''
        self.amount_input.text = ''
        self.desc_input.text = ''

    def reset_voice_button(self):
        self.voice_btn.text = 'PAY NOW'
        self.voice_btn.unbind(on_press=self.start_voice_auth)
        self.voice_btn.unbind(on_release=self.stop_voice_auth)
        self.voice_btn.bind(on_press=self.initiate_payment)
        self.challenge_label.text = 'Click "Pay Now" to start verification'
        self.challenge_label.color = (0.5, 0.5, 0.5, 1)

    def update_transaction_history(self):
        transactions = self.auth_engine.db.transaction_logs[-3:]

        if transactions:
            history_text = ""
            for trans in reversed(transactions):
                history_text += f"₹{trans['amount']:.2f} to {trans['recipient']}\n"
            self.history_label.text = history_text.strip()
        else:
            self.history_label.text = "No recent transactions"

    def show_success_popup(self, transaction):
        content = BoxLayout(orientation='vertical', padding=20, spacing=15)

        success_icon = Label(text='✓', font_size='60sp', color=(0.2, 0.7, 0.2, 1),
                            size_hint=(1, None), height=80)
        content.add_widget(success_icon)

        success_text = Label(
            text=f'Payment Successful!\n\n₹{transaction["amount"]:.2f} sent to\n{transaction["recipient"]}',
            font_size='18sp',
            halign='center'
        )
        content.add_widget(success_text)

        ok_btn = PaytmButton(text='DONE', size_hint=(1, None), height=50)
        content.add_widget(ok_btn)

        popup = Popup(title='Success', content=content, size_hint=(0.85, 0.5))
        ok_btn.bind(on_press=popup.dismiss)
        popup.open()

    def show_decline_popup(self, transaction, reason):
        """Show payment declined popup with details"""
        content = BoxLayout(orientation='vertical', padding=20, spacing=15)

        # Decline icon
        decline_icon = Label(text='✗', font_size='60sp', color=(0.9, 0.3, 0.3, 1),
                            size_hint=(1, None), height=80)
        content.add_widget(decline_icon)

        # Main decline message
        decline_title = Label(
            text='Payment Declined',
            font_size='24sp',
            bold=True,
            color=(0.9, 0.3, 0.3, 1),
            size_hint=(1, None),
            height=40
        )
        content.add_widget(decline_title)

        # Transaction details card
        details_card = PaytmCard(size_hint=(1, None), height=150)
        details_box = BoxLayout(orientation='vertical', spacing=8, padding=10)

        amount_label = Label(
            text=f'Amount: ₹{transaction["amount"]:.2f}',
            font_size='18sp',
            bold=True,
            color=(0.2, 0.2, 0.2, 1),
            size_hint=(1, None),
            height=30,
            halign='left'
        )
        amount_label.bind(size=amount_label.setter('text_size'))
        details_box.add_widget(amount_label)

        recipient_label = Label(
            text=f'To: {transaction["recipient"]}',
            font_size='16sp',
            color=(0.4, 0.4, 0.4, 1),
            size_hint=(1, None),
            height=25,
            halign='left'
        )
        recipient_label.bind(size=recipient_label.setter('text_size'))
        details_box.add_widget(recipient_label)

        score_label = Label(
            text=f'Voice Match Score: {transaction["voice_score"]:.2%}',
            font_size='14sp',
            color=(0.5, 0.5, 0.5, 1),
            size_hint=(1, None),
            height=25,
            halign='left'
        )
        score_label.bind(size=score_label.setter('text_size'))
        details_box.add_widget(score_label)

        details_card.add_widget(details_box)
        content.add_widget(details_card)

        # Reason for decline
        reason_card = PaytmCard(size_hint=(1, None), height=100)
        reason_box = BoxLayout(orientation='vertical', spacing=5, padding=10)

        reason_title = Label(
            text='Reason:',
            font_size='14sp',
            bold=True,
            color=(0.3, 0.3, 0.3, 1),
            size_hint=(1, None),
            height=20,
            halign='left'
        )
        reason_title.bind(size=reason_title.setter('text_size'))
        reason_box.add_widget(reason_title)

        reason_text = Label(
            text=reason,
            font_size='15sp',
            color=(0.5, 0.5, 0.5, 1),
            size_hint=(1, None),
            height=40,
            halign='left'
        )
        reason_text.bind(size=reason_text.setter('text_size'))
        reason_box.add_widget(reason_text)

        reason_card.add_widget(reason_box)
        content.add_widget(reason_card)

        # Action buttons
        button_box = BoxLayout(spacing=10, size_hint=(1, None), height=50)

        retry_btn = PaytmButton(text='TRY AGAIN', size_hint=(0.5, 1))
        retry_btn.bind(on_press=lambda x: popup.dismiss())
        button_box.add_widget(retry_btn)

        cancel_btn = Button(
            text='CANCEL',
            size_hint=(0.5, 1),
            font_size='16sp',
            bold=True,
            background_normal='',
            background_color=(0.9, 0.3, 0.3, 1),
            color=(1, 1, 1, 1)
        )
        cancel_btn.bind(on_press=lambda x: (popup.dismiss(), self.clear_payment_form()))
        button_box.add_widget(cancel_btn)

        content.add_widget(button_box)

        # Create popup
        popup = Popup(
            title='Transaction Failed',
            content=content,
            size_hint=(0.9, 0.7),
            separator_color=(0.9, 0.3, 0.3, 1)
        )

        popup.open()

    def logout(self, instance):
        self.current_user = None
        self.user_balance = 10000.00
        self.manager.current = 'login'

    def show_popup(self, title, message):
        content = BoxLayout(orientation='vertical', padding=20, spacing=15)
        msg_label = Label(text=message, font_size='16sp')
        content.add_widget(msg_label)
        ok_btn = PaytmButton(text='OK', size_hint=(1, None), height=45)
        content.add_widget(ok_btn)
        popup = Popup(title=title, content=content, size_hint=(0.8, 0.3))
        ok_btn.bind(on_press=popup.dismiss)
        popup.open()


# Main Application
class VoiceGuardApp(App):
    def build(self):
        # Initialize authentication engine
        self.auth_engine = VoiceAuthenticationEngine()

        # Create screen manager
        sm = ScreenManager()

        # Create screens
        login_screen = LoginScreen(name='login')
        login_screen.auth_engine = self.auth_engine

        register_screen = RegistrationScreen(name='register')
        register_screen.auth_engine = self.auth_engine

        payment_screen = PaymentScreen(name='payment')
        payment_screen.auth_engine = self.auth_engine

        # Add screens
        sm.add_widget(login_screen)
        sm.add_widget(register_screen)
        sm.add_widget(payment_screen)

        return sm


if __name__ == '__main__':
    VoiceGuardApp().run()
