#Section 1
!pip install \
  torch torchvision \
  sentence-transformers \
  faiss-cpu \
  datasets \
  Pillow \
  numpy pandas \
  scikit-learn \
  matplotlib seaborn \
  requests \
  codecarbon \
  gradio \
  tqdm -q

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2: IMPORTS & SETUP
# ──────────────────────────────────────────────────────────────────────────────
import os
import json
import time
import random
import smtplib
import ssl
import warnings
import logging
import requests
import numpy as np
import pandas as pd
import torch

from PIL import Image
from io import BytesIO
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

from tqdm import tqdm
from collections import Counter

warnings.filterwarnings("ignore")

# ── Visualisation ──────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Metrics ───────────────────────────────────────────────────────────────────
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score,
    roc_curve, auc, roc_auc_score, accuracy_score,
)
from sklearn.preprocessing import label_binarize

# ── Gaussian Process Regression ───────────────────────────────────────────────
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, WhiteKernel, ConstantKernel as C,
)
from sklearn.preprocessing import StandardScaler

# ── CO₂ Tracking ──────────────────────────────────────────────────────────────
from codecarbon import EmissionsTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Device: {device}")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3: CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

class Config:
    """Central configuration."""

    # ── OpenWeatherMap (optional) ──────────────────────────────────────────────
    OPENWEATHER_API_KEY: str = ""

    # ── Dataset ───────────────────────────────────────────────────────────────
    DATASET_NAME:  str = "GVJahnavi/Plant_village_subset"
    TRAIN_SAMPLES: int = 9057
    VAL_SAMPLES:   int = 2265
    TEST_SAMPLES:  int = 2916

    # ── CLIP ──────────────────────────────────────────────────────────────────
    CLIP_MODEL:    str = "clip-ViT-B-32"
    EMBEDDING_DIM: int = 512
    TOP_K_SIMILAR: int = 5

    # ── Default location (New Delhi) ──────────────────────────────────────────
    DEFAULT_LAT:  float = 28.6139
    DEFAULT_LON:  float = 77.2090
    DEFAULT_CITY: str   = "New Delhi"

    # ── Outputs ───────────────────────────────────────────────────────────────
    OUTPUT_DIR:     str = "/content/outputs"
    EMISSIONS_FILE: str = "/content/outputs/emissions.csv"

    # ── Email recipient ───────────────────────────────────────────────────────
    NOTIFY_EMAIL: str = "tarunjit05@gmail.com"


config = Config()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4: LOAD DATASET
# ──────────────────────────────────────────────────────────────────────────────

from datasets import load_dataset

print("📥 Loading PlantVillage dataset...")

try:
    dataset = load_dataset(config.DATASET_NAME)
    label_names: List[str] = dataset["train"].features["label"].names
    num_classes = len(label_names)

    if "validation" not in dataset or "test" not in dataset:
        full_train  = dataset["train"]
        total       = len(full_train)
        test_ratio  = config.TEST_SAMPLES  / total
        val_ratio   = config.VAL_SAMPLES   / total
        train_ratio = config.TRAIN_SAMPLES / total

        split1   = full_train.train_test_split(test_size=test_ratio, seed=42)
        test_set = split1["test"]

        val_size = val_ratio / (train_ratio + val_ratio)
        split2   = split1["train"].train_test_split(test_size=val_size, seed=42)

        dataset["train"]      = split2["train"]
        dataset["validation"] = split2["test"]
        dataset["test"]       = test_set

    print(f"✅ Dataset loaded  — {num_classes} classes")
    print(f"   Train : {len(dataset['train']):,}")
    print(f"   Val   : {len(dataset['validation']):,}")
    print(f"   Test  : {len(dataset['test']):,}")
    print(f"   Labels (first 5): {label_names[:5]}...")

except Exception as exc:
    raise RuntimeError(f"❌ Dataset loading failed: {exc}")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5: DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PlantImageData:
    image_id:     str
    image_pil:    Image.Image
    label:        int
    label_name:   str
    disease_type: str
    plant_type:   str
    is_healthy:   bool = False
    embedding:    Optional[np.ndarray] = None


@dataclass
class DiagnosisResult:
    disease:         str
    plant_species:   str
    confidence:      float
    is_healthy:      bool
    predicted_label: int = -1
    evidence:        List[str] = field(default_factory=list)
    agent_name:      str = ""


@dataclass
class WeatherData:
    city:                   str
    temperature_c:          float
    humidity_pct:           float
    condition:              str
    wind_kmh:               float
    soil_moisture_estimate: str
    season:                 str


@dataclass
class RiskAssessment:
    risk_score:          float
    risk_level:          str
    uncertainty:         float
    confidence_interval: Tuple[float, float]
    mitigation_actions:  List[str]
    spread_probability:  float

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6: DATASET PROCESSOR
# ──────────────────────────────────────────────────────────────────────────────

class PlantVillageProcessor:
    def __init__(self, dataset, label_names: List[str]):
        self.dataset     = dataset
        self.label_names = label_names
        self.mapping     = self._build_mapping()
        self.all_plants: List[str] = sorted(
            {v["plant"] for v in self.mapping.values()}
        )
        print(f"🌿 Plants ({len(self.all_plants)}): {self.all_plants}")

    def _build_mapping(self) -> Dict[str, Dict]:
        mapping: Dict[str, Dict] = {}
        for label in self.label_names:
            if "___" in label:
                plant_part, disease_part = label.split("___", 1)
            else:
                parts        = label.split("_")
                plant_part   = parts[0]
                disease_part = "_".join(parts[1:])
            disease_readable = disease_part.replace("_", " ").strip()
            mapping[label] = {
                "plant":     plant_part,
                "disease":   disease_readable or "Unknown",
                "healthy":   "healthy" in disease_readable.lower(),
                "raw_label": label,
            }
        return mapping

    def load_split(
        self, split: str = "train", max_samples: int = None
    ) -> List[PlantImageData]:
        items: List[PlantImageData] = []
        ds = self.dataset[split]
        n  = min(max_samples, len(ds)) if max_samples else len(ds)
        for i in tqdm(range(n), desc=f"Loading {split}"):
            sample          = ds[i]
            img: Image.Image = sample["image"]
            label: int       = sample["label"]
            label_name: str  = self.label_names[label]
            meta             = self.mapping[label_name]
            items.append(PlantImageData(
                image_id=f"{split}_{i}",
                image_pil=img,
                label=label,
                label_name=label_name,
                disease_type=meta["disease"],
                plant_type=meta["plant"],
                is_healthy=meta["healthy"],
            ))
        return items


processor = PlantVillageProcessor(dataset, label_names)

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7: CLIP EMBEDDING GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

from sentence_transformers import SentenceTransformer

class CLIPEmbeddingGenerator:
    def __init__(self, model_name: str = config.CLIP_MODEL):
        print(f"🔧 Loading CLIP model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.dim: int = 512

    def encode_image(self, img: Image.Image) -> np.ndarray:
        emb = self.model.encode(img, convert_to_tensor=True)
        emb = emb / emb.norm()
        return emb.cpu().numpy().astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        emb = self.model.encode(text, convert_to_tensor=True)
        emb = emb / emb.norm()
        return emb.cpu().numpy().astype(np.float32)


clip_generator = CLIPEmbeddingGenerator()

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8: FAISS VECTOR DATABASE
# ──────────────────────────────────────────────────────────────────────────────

import faiss

class PlantVillageVectorDB:
    def __init__(self, dim: int = config.EMBEDDING_DIM):
        self.dim   = dim
        self.data: List[PlantImageData] = []
        self.index = faiss.IndexFlatIP(dim)

    def add(self, image_data: List[PlantImageData],
            embedder: CLIPEmbeddingGenerator):
        batch_embs: List[np.ndarray] = []
        for item in tqdm(image_data, desc="Embedding & indexing"):
            emb            = embedder.encode_image(item.image_pil)
            item.embedding = emb
            batch_embs.append(emb)
            self.data.append(item)
        embs = np.array(batch_embs, dtype=np.float32)
        faiss.normalize_L2(embs)
        self.index.add(embs)
        print(f"✅ Indexed {len(batch_embs)} images (total: {len(self.data)})")

    def search(self, query_emb: np.ndarray,
               k: int = 5) -> List[Tuple[PlantImageData, float]]:
        q = query_emb.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q)
        sims, idxs = self.index.search(q, k)
        return [
            (self.data[i], float(sims[0][j]))
            for j, i in enumerate(idxs[0])
            if i < len(self.data)
        ]


vector_db = PlantVillageVectorDB(dim=clip_generator.dim)

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9: BUILD VECTOR DB INDEX
# ──────────────────────────────────────────────────────────────────────────────

print("\n📚 Building vector database from training data...")
train_images = processor.load_split("train", max_samples=config.TRAIN_SAMPLES)
vector_db.add(train_images, clip_generator)

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 10: RAG CORE SYSTEM
# ──────────────────────────────────────────────────────────────────────────────

class PlantDiseaseRAG:
    def __init__(self, vector_db: PlantVillageVectorDB,
                 clip_generator: CLIPEmbeddingGenerator):
        self.vector_db      = vector_db
        self.clip_generator = clip_generator

    def find_similar_cases(self, image_pil: Image.Image,
                           k: int = config.TOP_K_SIMILAR):
        query_emb = self.clip_generator.encode_image(image_pil)
        return self.vector_db.search(query_emb, k)

    def analyze_similar_cases(
        self, cases: List[Tuple[PlantImageData, float]]
    ) -> Dict[str, Any]:
        if not cases:
            return {
                "top_diseases": [], "top_plants": [], "label_votes": {},
                "total_cases": 0, "avg_similarity": 0.0, "is_healthy": False,
            }
        disease_scores: Dict[str, float] = {}
        plant_scores:   Dict[str, float] = {}
        label_votes:    Dict[str, float] = {}
        health_score = 0.0
        total_sim    = sum(sim for _, sim in cases)
        for item, sim in cases:
            weight = sim / (total_sim + 1e-8)
            disease_scores[item.disease_type] = \
                disease_scores.get(item.disease_type, 0) + sim
            plant_scores[item.plant_type] = \
                plant_scores.get(item.plant_type, 0) + sim
            label_votes[item.label_name] = \
                label_votes.get(item.label_name, 0) + weight
            if item.is_healthy:
                health_score += weight
        top_diseases = sorted(
            disease_scores.items(), key=lambda x: x[1], reverse=True
        )[:3]
        top_plants = sorted(
            plant_scores.items(), key=lambda x: x[1], reverse=True
        )[:2]
        return {
            "top_diseases": [
                {"disease": d, "score": s, "confidence": s / total_sim}
                for d, s in top_diseases
            ],
            "top_plants": [
                {"plant": p, "score": s, "confidence": s / total_sim}
                for p, s in top_plants
            ],
            "label_votes":    label_votes,
            "total_cases":    len(cases),
            "avg_similarity": total_sim / len(cases),
            "is_healthy":     health_score > 0.5,
        }

    def create_rag_context(self, analysis: Dict[str, Any]) -> str:
        if analysis["total_cases"] == 0:
            return "No similar cases found."
        lines = [
            f"[RAG] {analysis['total_cases']} similar cases "
            f"(avg sim: {analysis['avg_similarity']:.3f})\n",
            "=== Top Diseases ===",
        ]
        for d in analysis["top_diseases"]:
            lines.append(f"  • {d['disease']} ({d['confidence']*100:.1f}%)")
        lines.append("\n=== Top Plants ===")
        for p in analysis["top_plants"]:
            lines.append(f"  • {p['plant']} ({p['confidence']*100:.1f}%)")
        return "\n".join(lines)


rag_system = PlantDiseaseRAG(vector_db, clip_generator)

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 11: AGENT 1 — DISEASE PREDICTION
# ──────────────────────────────────────────────────────────────────────────────

class DiseasePredictionAgent:
    AGENT_NAME = "DiseasePredictionAgent"

    def __init__(self, rag_system: PlantDiseaseRAG,
                 processor: PlantVillageProcessor):
        self.rag       = rag_system
        self.processor = processor

    def run(self, image_pil: Image.Image,
            rag_analysis: Optional[Dict] = None) -> DiagnosisResult:
        similar_cases = self.rag.find_similar_cases(
            image_pil, k=config.TOP_K_SIMILAR
        )
        if rag_analysis is None:
            rag_analysis = self.rag.analyze_similar_cases(similar_cases)
        label_votes = rag_analysis.get("label_votes", {})
        if not label_votes:
            return DiagnosisResult(
                disease="Unknown", plant_species="Unknown",
                confidence=0.0, is_healthy=False,
                predicted_label=-1, agent_name=self.AGENT_NAME,
                evidence=["No similar cases found."],
            )
        best_label, best_weight = max(label_votes.items(), key=lambda x: x[1])
        meta       = self.processor.mapping.get(best_label, {})
        disease    = meta.get("disease", best_label)
        plant      = meta.get("plant", "Unknown")
        is_healthy = meta.get("healthy", False)
        predicted_label = (
            self.processor.label_names.index(best_label)
            if best_label in self.processor.label_names else -1
        )
        top_sim    = similar_cases[0][1] if similar_cases else 0.0
        avg_sim    = rag_analysis.get("avg_similarity", 0.0)
        confidence = float(np.clip(
            0.4 * top_sim + 0.4 * avg_sim + 0.2 * best_weight, 0.0, 1.0
        ))
        return DiagnosisResult(
            disease=disease,
            plant_species=plant,
            confidence=confidence,
            is_healthy=is_healthy,
            predicted_label=predicted_label,
            evidence=[
                f"Top label: '{best_label}' (weight: {best_weight:.3f})",
                f"Top similarity: {top_sim:.3f}",
            ],
            agent_name=self.AGENT_NAME,
        )

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 12: AGENT 2 — WEATHER & CROP RECOMMENDATION
# ──────────────────────────────────────────────────────────────────────────────

class WeatherCropAgent:
    AGENT_NAME = "WeatherCropAgent"

    PLANT_PROFILES: Dict[str, Dict] = {
        "apple":      {"temp_range": (15, 25), "humidity": (50, 80),  "seasons": ["spring", "autumn"]},
        "blueberry":  {"temp_range": (18, 28), "humidity": (60, 80),  "seasons": ["spring", "summer"]},
        "cherry":     {"temp_range": (15, 25), "humidity": (50, 75),  "seasons": ["spring", "summer"]},
        "corn":       {"temp_range": (20, 32), "humidity": (55, 75),  "seasons": ["summer"]},
        "grape":      {"temp_range": (20, 30), "humidity": (40, 70),  "seasons": ["summer", "autumn"]},
        "orange":     {"temp_range": (22, 35), "humidity": (55, 80),  "seasons": ["autumn", "winter"]},
        "peach":      {"temp_range": (18, 28), "humidity": (50, 70),  "seasons": ["spring", "summer"]},
        "pepper":     {"temp_range": (22, 32), "humidity": (50, 75),  "seasons": ["summer"]},
        "potato":     {"temp_range": (15, 25), "humidity": (60, 80),  "seasons": ["spring", "autumn"]},
        "raspberry":  {"temp_range": (15, 25), "humidity": (60, 80),  "seasons": ["spring", "summer"]},
        "soybean":    {"temp_range": (20, 30), "humidity": (55, 75),  "seasons": ["summer"]},
        "squash":     {"temp_range": (20, 35), "humidity": (50, 70),  "seasons": ["summer"]},
        "strawberry": {"temp_range": (15, 25), "humidity": (60, 80),  "seasons": ["spring"]},
        "tomato":     {"temp_range": (20, 30), "humidity": (50, 70),  "seasons": ["summer", "autumn"]},
    }

    SEASON_MAP = {
        (12, 1, 2):  "winter",
        (3, 4, 5):   "spring",
        (6, 7, 8):   "summer",
        (9, 10, 11): "autumn",
    }

    def __init__(self, api_key: str = "",
                 all_plants: Optional[List[str]] = None):
        self.api_key    = api_key or config.OPENWEATHER_API_KEY
        self.all_plants = [p.lower() for p in (all_plants or processor.all_plants)]
        self._cache: Dict[str, WeatherData] = {}

    def _get_season(self, month: int) -> str:
        for months, season in self.SEASON_MAP.items():
            if month in months:
                return season
        return "unknown"

    def _estimate_soil_moisture(self, humidity: float, condition: str) -> str:
        cond = condition.lower()
        if any(w in cond for w in ["rain", "drizzle", "storm"]):
            return "High (wet)"
        if humidity > 75: return "Moderate-High"
        if humidity > 50: return "Moderate"
        return "Low (dry)"

    def fetch_weather(self, lat: float = config.DEFAULT_LAT,
                      lon: float = config.DEFAULT_LON) -> WeatherData:
        key = f"{lat:.2f},{lon:.2f}"
        if key in self._cache:
            return self._cache[key]
        if not self.api_key:
            return self._simulated_weather()
        try:
            url  = (f"https://api.openweathermap.org/data/2.5/weather"
                    f"?lat={lat}&lon={lon}&appid={self.api_key}&units=metric")
            resp = requests.get(url, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            import datetime
            month  = datetime.datetime.now().month
            season = self._get_season(month)
            w = WeatherData(
                city=data.get("name", config.DEFAULT_CITY),
                temperature_c=data["main"]["temp"],
                humidity_pct=data["main"]["humidity"],
                condition=data["weather"][0]["description"],
                wind_kmh=data["wind"]["speed"] * 3.6,
                soil_moisture_estimate=self._estimate_soil_moisture(
                    data["main"]["humidity"],
                    data["weather"][0]["description"],
                ),
                season=season,
            )
            self._cache[key] = w
            return w
        except Exception as exc:
            logger.warning(f"Weather API error: {exc} — using simulated data")
            return self._simulated_weather()

    def _simulated_weather(self) -> WeatherData:
        import datetime
        month  = datetime.datetime.now().month
        season = self._get_season(month)
        t = {"spring": 18, "summer": 28, "autumn": 20, "winter": 8}.get(season, 22)
        h = {"spring": 65, "summer": 60, "autumn": 70, "winter": 75}.get(season, 65)
        return WeatherData(
            city=config.DEFAULT_CITY,
            temperature_c=float(t),
            humidity_pct=float(h),
            condition="partly cloudy (simulated)",
            wind_kmh=12.0,
            soil_moisture_estimate=self._estimate_soil_moisture(float(h), ""),
            season=season,
        )

    def recommend_crops(self, weather: WeatherData) -> Dict[str, Any]:
        scored: List[Tuple[str, float, List[str]]] = []
        for plant in self.all_plants:
            profile = self.PLANT_PROFILES.get(plant)
            if profile is None:
                continue
            score = 0.0; reasons: List[str] = []
            t_min, t_max = profile["temp_range"]
            if t_min <= weather.temperature_c <= t_max:
                score += 40; reasons.append(f"Temp {weather.temperature_c:.1f}°C ideal")
            else:
                delta  = min(abs(weather.temperature_c - t_min),
                             abs(weather.temperature_c - t_max))
                score += max(0, 40 - delta * 3)
            h_min, h_max = profile["humidity"]
            if h_min <= weather.humidity_pct <= h_max:
                score += 35; reasons.append(f"Humidity {weather.humidity_pct:.0f}% ideal")
            else:
                delta  = min(abs(weather.humidity_pct - h_min),
                             abs(weather.humidity_pct - h_max))
                score += max(0, 35 - delta * 2)
            if weather.season in profile["seasons"]:
                score += 25; reasons.append(f"Season '{weather.season}' ideal")
            scored.append((plant.capitalize(), score, reasons))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_crops = [
            {"plant": p, "suitability_score": round(s, 1),
             "max_score": 100, "reasons": r}
            for p, s, r in scored[:5]
        ]
        return {
            "weather": {
                "city":          weather.city,
                "temperature_c": weather.temperature_c,
                "humidity_pct":  weather.humidity_pct,
                "condition":     weather.condition,
                "wind_kmh":      weather.wind_kmh,
                "soil_moisture": weather.soil_moisture_estimate,
                "season":        weather.season,
            },
            "top_recommended_crops": top_crops,
            "best_crop": top_crops[0]["plant"] if top_crops else "Unknown",
        }

    def run(self, lat: float = config.DEFAULT_LAT,
            lon: float = config.DEFAULT_LON) -> Dict[str, Any]:
        weather = self.fetch_weather(lat, lon)
        return self.recommend_crops(weather)

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 12b: AGENT 3 — GAUSSIAN PROCESS REGRESSION RISK MITIGATION
# ──────────────────────────────────────────────────────────────────────────────

class GPRiskMitigationAgent:
    AGENT_NAME = "GPRiskMitigationAgent"

    DISEASE_SEVERITY: Dict[str, float] = {
        "healthy":                0.00,
        "early blight":           0.45,
        "late blight":            0.85,
        "bacterial spot":         0.60,
        "leaf mold":              0.50,
        "septoria leaf spot":     0.55,
        "spider mites":           0.40,
        "target spot":            0.50,
        "mosaic virus":           0.90,
        "yellow leaf curl virus": 0.88,
        "powdery mildew":         0.55,
        "black rot":              0.75,
        "esca":                   0.80,
        "leaf blight":            0.65,
        "common rust":            0.60,
        "gray leaf spot":         0.55,
        "northern leaf blight":   0.65,
        "haunglongbing":          0.95,
        "apple scab":             0.70,
        "cedar apple rust":       0.65,
        "fire blight":            0.80,
    }

    SEASON_ENC = {"spring": 0.25, "summer": 0.75, "autumn": 0.50, "winter": 0.10}
    SOIL_ENC   = {"Low (dry)": 0.2, "Moderate": 0.5,
                  "Moderate-High": 0.7, "High (wet)": 0.9}

    MITIGATION_DB: Dict[str, List[str]] = {
        "Low": [
            "Monitor plants weekly for early symptoms.",
            "Maintain good field drainage to prevent waterlogging.",
        ],
        "Moderate": [
            "Apply preventive fungicide every 14 days.",
            "Remove and dispose of affected leaves immediately.",
            "Increase plant spacing to improve air circulation.",
            "Avoid overhead irrigation; switch to drip.",
        ],
        "High": [
            "Apply systemic fungicide within 48 hours.",
            "Isolate infected plants from healthy sections of the field.",
            "Notify neighbouring farm owners of potential spread.",
            "Reduce irrigation frequency to lower canopy humidity.",
        ],
        "Critical": [
            "Remove and destroy ALL infected plants immediately.",
            "Fumigate or solarise soil before replanting.",
            "Quarantine the field for 3–4 weeks minimum.",
            "Contact local agricultural extension officer.",
            "Document the outbreak for the national disease-tracking registry.",
        ],
    }

    def __init__(self):
        kernel = (
            C(1.0, (1e-3, 1e3))
            * RBF(length_scale=[1.0] * 7, length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1e-3)
        )
        self.gpr = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=5,
            normalize_y=True, random_state=42,
        )
        self.scaler  = StandardScaler()
        self._fitted = False
        self._fit_on_synthetic_data()

    def _synthetic_risk(self, temp, hum, season_enc, dis_conf,
                        is_diseased, wind, soil_enc, disease_name) -> float:
        base = (self.DISEASE_SEVERITY.get(disease_name.lower(), 0.5)
                if is_diseased else 0.0)
        env = 0.0
        if temp > 30 or temp < 10: env += 0.10
        if hum > 80:               env += 0.15
        if soil_enc > 0.7:         env += 0.10
        if wind > 20:              env += 0.05
        return float(np.clip(base + env + dis_conf * 0.20, 0.0, 1.0))

    def _fit_on_synthetic_data(self):
        rng = np.random.default_rng(0)
        n   = 400
        temps    = rng.uniform(5,  40,  n)
        hums     = rng.uniform(20, 100, n)
        seasons  = rng.choice(list(self.SEASON_ENC.values()), n)
        confs    = rng.uniform(0,  1,   n)
        diseased = rng.integers(0, 2,   n)
        winds    = rng.uniform(0,  35,  n)
        soils    = rng.choice(list(self.SOIL_ENC.values()), n)
        d_keys   = list(self.DISEASE_SEVERITY.keys())
        diseases = [rng.choice(d_keys) for _ in range(n)]
        X, y = [], []
        for i in range(n):
            score = self._synthetic_risk(
                temps[i], hums[i], seasons[i], confs[i],
                bool(diseased[i]), winds[i], soils[i], diseases[i],
            )
            X.append([temps[i], hums[i], seasons[i], confs[i],
                      float(diseased[i]), winds[i], soils[i]])
            y.append(score)
        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.float32)
        X_sc  = self.scaler.fit_transform(X_arr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gpr.fit(X_sc, y_arr)
        self._fitted = True
        print(f"✅ GPR Risk Model fitted  |  kernel: {self.gpr.kernel_}")

    def _features(self, weather: WeatherData,
                  diagnosis: DiagnosisResult) -> np.ndarray:
        return np.array([[
            weather.temperature_c,
            weather.humidity_pct,
            self.SEASON_ENC.get(weather.season, 0.3),
            diagnosis.confidence,
            0.0 if diagnosis.is_healthy else 1.0,
            weather.wind_kmh,
            self.SOIL_ENC.get(weather.soil_moisture_estimate, 0.5),
        ]], dtype=np.float32)

    @staticmethod
    def _level(score: float) -> str:
        if score < 0.25: return "Low"
        if score < 0.50: return "Moderate"
        if score < 0.75: return "High"
        return "Critical"

    def _spread_prob(self, risk: float, weather: WeatherData) -> float:
        base  = risk * 0.60
        humid = (weather.humidity_pct - 50) / 100 * 0.25
        wind  = min(weather.wind_kmh / 50, 0.15)
        return float(np.clip(base + humid + wind, 0.0, 1.0))

    def run(self, weather: WeatherData,
            diagnosis: DiagnosisResult) -> RiskAssessment:
        X_sc = self.scaler.transform(self._features(weather, diagnosis))
        mu, sigma   = self.gpr.predict(X_sc, return_std=True)
        risk_score  = float(np.clip(mu[0],    0.0, 1.0))
        uncertainty = float(np.clip(sigma[0], 0.0, 0.50))
        ci_lo = float(np.clip(risk_score - 1.96 * uncertainty, 0.0, 1.0))
        ci_hi = float(np.clip(risk_score + 1.96 * uncertainty, 0.0, 1.0))
        level   = self._level(risk_score)
        actions = self.MITIGATION_DB[level].copy()
        if not diagnosis.is_healthy:
            actions.append(
                f"Target treatment: {diagnosis.disease} "
                f"in {diagnosis.plant_species}."
            )
        return RiskAssessment(
            risk_score=risk_score, risk_level=level,
            uncertainty=uncertainty,
            confidence_interval=(ci_lo, ci_hi),
            mitigation_actions=actions,
            spread_probability=self._spread_prob(risk_score, weather),
        )

    def run_batch(
        self,
        weather_list:   List[WeatherData],
        diagnosis_list: List[DiagnosisResult],
    ) -> List[RiskAssessment]:
        X_rows = np.vstack([
            self._features(w, d)
            for w, d in zip(weather_list, diagnosis_list)
        ]).astype(np.float32)
        X_sc = self.scaler.transform(X_rows)
        mus, sigmas = self.gpr.predict(X_sc, return_std=True)
        results: List[RiskAssessment] = []
        for i, (w, d) in enumerate(zip(weather_list, diagnosis_list)):
            risk_score  = float(np.clip(mus[i],    0.0, 1.0))
            uncertainty = float(np.clip(sigmas[i], 0.0, 0.50))
            ci_lo = float(np.clip(risk_score - 1.96 * uncertainty, 0.0, 1.0))
            ci_hi = float(np.clip(risk_score + 1.96 * uncertainty, 0.0, 1.0))
            level = self._level(risk_score)
            results.append(RiskAssessment(
                risk_score=risk_score, risk_level=level,
                uncertainty=uncertainty,
                confidence_interval=(ci_lo, ci_hi),
                mitigation_actions=self.MITIGATION_DB[level].copy(),
                spread_probability=self._spread_prob(risk_score, w),
            ))
        return results

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 12c: EMAIL NOTIFICATION  (SMTP  +  Resend HTTP  fallback)
# ──────────────────────────────────────────────────────────────────────────────
#
#  METHOD A — smtplib (Python built-in)
#    Gmail: generate a 16-char App Password at
#           https://myaccount.google.com/apppasswords  (requires 2-FA)
#    Brevo / Zoho / Outlook: use their SMTP credentials
#
#  METHOD B — Resend HTTP API  (https://resend.com)
#    Free tier: 3 000 emails / month, no credit card needed
#    Sign up → API Keys → Create → paste below
#
#  The dispatcher tries METHOD A first, falls back to B, then logs a warning.
# ──────────────────────────────────────────────────────────────────────────────

EMAIL_CONFIG: Dict[str, str] = {
    # ── METHOD A : SMTP ───────────────────────────────────────────────────────
    "smtp_host":       "smtp.gmail.com",   # or smtp-relay.brevo.com / smtp.zoho.com
    "smtp_port":       "587",              # STARTTLS port
    "sender_email":    "",                 # ← your sending address
    "sender_password": "",                 # ← Gmail App Password (NOT your login)

    # ── METHOD B : Resend ─────────────────────────────────────────────────────
    "resend_api_key":  "",                 # ← Resend API key  (re_xxxx…)
    # Free test sender — works without a verified domain on Resend's free tier:
    "resend_from":     "Plant-AI <onboarding@resend.dev>",
}


def _send_via_smtp(subject: str, body: str,
                   recipient: str = config.NOTIFY_EMAIL) -> Dict[str, str]:
    """Send through any STARTTLS SMTP server (free)."""
    if not EMAIL_CONFIG["sender_password"]:
        raise ValueError(
            "SMTP password not set in EMAIL_CONFIG['sender_password']. "
            "For Gmail use an App Password from "
            "https://myaccount.google.com/apppasswords"
        )
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = EMAIL_CONFIG["sender_email"]
    msg["To"]      = recipient
    msg.attach(MIMEText(body, "plain"))

    ctx = ssl.create_default_context()
    with smtplib.SMTP(EMAIL_CONFIG["smtp_host"],
                      int(EMAIL_CONFIG["smtp_port"])) as server:
        server.ehlo()
        server.starttls(context=ctx)
        server.login(EMAIL_CONFIG["sender_email"],
                     EMAIL_CONFIG["sender_password"])
        server.sendmail(EMAIL_CONFIG["sender_email"],
                        recipient, msg.as_string())

    logger.info(f"✅ SMTP email sent → {recipient}")
    return {"status": "sent", "method": "smtp", "recipient": recipient}


def _send_via_resend(subject: str, body: str,
                     recipient: str = config.NOTIFY_EMAIL) -> Dict[str, str]:
    """Send through the Resend HTTP API (free tier: 3 000 emails/month)."""
    api_key = EMAIL_CONFIG.get("resend_api_key", "")
    if not api_key:
        raise ValueError(
            "Resend API key not set in EMAIL_CONFIG['resend_api_key']. "
            "Sign up free at https://resend.com"
        )
    resp = requests.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        },
        json={
            "from":    EMAIL_CONFIG["resend_from"],
            "to":      [recipient],
            "subject": subject,
            "text":    body,
        },
        timeout=15,
    )
    resp.raise_for_status()
    email_id = resp.json().get("id", "unknown")
    logger.info(f"✅ Resend email sent → {recipient}  (id: {email_id})")
    return {"status": "sent", "method": "resend",
            "recipient": recipient, "email_id": email_id}


def _dispatch_email(subject: str, body: str,
                    recipient: str = config.NOTIFY_EMAIL) -> Dict[str, str]:
    """
    Smart dispatcher:
      1. SMTP   — if EMAIL_CONFIG['sender_password'] is set.
      2. Resend — if EMAIL_CONFIG['resend_api_key'] is set.
      3. Graceful skip — logs a warning, never crashes the pipeline.
    """
    if EMAIL_CONFIG.get("sender_password", "").strip():
        try:
            return _send_via_smtp(subject, body, recipient)
        except Exception as exc:
            logger.warning(f"SMTP failed ({exc}); trying Resend…")

    if EMAIL_CONFIG.get("resend_api_key", "").strip():
        try:
            return _send_via_resend(subject, body, recipient)
        except Exception as exc:
            logger.warning(f"Resend failed: {exc}")
            return {"status": "failed", "error": str(exc)}

    logger.warning(
        "⚠️  Email skipped — no credentials configured.\n"
        "    Set EMAIL_CONFIG['sender_password'] for SMTP (Gmail App Password)\n"
        "    or EMAIL_CONFIG['resend_api_key'] for Resend (free at resend.com)."
    )
    return {"status": "skipped", "reason": "no_credentials_configured"}


# ── Email body formatters ──────────────────────────────────────────────────────

def _format_diagnosis_email_body(result: Dict, risk: RiskAssessment) -> str:
    """
    Full per-image report: diagnosis + GPR risk + mitigation actions
    + weather + crop recommendations.  Sent when the user uploads an image.
    """
    fd = result["final_diagnosis"]
    w  = result["weather_agent"]["result"]["weather"]
    cr = result["weather_agent"]["result"]["top_recommended_crops"]
    ra = result["risk_assessment"]

    crop_lines = "\n".join(
        f"  {i+1}. {c['plant']}  (score {c['suitability_score']}/100)"
        for i, c in enumerate(cr[:3])
    )
    actions = "\n".join(f"  • {a}" for a in risk.mitigation_actions)

    # Build RAG disease candidates section
    rag_diseases = result.get("rag_analysis", {}).get("top_diseases", [])
    rag_section  = ""
    if rag_diseases:
        rag_lines  = "\n".join(
            f"  {i+1}. {d['disease']}  ({d['confidence']*100:.1f}% confidence)"
            for i, d in enumerate(rag_diseases)
        )
        rag_section = f"""
RAG — DISEASE CANDIDATES (CLIP + FAISS)
----------------------------------------
{rag_lines}
"""

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""
PLANT DISEASE DIAGNOSIS & MITIGATION REPORT
=============================================
Generated : {timestamp}
Recipient : {config.NOTIFY_EMAIL}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIAGNOSIS SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Plant Species  : {fd['plant_species']}
Disease        : {fd['disease']}
Health Status  : {'Healthy ✅' if fd['is_healthy'] else 'Diseased ⚠️'}
Confidence     : {fd['confidence']*100:.1f}%
{rag_section}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GPR RISK ASSESSMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Risk Level     : {ra['risk_level']}
Risk Score     : {ra['risk_score']:.3f}
95% CI         : [{ra['confidence_interval'][0]:.3f} – {ra['confidence_interval'][1]:.3f}]
GPR Uncertainty: ± {ra['uncertainty']:.3f}
Spread Prob.   : {ra['spread_probability']*100:.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOMMENDED MITIGATION ACTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{actions}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CURRENT WEATHER  ({w['city']})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Temperature    : {w['temperature_c']:.1f} °C
Humidity       : {w['humidity_pct']} %
Condition      : {w['condition'].title()}
Wind Speed     : {w['wind_kmh']:.1f} km/h
Soil Moisture  : {w['soil_moisture']}
Season         : {w['season'].capitalize()}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOP CROP RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{crop_lines}

--
Sent automatically by Agentic-RAG Plant Disease Diagnosis System v3
(SMTP / Resend HTTP — no Anthropic SDK dependency)
"""


def _format_evaluation_email_body(metrics: Dict) -> str:
    """Full model evaluation report sent after test-set evaluation completes."""
    gpr = metrics.get("gpr_eval", {})
    gpr_block = ""
    if gpr:
        level_str = "  |  ".join(
            f"{k}: {v}" for k, v in gpr.get("level_counts", {}).items()
        )
        gpr_block = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GPR EVALUATION (Test Set)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Mean Risk Score   : {gpr.get('mean_risk', 0):.4f}
Mean Uncertainty  : {gpr.get('mean_uncertainty', 0):.4f}
Mean Spread Prob. : {gpr.get('mean_spread', 0):.4f}
Risk Distribution : {level_str}
"""

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""
MODEL EVALUATION REPORT — Agentic-RAG Plant Disease System v3
==============================================================
Generated : {timestamp}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OVERALL METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Accuracy              : {metrics['accuracy']:.4f}
F1-Score (Macro)      : {metrics['f1_score_macro']:.4f}
F1-Score (Weighted)   : {metrics['f1_score_weighted']:.4f}
ROC-AUC (Macro)       : {metrics['roc_auc_macro']:.4f}
CO2 Emissions         : {metrics['co2_emissions_kg']:.8f} kg
{gpr_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET SPLIT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Train   : {config.TRAIN_SAMPLES:,} samples
Val     : {config.VAL_SAMPLES:,}   samples
Test    : {config.TEST_SAMPLES:,}  samples
Classes : {num_classes}

--
Sent automatically by Agentic-RAG Plant Disease Diagnosis System v3
(SMTP / Resend HTTP — no Anthropic SDK dependency)
"""


def send_diagnosis_email(result: Dict, risk: RiskAssessment,
                          recipient: str = config.NOTIFY_EMAIL) -> Dict:
    """Send per-image diagnosis + mitigation email when user uploads an image."""
    plant   = result["final_diagnosis"]["plant_species"]
    disease = result["final_diagnosis"]["disease"]
    level   = risk.risk_level
    subject = (f"[{level.upper()} RISK] Plant Disease Alert — "
               f"{plant} · {disease}")
    body    = _format_diagnosis_email_body(result, risk)
    try:
        return _dispatch_email(subject, body, recipient)
    except Exception as exc:
        logger.warning(f"Diagnosis email failed: {exc}")
        return {"status": "failed", "error": str(exc)}


def send_evaluation_email(metrics: Dict,
                           recipient: str = config.NOTIFY_EMAIL) -> Dict:
    """Send full evaluation report email after test-set run completes."""
    subject = "[Model Report] Agentic-RAG Evaluation Complete"
    body    = _format_evaluation_email_body(metrics)
    try:
        return _dispatch_email(subject, body, recipient)
    except Exception as exc:
        logger.warning(f"Evaluation email failed: {exc}")
        return {"status": "failed", "error": str(exc)}

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 13: AGENTIC ORCHESTRATOR
# ──────────────────────────────────────────────────────────────────────────────

class AgenticOrchestrator:
    """
    Coordinates all three agents.
    When send_email=True, automatically dispatches the diagnosis +
    mitigation email for the uploaded image via _dispatch_email.
    """

    def __init__(self, rag_system, disease_agent, weather_agent, gpr_agent):
        self.rag           = rag_system
        self.disease_agent = disease_agent
        self.weather_agent = weather_agent
        self.gpr_agent     = gpr_agent

    def diagnose(
        self,
        image_pil:  Image.Image,
        use_rag:    bool  = True,
        lat:        float = config.DEFAULT_LAT,
        lon:        float = config.DEFAULT_LON,
        send_email: bool  = False,
    ) -> Dict[str, Any]:

        rag_analysis: Dict = {}
        if use_rag:
            _, rag_analysis = self._step_rag(image_pil)

        diagnosis    = self.disease_agent.run(image_pil, rag_analysis)
        weather_crop = self.weather_agent.run(lat, lon)
        weather_obj  = self.weather_agent.fetch_weather(lat, lon)
        risk: RiskAssessment = self.gpr_agent.run(weather_obj, diagnosis)

        result = {
            "final_diagnosis": {
                "disease":         diagnosis.disease,
                "plant_species":   diagnosis.plant_species,
                "is_healthy":      diagnosis.is_healthy,
                "confidence":      diagnosis.confidence,
                "predicted_label": diagnosis.predicted_label,
            },
            "disease_agent": {
                "name":     diagnosis.agent_name,
                "result":   {
                    "disease":       diagnosis.disease,
                    "plant_species": diagnosis.plant_species,
                    "confidence":    diagnosis.confidence,
                },
                "evidence": diagnosis.evidence,
            },
            "weather_agent": {
                "name":   self.weather_agent.AGENT_NAME,
                "result": weather_crop,
            },
            "risk_assessment": {
                "agent":               self.gpr_agent.AGENT_NAME,
                "risk_level":          risk.risk_level,
                "risk_score":          round(risk.risk_score,          4),
                "uncertainty":         round(risk.uncertainty,         4),
                "confidence_interval": [
                    round(risk.confidence_interval[0], 4),
                    round(risk.confidence_interval[1], 4),
                ],
                "spread_probability":  round(risk.spread_probability, 4),
                "mitigation_actions":  risk.mitigation_actions,
            },
            "rag_analysis": rag_analysis,
            "risk_obj":     risk,
        }

        # ── Auto-send diagnosis + mitigation email when image is uploaded ──
        if send_email:
            email_status = send_diagnosis_email(result, risk)
            result["email_status"] = email_status
            logger.info(f"📧 Diagnosis email: {email_status.get('status','?').upper()}"
                        f"  via {email_status.get('method','')}")

        return result

    def _step_rag(self, image_pil: Image.Image):
        similar_cases = self.rag.find_similar_cases(image_pil)
        analysis      = self.rag.analyze_similar_cases(similar_cases)
        return similar_cases, analysis


disease_agent = DiseasePredictionAgent(rag_system, processor)
weather_agent = WeatherCropAgent(all_plants=processor.all_plants)
gpr_agent     = GPRiskMitigationAgent()
orchestrator  = AgenticOrchestrator(
    rag_system, disease_agent, weather_agent, gpr_agent
)

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 14: EVALUATION & METRICS
# ──────────────────────────────────────────────────────────────────────────────

class ModelEvaluator:
    def __init__(self, orchestrator, processor, label_names):
        self.orchestrator = orchestrator
        self.processor    = processor
        self.label_names  = label_names
        self.num_classes  = len(label_names)

    def evaluate_on_test_set(
        self, test_data: List[PlantImageData]
    ) -> Dict[str, Any]:
        print("\n" + "="*60)
        print("  📊 EVALUATING MODEL ON TEST SET")
        print("="*60)

        y_true, y_pred, y_scores = [], [], []
        predictions_list: List[Dict] = []

        tracker = EmissionsTracker(
            project_name="PlantDiseaseRAG_Evaluation",
            output_dir=config.OUTPUT_DIR,
            output_file="emissions.csv",
            allow_multiple_runs=True,
        )
        tracker.start()

        for item in tqdm(test_data, desc="Evaluating"):
            result     = self.orchestrator.diagnose(item.image_pil, use_rag=True)
            true_label = item.label
            pred_label = result["final_diagnosis"]["predicted_label"]
            confidence = result["final_diagnosis"]["confidence"]

            y_true.append(true_label)
            y_pred.append(pred_label if pred_label != -1 else 0)

            prob_dist = np.zeros(self.num_classes)
            if pred_label != -1:
                prob_dist[pred_label] = confidence
            y_scores.append(prob_dist)

            predictions_list.append({
                "image_id":        item.image_id,
                "true_label":      true_label,
                "true_label_name": item.label_name,
                "pred_label":      pred_label,
                "pred_label_name": (self.label_names[pred_label]
                                    if pred_label != -1 else "Unknown"),
                "confidence":      confidence,
                "risk_score":      result["risk_assessment"]["risk_score"],
                "risk_level":      result["risk_assessment"]["risk_level"],
                "uncertainty":     result["risk_assessment"]["uncertainty"],
                "spread_prob":     result["risk_assessment"]["spread_probability"],
            })

        emissions = tracker.stop()

        y_true   = np.array(y_true)
        y_pred   = np.array(y_pred)
        y_scores = np.array(y_scores)

        accuracy    = accuracy_score(y_true, y_pred)
        f1_macro    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        cm          = confusion_matrix(y_true, y_pred)
        report      = classification_report(
            y_true, y_pred, target_names=self.label_names,
            output_dict=True, zero_division=0,
        )

        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
        try:
            roc_auc = roc_auc_score(
                y_true_bin, y_scores, average="macro", multi_class="ovr"
            )
        except Exception:
            roc_auc = 0.0

        all_risks    = [p["risk_score"]  for p in predictions_list]
        all_unc      = [p["uncertainty"] for p in predictions_list]
        all_spread   = [p["spread_prob"] for p in predictions_list]
        level_counts = Counter(p["risk_level"] for p in predictions_list)

        gpr_eval = {
            "risk_scores":      all_risks,
            "uncertainties":    all_unc,
            "spread_probs":     all_spread,
            "level_counts":     dict(level_counts),
            "mean_risk":        float(np.mean(all_risks)),
            "mean_uncertainty": float(np.mean(all_unc)),
            "mean_spread":      float(np.mean(all_spread)),
        }

        metrics = {
            "accuracy":              accuracy,
            "f1_score_macro":        f1_macro,
            "f1_score_weighted":     f1_weighted,
            "roc_auc_macro":         roc_auc,
            "confusion_matrix":      cm,
            "classification_report": report,
            "predictions":           predictions_list,
            "y_true":                y_true,
            "y_pred":                y_pred,
            "y_scores":              y_scores,
            "y_true_binarized":      y_true_bin,
            "co2_emissions_kg":      emissions if emissions else 0.0,
            "gpr_eval":              gpr_eval,
        }

        print(f"\n✅ Evaluation complete!")
        print(f"   Accuracy           : {accuracy:.4f}")
        print(f"   F1-Score (Macro)   : {f1_macro:.4f}")
        print(f"   F1-Score (Weighted): {f1_weighted:.4f}")
        print(f"   ROC-AUC (Macro)    : {roc_auc:.4f}")
        print(f"   GPR Mean Risk      : {gpr_eval['mean_risk']:.4f}")
        print(f"   GPR Mean Uncert.   : {gpr_eval['mean_uncertainty']:.4f}")
        if emissions:
            print(f"   CO2 Emissions      : {emissions:.6f} kg")

        return metrics

    # ── Plots ─────────────────────────────────────────────────────────────────

    def plot_confusion_matrix(self, cm, save_path=None):
        figsize = max(20, min(40, self.num_classes // 2))
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        mask    = cm_norm == 0
        sns.heatmap(
            cm_norm, annot=False, fmt=".2f", cmap="YlOrRd",
            xticklabels=self.label_names, yticklabels=self.label_names,
            cbar_kws={"label": "Normalised Count", "shrink": 0.8},
            mask=mask, vmin=0, vmax=1, ax=ax,
        )
        ax.set_title("Confusion Matrix (Normalised)", fontsize=16,
                     fontweight="bold", pad=20)
        ax.set_ylabel("True Label",      fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=14, fontweight="bold")
        plt.xticks(rotation=90, ha="center", fontsize=8)
        plt.yticks(rotation=0,  fontsize=8)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def plot_f1_scores(self, report, save_path=None):
        classes   = [k for k in report
                     if k not in ["accuracy", "macro avg", "weighted avg"]]
        f1_scores = [report[c]["f1-score"] for c in classes]
        idx       = np.argsort(f1_scores)[::-1]
        classes   = [classes[i]   for i in idx]
        f1_scores = [f1_scores[i] for i in idx]
        fig_h = max(10, len(classes) * 0.3)
        fig, ax = plt.subplots(figsize=(14, fig_h))
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(classes)))
        bars   = ax.barh(range(len(classes)), f1_scores,
                         color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("F1-Score",          fontsize=14, fontweight="bold")
        ax.set_ylabel("Disease Class",     fontsize=14, fontweight="bold")
        ax.set_title("F1-Score per Class", fontsize=16, fontweight="bold")
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes, fontsize=9)
        ax.set_xlim(0, 1.1)
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.axvline(x=0.5, color="red", linestyle="--",
                   alpha=0.5, label="Threshold 0.5")
        for bar, score in zip(bars, f1_scores):
            ax.text(bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{score:.3f}", ha="left", va="center",
                    fontsize=8, fontweight="bold")
        ax.legend(loc="lower right")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def plot_roc_curves(self, y_true_bin, y_scores, save_path=None):
        fpr, tpr, roc_auc_per = {}, {}, {}
        for i in range(self.num_classes):
            try:
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
                roc_auc_per[i] = auc(fpr[i], tpr[i])
            except Exception:
                fpr[i], tpr[i], roc_auc_per[i] = [0, 1], [0, 1], 0.0
        fig, ax = plt.subplots(figsize=(14, 10))
        n_plot  = min(15, self.num_classes)
        top_cls = sorted(roc_auc_per.items(),
                         key=lambda x: x[1], reverse=True)[:n_plot]
        colors  = plt.cm.tab20(np.linspace(0, 1, len(top_cls)))
        for (i, auc_v), col in zip(top_cls, colors):
            lbl = (self.label_names[i][:40] + "…"
                   if len(self.label_names[i]) > 40 else self.label_names[i])
            ax.plot(fpr[i], tpr[i], color=col, lw=2, alpha=0.8, label=lbl)
        ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random (AUC=0.5)")
        all_fpr  = np.unique(
            np.concatenate([fpr[i] for i in range(self.num_classes)])
        )
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.num_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= self.num_classes
        macro_auc = auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr, color="deeppink", lw=3, linestyle=":",
                label=f"Macro-avg (AUC={macro_auc:.3f})")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=14, fontweight="bold")
        ax.set_ylabel("True Positive Rate",  fontsize=14, fontweight="bold")
        ax.set_title("ROC Curves — Multi-class", fontsize=16, fontweight="bold")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def plot_co2_emissions(self, emissions_kg, save_path=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        ax1.bar(["Model Evaluation"], [emissions_kg],
                color="#e74c3c", edgecolor="black", linewidth=2, alpha=0.8)
        ax1.set_ylabel("CO₂ Emissions (kg)", fontsize=12, fontweight="bold")
        ax1.set_title(f"CO₂ during Evaluation\n{emissions_kg:.6f} kg",
                      fontsize=14, fontweight="bold")
        ax1.grid(axis="y", alpha=0.3, linestyle="--")
        ax1.text(0, emissions_kg * 1.1, f"{emissions_kg:.6f} kg",
                 ha="center", fontsize=12, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        comparisons = {
            "This Model":     emissions_kg,
            "Smartphone\n1h": 0.007,
            "LED Bulb\n1h":   0.008,
            "Car\n1 km":      0.120,
            "Air\n1 km":      0.255,
            "Desktop\n1h":    0.055,
        }
        items  = sorted(comparisons.items(), key=lambda x: x[1])
        labels, values = zip(*items)
        colors = ["#e74c3c" if "Model" in l else "#3498db" for l in labels]
        bars2  = ax2.barh(labels, values, color=colors,
                          edgecolor="black", linewidth=1.5, alpha=0.8)
        ax2.set_xlabel("CO₂ Emissions (kg)", fontsize=12, fontweight="bold")
        ax2.set_title("Comparison with Daily Activities",
                      fontsize=14, fontweight="bold")
        ax2.grid(axis="x", alpha=0.3, linestyle="--")
        for bar, val in zip(bars2, values):
            ax2.text(val + val * 0.05,
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}" if val < 0.01 else f"{val:.3f}",
                     ha="left", va="center", fontsize=10, fontweight="bold")
        if emissions_kg > 0:
            trees = emissions_kg / 0.021
            fig.text(0.5, 0.02,
                     f"🌳 {trees:.1f} tree-years needed to absorb this CO₂",
                     ha="center", fontsize=11, style="italic",
                     bbox=dict(boxstyle="round,pad=0.5",
                               facecolor="lightgreen", alpha=0.7))
        plt.suptitle("CO₂ Emissions Analysis", fontsize=16,
                     fontweight="bold", y=1.02)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def plot_gpr_evaluation(self, gpr_eval: Dict, save_path=None) -> plt.Figure:
        risks      = np.array(gpr_eval["risk_scores"])
        unc        = np.array(gpr_eval["uncertainties"])
        spread     = np.array(gpr_eval["spread_probs"])
        lev_counts = gpr_eval["level_counts"]

        level_colors = {
            "Low":      "#27ae60",
            "Moderate": "#f1c40f",
            "High":     "#e67e22",
            "Critical": "#e74c3c",
        }

        fig, axes = plt.subplots(2, 2, figsize=(18, 13))
        fig.suptitle(
            "Gaussian Process Regression — Test-Set Evaluation",
            fontsize=18, fontweight="bold", y=1.02,
        )

        ax = axes[0, 0]
        n_bins = min(50, max(10, len(risks) // 30))
        counts, bin_edges, patches = ax.hist(
            risks, bins=n_bins, edgecolor="black",
            linewidth=0.5, alpha=0.85, color="#3498db",
        )
        for patch, left in zip(patches, bin_edges[:-1]):
            mid = left + (bin_edges[1] - bin_edges[0]) / 2
            if   mid < 0.25: patch.set_facecolor("#27ae60")
            elif mid < 0.50: patch.set_facecolor("#f1c40f")
            elif mid < 0.75: patch.set_facecolor("#e67e22")
            else:            patch.set_facecolor("#e74c3c")
        ax.axvline(gpr_eval["mean_risk"], color="navy", lw=2.5,
                   linestyle="--", label=f"Mean={gpr_eval['mean_risk']:.3f}")
        ax.set_xlabel("GPR Risk Score"); ax.set_ylabel("Count")
        ax.set_title("Risk Score Distribution (Test Set)", fontweight="bold")
        legend_patches = [
            mpatches.Patch(color=c, label=l)
            for l, c in level_colors.items()
        ]
        ax.legend(handles=legend_patches + [
            plt.Line2D([0], [0], color="navy", lw=2.5, linestyle="--",
                       label=f"Mean = {gpr_eval['mean_risk']:.3f}")
        ], fontsize=9, loc="upper right")

        ax = axes[0, 1]
        ax.hist(unc, bins=n_bins, color="#9b59b6", edgecolor="black",
                linewidth=0.5, alpha=0.85)
        ax.axvline(gpr_eval["mean_uncertainty"], color="darkred", lw=2.5,
                   linestyle="--",
                   label=f"Mean σ = {gpr_eval['mean_uncertainty']:.4f}")
        ax.set_xlabel("GPR Posterior Std (σ)"); ax.set_ylabel("Count")
        ax.set_title("Prediction Uncertainty Distribution", fontweight="bold")
        ax.legend(fontsize=10)

        ax = axes[1, 0]
        ordered_levels = ["Low", "Moderate", "High", "Critical"]
        pie_labels = [l for l in ordered_levels if lev_counts.get(l, 0) > 0]
        pie_values = [lev_counts[l] for l in pie_labels]
        pie_colors = [level_colors[l] for l in pie_labels]
        wedges, texts, autotexts = ax.pie(
            pie_values, labels=pie_labels, colors=pie_colors,
            autopct=lambda p: f"{p:.1f}%\n({int(round(p/100*sum(pie_values)))})",
            startangle=140, pctdistance=0.78,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        for at in autotexts:
            at.set_fontsize(10); at.set_fontweight("bold")
        ax.set_title("Risk Level Breakdown (Test Set)", fontweight="bold", pad=14)

        ax = axes[1, 1]
        colours_scatter = []
        for r in risks:
            if   r < 0.25: colours_scatter.append("#27ae60")
            elif r < 0.50: colours_scatter.append("#f1c40f")
            elif r < 0.75: colours_scatter.append("#e67e22")
            else:          colours_scatter.append("#e74c3c")
        ax.scatter(risks, spread, c=colours_scatter,
                   alpha=0.45, s=18, edgecolors="none")
        z  = np.polyfit(risks, spread, 1)
        p  = np.poly1d(z)
        xs = np.linspace(risks.min(), risks.max(), 200)
        ax.plot(xs, p(xs), "k--", lw=2,
                label=f"y = {z[0]:.2f}x + {z[1]:.2f}")
        ax.set_xlabel("GPR Risk Score"); ax.set_ylabel("Spread Probability")
        ax.set_title("Spread Probability vs Risk Score", fontweight="bold")
        ax.legend(fontsize=10)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def generate_all_plots(self, metrics: Dict) -> Dict[str, str]:
        print("\n📊 Generating evaluation plots…")
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        paths = {
            "confusion_matrix": os.path.join(config.OUTPUT_DIR, "confusion_matrix.png"),
            "f1_scores":        os.path.join(config.OUTPUT_DIR, "f1_scores.png"),
            "roc_curves":       os.path.join(config.OUTPUT_DIR, "roc_curves.png"),
            "co2_emissions":    os.path.join(config.OUTPUT_DIR, "co2_emissions.png"),
            "gpr_evaluation":   os.path.join(config.OUTPUT_DIR, "gpr_evaluation.png"),
        }
        tasks = [
            ("confusion matrix", self.plot_confusion_matrix,
             [metrics["confusion_matrix"]], "confusion_matrix"),
            ("F1 scores",        self.plot_f1_scores,
             [metrics["classification_report"]], "f1_scores"),
            ("ROC curves",       self.plot_roc_curves,
             [metrics["y_true_binarized"], metrics["y_scores"]], "roc_curves"),
            ("CO₂ emissions",    self.plot_co2_emissions,
             [metrics["co2_emissions_kg"]], "co2_emissions"),
            ("GPR evaluation",   self.plot_gpr_evaluation,
             [metrics["gpr_eval"]], "gpr_evaluation"),
        ]
        for name, fn, args, key in tasks:
            print(f"   - {name}…")
            fig = fn(*args, save_path=paths[key])
            plt.close(fig)
        print("✅ All plots saved!")
        return paths

    def save_metrics_report(self, metrics: Dict, save_path: str = None) -> str:
        if save_path is None:
            save_path = os.path.join(config.OUTPUT_DIR, "evaluation_report.txt")
        gpr = metrics.get("gpr_eval", {})
        with open(save_path, "w") as f:
            f.write("="*80 + "\n")
            f.write("  AGENTIC-RAG PLANT DISEASE DIAGNOSIS — EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            f.write("DATASET CONFIGURATION\n" + "-"*40 + "\n")
            f.write(f"  Train  : {config.TRAIN_SAMPLES:,}\n")
            f.write(f"  Val    : {config.VAL_SAMPLES:,}\n")
            f.write(f"  Test   : {config.TEST_SAMPLES:,}\n")
            f.write(f"  Classes: {self.num_classes}\n\n")
            f.write("OVERALL METRICS\n" + "-"*40 + "\n")
            f.write(f"  Accuracy              : {metrics['accuracy']:.4f}\n")
            f.write(f"  F1-Score (Macro)      : {metrics['f1_score_macro']:.4f}\n")
            f.write(f"  F1-Score (Weighted)   : {metrics['f1_score_weighted']:.4f}\n")
            f.write(f"  ROC-AUC (Macro)       : {metrics['roc_auc_macro']:.4f}\n")
            f.write(f"  CO₂ Emissions         : {metrics['co2_emissions_kg']:.8f} kg\n\n")
            if gpr:
                f.write("GPR EVALUATION (Test Set)\n" + "-"*40 + "\n")
                f.write(f"  Mean Risk Score   : {gpr['mean_risk']:.4f}\n")
                f.write(f"  Mean Uncertainty  : {gpr['mean_uncertainty']:.4f}\n")
                f.write(f"  Mean Spread Prob. : {gpr['mean_spread']:.4f}\n")
                f.write(f"  Risk Distribution : {gpr['level_counts']}\n\n")
            f.write("PER-CLASS METRICS\n" + "-"*80 + "\n")
            f.write(f"{'Class':<50} {'Prec':>8} {'Rec':>8} "
                    f"{'F1':>8} {'Sup':>8}\n")
            f.write("-"*80 + "\n")
            report = metrics["classification_report"]
            for cls in self.label_names:
                if cls in report:
                    m = report[cls]
                    f.write(f"{cls:<50} {m['precision']:>8.4f} "
                            f"{m['recall']:>8.4f} {m['f1-score']:>8.4f} "
                            f"{m['support']:>8.0f}\n")
            f.write("\n" + "="*80 + "\n  END OF REPORT\n" + "="*80 + "\n")
        print(f"✅ Report → {save_path}")
        return save_path

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 15: RUN EVALUATION + SEND EVALUATION EMAIL
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  🧪 LOADING TEST SET AND RUNNING EVALUATION")
print("="*60 + "\n")

test_images = processor.load_split("test", max_samples=config.TEST_SAMPLES)
evaluator   = ModelEvaluator(orchestrator, processor, label_names)
metrics     = evaluator.evaluate_on_test_set(test_images)
plot_paths  = evaluator.generate_all_plots(metrics)
report_path = evaluator.save_metrics_report(metrics)

# ── Send evaluation report email automatically ────────────────────────────────
print(f"\n📧 Sending evaluation report to {config.NOTIFY_EMAIL}…")
eval_email_result = send_evaluation_email(metrics)
print(f"   Evaluation email status: "
      f"{eval_email_result.get('status','unknown').upper()}"
      f"  (method: {eval_email_result.get('method','-')})")

print("\n" + "="*60)
print("  ✅ EVALUATION COMPLETE!")
print("="*60)
print(f"\n📁 Outputs: {config.OUTPUT_DIR}")
for k, v in plot_paths.items():
    print(f"   {k}: {v}")
print(f"   Report: {report_path}\n")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 16: GRADIO APP
# ──────────────────────────────────────────────────────────────────────────────

import gradio as gr

_RISK_COLORS = {
    "Low":      "#27ae60",
    "Moderate": "#f1c40f",
    "High":     "#e67e22",
    "Critical": "#e74c3c",
}


class PlantDiagnosisApp:
    """
    Gradio UI — five tabs:
      🔍 Diagnosis      — upload image → disease / risk / crop / weather
                          (email with mitigation actions auto-sent on diagnosis)
      📊 Evaluation     — pre-computed metrics & four standard plots
      🧬 GPR Evaluation — four-panel GPR analysis over test set
      📧 Email Settings — configure SMTP / Resend credentials + manual send
      ℹ️  About
    """

    def __init__(self, orchestrator, metrics, plot_paths):
        self.orc        = orchestrator
        self.metrics    = metrics
        self.plot_paths = plot_paths

    # ── Primary callback ──────────────────────────────────────────────────────

    def diagnose(self, image, use_rag, lat, lon,
                 smtp_password, resend_key, resend_from,
                 smtp_sender, send_email_flag):
        if image is None:
            empty = {}
            return empty, empty, empty, "<p>No image uploaded.</p>", None, None, None

        # Inject credentials at runtime if provided via UI
        if smtp_password.strip():
            EMAIL_CONFIG["sender_password"] = smtp_password.strip()
        if resend_key.strip():
            EMAIL_CONFIG["resend_api_key"] = resend_key.strip()
        if resend_from.strip():
            EMAIL_CONFIG["resend_from"] = resend_from.strip()
        if smtp_sender.strip():
            EMAIL_CONFIG["sender_email"] = smtp_sender.strip()

        result = self.orc.diagnose(
            image, use_rag=use_rag, lat=lat, lon=lon,
            send_email=send_email_flag,
        )

        fd = result["final_diagnosis"]
        w  = result["weather_agent"]["result"]["weather"]
        ra = result["risk_assessment"]
        em = result.get("email_status", {})

        summary = {
            "🌿 Plant":      fd["plant_species"],
            "🦠 Disease":    fd["disease"],
            "🩺 Healthy":    "✅ Yes" if fd["is_healthy"] else "❌ No",
            "📊 Confidence": f"{fd['confidence']*100:.1f}%",
        }
        weather_card = {
            "📍 Location":    w["city"],
            "🌡️ Temperature": f"{w['temperature_c']:.1f}°C",
            "💧 Humidity":    f"{w['humidity_pct']}%",
            "☁️ Condition":   w["condition"].title(),
            "🌱 Best Crop":   result["weather_agent"]["result"]["best_crop"],
        }
        risk_card = {
            "⚠️ Risk Level":         ra["risk_level"],
            "📈 Risk Score":         f"{ra['risk_score']:.3f}",
            "📉 Uncertainty (σ)":    f"± {ra['uncertainty']:.3f}",
            "🔬 95% CI":             (f"[{ra['confidence_interval'][0]:.3f}, "
                                      f"{ra['confidence_interval'][1]:.3f}]"),
            "🌬️ Spread Probability": f"{ra['spread_probability']*100:.1f}%",
            "✉️ Email":              em.get("status", "not sent").upper(),
        }

        evidence_html = self._evidence_html(result)
        disease_fig   = self._disease_chart(result)
        crop_fig      = self._crop_chart(result)
        risk_fig      = self._risk_chart(ra)

        return (summary, weather_card, risk_card,
                evidence_html, disease_fig, crop_fig, risk_fig)

    # ── Manual evaluation-email send ──────────────────────────────────────────

    def send_eval_email_callback(self, smtp_password, resend_key,
                                  resend_from, smtp_sender):
        if smtp_password.strip():
            EMAIL_CONFIG["sender_password"] = smtp_password.strip()
        if resend_key.strip():
            EMAIL_CONFIG["resend_api_key"] = resend_key.strip()
        if resend_from.strip():
            EMAIL_CONFIG["resend_from"] = resend_from.strip()
        if smtp_sender.strip():
            EMAIL_CONFIG["sender_email"] = smtp_sender.strip()

        res    = send_evaluation_email(self.metrics)
        status = res.get("status", "unknown").upper()
        method = res.get("method", "")
        detail = f" via {method}" if method else ""
        if res.get("status") == "skipped":
            detail = f" — {res.get('reason', '')}"
        return (f"📧 Status: {status}{detail} → {config.NOTIFY_EMAIL}\n"
                f"Detail : {res.get('email_id', res.get('error', ''))}")

    # ── Plotting helpers ──────────────────────────────────────────────────────

    def _risk_chart(self, ra: Dict) -> plt.Figure:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        score        = ra["risk_score"]
        ci_lo, ci_hi = ra["confidence_interval"]
        segs = [(0.00,0.25,"#2ecc71"),(0.25,0.50,"#f1c40f"),
                (0.50,0.75,"#e67e22"),(0.75,1.00,"#e74c3c")]
        for lo, hi, col in segs:
            ax.barh(0, hi - lo, left=lo, height=0.4,
                    color=col, alpha=0.35, edgecolor="white", linewidth=2)
        ax.barh(0, ci_hi - ci_lo, left=ci_lo, height=0.18,
                color="navy", alpha=0.60,
                label=f"95% CI [{ci_lo:.2f} – {ci_hi:.2f}]")
        ax.scatter([score], [0], s=260, zorder=6, color="black",
                   marker="D", label=f"GPR mean = {score:.3f}")
        ax.set_xlim(0, 1); ax.set_ylim(-0.5, 0.8)
        ax.set_xlabel("Risk Score", fontsize=12, fontweight="bold")
        ax.set_title(
            f"GPR Risk Meter — {ra['risk_level']}",
            fontsize=13, fontweight="bold",
            color=_RISK_COLORS.get(ra["risk_level"], "#000"),
        )
        ax.set_yticks([])
        ax.legend(fontsize=9, loc="upper left")
        for txt, x in zip(["Low","Moderate","High","Critical"],
                           [0.125,0.375,0.625,0.875]):
            ax.text(x, 0.32, txt, ha="center", fontsize=9,
                    fontweight="bold", color="#333")
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.spines[["top","right"]].set_visible(False)

        ax2.axis("off")
        ax2.set_title("Mitigation Actions", fontsize=13, fontweight="bold")
        y = 0.95
        for i, action in enumerate(ra["mitigation_actions"]):
            icon    = "⚡" if i == 0 else "•"
            wrapped = "\n  ".join(action[j:j+44] for j in range(0,len(action),44))
            ax2.text(0.02, y - i * 0.17,
                     f"{icon}  {wrapped}",
                     transform=ax2.transAxes, fontsize=9.5, va="top",
                     color="#000",
                     fontweight="bold" if i == 0 else "normal")
        plt.tight_layout()
        return fig

    def _disease_chart(self, result: Dict) -> plt.Figure:
        diseases = result.get("rag_analysis", {}).get("top_diseases", [])
        fig, ax  = plt.subplots(figsize=(8, 4.5))
        if not diseases:
            ax.text(0.5, 0.5, "No disease data", ha="center",
                    va="center", fontsize=13); ax.axis("off"); return fig
        names  = [d["disease"] for d in diseases]
        confs  = [d["confidence"] * 100 for d in diseases]
        colors = ["#e74c3c","#e67e22","#3498db"][:len(names)]
        bars   = ax.barh(range(len(names)), confs, color=colors,
                         edgecolor="black", linewidth=1.5, alpha=0.9, height=0.5)
        for bar, c in zip(bars, confs):
            inside = c > 15
            ax.text(c-3 if inside else c+2,
                    bar.get_y()+bar.get_height()/2,
                    f"{c:.1f}%", ha="right" if inside else "left",
                    va="center", fontsize=11, fontweight="bold",
                    color="white" if inside else "#000")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=11)
        ax.set_xlim(0,100); ax.invert_yaxis()
        ax.set_xlabel("Confidence (%)", fontsize=12, fontweight="bold")
        ax.set_title("🎯 Disease Candidates", fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        return fig

    def _crop_chart(self, result: Dict) -> plt.Figure:
        crops = (result.get("weather_agent",{})
                       .get("result",{})
                       .get("top_recommended_crops",[]))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if not crops:
            ax.text(0.5,0.5,"No crop data",ha="center",
                    va="center",fontsize=13); ax.axis("off"); return fig
        names  = [c["plant"] for c in crops]
        scores = [c["suitability_score"] for c in crops]
        bars   = ax.bar(range(len(names)), scores, color="#27ae60",
                        alpha=0.9, edgecolor="black", linewidth=1.5, width=0.5)
        for bar, sc, nm in zip(bars, scores, names):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+1.5,
                    f"{sc:.0f}", ha="center", va="bottom",
                    fontsize=13, fontweight="bold")
            ax.text(bar.get_x()+bar.get_width()/2,
                    -3, nm, ha="center", va="top",
                    fontsize=12, fontweight="bold")
        ax.set_xticks([]); ax.set_ylim(0,115)
        ax.set_ylabel("Suitability", fontsize=12, fontweight="bold")
        ax.set_title("🌾 Crop Recommendations", fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        return fig

    def _evidence_html(self, result: Dict) -> str:
        ev_items     = result["disease_agent"]["evidence"]
        top_diseases = result.get("rag_analysis",{}).get("top_diseases",[])
        top_plants   = result.get("rag_analysis",{}).get("top_plants",[])
        ra           = result["risk_assessment"]
        risk_col     = _RISK_COLORS.get(ra["risk_level"], "#888")

        html = f"""
        <div style='font-family:Arial,sans-serif;background:#f5f5f5;
                    padding:20px;border-radius:10px;
                    border:2px solid #4CAF50;margin-top:10px;'>
          <h3 style='color:#000;margin-top:0;
                     border-bottom:2px solid #4CAF50;padding-bottom:10px;'>
            🔬 Evidence & Analysis
          </h3>
          <div style='display:flex;flex-wrap:wrap;gap:12px;margin-bottom:20px;'>
        """
        for ev in ev_items:
            bg = "#e3f2fd" if "label" in ev.lower() else (
                 "#fff3e0" if "similar" in ev.lower() else "#e8f5e9")
            bd = "#2196F3" if "label" in ev.lower() else (
                 "#FF9800" if "similar" in ev.lower() else "#4CAF50")
            html += (f"<div style='flex:1 1 calc(50%-12px);background:{bg};"
                     f"border-left:5px solid {bd};padding:10px;"
                     f"border-radius:5px;'>"
                     f"<span style='color:#000;font-weight:600;'>{ev}</span></div>")

        html += f"""
          </div>
          <div style='background:{risk_col};color:#fff;padding:12px 20px;
                      border-radius:8px;margin-bottom:20px;
                      display:flex;justify-content:space-between;align-items:center;'>
            <span style='font-size:16px;font-weight:bold;'>
              ⚠️ Risk Level: {ra['risk_level']}
            </span>
            <span style='font-size:14px;'>
              Score {ra['risk_score']:.3f} &nbsp;|&nbsp;
              ±{ra['uncertainty']:.3f} &nbsp;|&nbsp;
              Spread {ra['spread_probability']*100:.1f}%
            </span>
          </div>
          <div style='display:flex;gap:20px;'>
        """
        html += """
            <div style='flex:1;background:#fff;padding:18px;border-radius:8px;
                        border:1px solid #e74c3c;'>
              <h4 style='color:#000;margin-top:0;
                         border-bottom:2px solid #e74c3c;padding-bottom:8px;'>
                🦠 Top Disease Candidates
              </h4>
        """
        for d in (top_diseases or []):
            col = ("#27ae60" if d["confidence"] > 0.7 else
                   "#e67e22" if d["confidence"] > 0.4 else "#e74c3c")
            pct = d["confidence"] * 100
            html += (f"<div style='margin-bottom:12px;'>"
                     f"<div style='display:flex;justify-content:space-between;'>"
                     f"<span style='color:#000;font-weight:bold;'>{d['disease']}</span>"
                     f"<span style='color:{col};font-weight:bold;'>{pct:.1f}%</span></div>"
                     f"<div style='height:8px;background:#ecf0f1;border-radius:4px;"
                     f"overflow:hidden;margin-top:4px;'>"
                     f"<div style='width:{pct}%;height:8px;background:{col};"
                     f"border-radius:4px;'></div></div></div>")
        if not top_diseases:
            html += "<p style='color:#000;'>No candidates found.</p>"
        html += "</div>"

        html += """
            <div style='flex:1;background:#fff;padding:18px;border-radius:8px;
                        border:1px solid #27ae60;'>
              <h4 style='color:#000;margin-top:0;
                         border-bottom:2px solid #27ae60;padding-bottom:8px;'>
                🌿 Top Plant Matches
              </h4>
        """
        for p in (top_plants or []):
            html += (f"<div style='display:flex;justify-content:space-between;"
                     f"margin-bottom:10px;padding:8px;background:#f8f9fa;"
                     f"border-radius:6px;'>"
                     f"<span style='color:#000;font-weight:bold;'>{p['plant']}</span>"
                     f"<span style='background:#27ae60;color:white;padding:3px 10px;"
                     f"border-radius:20px;font-weight:bold;'>"
                     f"{p['confidence']*100:.1f}%</span></div>")
        if not top_plants:
            html += "<p style='color:#000;'>No matches found.</p>"
        html += "</div></div></div>"
        return html

    def get_metrics_summary(self) -> Dict:
        gpr = self.metrics.get("gpr_eval", {})
        d = {
            "Accuracy":            f"{self.metrics['accuracy']:.4f}",
            "F1-Score (Macro)":    f"{self.metrics['f1_score_macro']:.4f}",
            "F1-Score (Weighted)": f"{self.metrics['f1_score_weighted']:.4f}",
            "ROC-AUC (Macro)":     f"{self.metrics['roc_auc_macro']:.4f}",
            "CO₂ Emissions":       f"{self.metrics['co2_emissions_kg']:.8f} kg",
            "Test Samples":        f"{config.TEST_SAMPLES:,}",
        }
        if gpr:
            d.update({
                "GPR Mean Risk":        f"{gpr['mean_risk']:.4f}",
                "GPR Mean Uncertainty": f"{gpr['mean_uncertainty']:.4f}",
                "GPR Mean Spread":      f"{gpr['mean_spread']:.4f}",
            })
        return d

    # ── Launch ────────────────────────────────────────────────────────────────

    def launch(self):
        custom_css = """
        .json-container {
            color:#000!important;background:#fff!important;
            border:2px solid #4CAF50!important;
            border-radius:8px!important;padding:15px!important;
        }
        """
        with gr.Blocks(title="🌿 Plant Disease Diagnosis",
                       theme=gr.themes.Soft(), css=custom_css) as demo:

            gr.Markdown(
                "# 🌿 Agentic-RAG Plant Disease Diagnosis System  ·  v3\n"
                "**Agents**: Disease Prediction (CLIP+FAISS) · "
                "Weather & Crops · GPR Risk Mitigation · "
                "**Email**: SMTP (Gmail/Brevo/Zoho)  or  Resend HTTP API"
            )

            with gr.Tabs():

                # ── Tab 1: Diagnosis ───────────────────────────────────────
                with gr.TabItem("🔍 Diagnosis"):
                    gr.Markdown(
                        "> 📧 **Tip:** tick *Send email* below to automatically "
                        "receive the full diagnosis + mitigation report in your inbox "
                        "as soon as the result is ready."
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            image_input = gr.Image(
                                type="pil", label="📷 Upload Plant Image"
                            )
                            use_rag = gr.Checkbox(
                                value=True,
                                label="Use RAG for enhanced diagnosis",
                            )
                            send_email_flag = gr.Checkbox(
                                value=False,
                                label=f"📧 Email diagnosis + mitigation → {config.NOTIFY_EMAIL}",
                            )
                            with gr.Accordion("📍 Location", open=False):
                                lat_input = gr.Number(
                                    value=config.DEFAULT_LAT, label="Latitude"
                                )
                                lon_input = gr.Number(
                                    value=config.DEFAULT_LON, label="Longitude"
                                )
                            with gr.Accordion("🔑 Email Credentials", open=False):
                                gr.Markdown(
                                    "**Method A — SMTP (Gmail App Password)**\n\n"
                                    "Generate at: https://myaccount.google.com/apppasswords\n\n"
                                    "**Method B — Resend HTTP API**\n\n"
                                    "Free 3 000 emails/month at https://resend.com/api-keys\n\n"
                                    "_The dispatcher tries SMTP first, then Resend._"
                                )
                                smtp_sender_input = gr.Textbox(
                                    value=EMAIL_CONFIG.get("sender_email",""),
                                    placeholder="you@gmail.com",
                                    label="SMTP From Address",
                                )
                                smtp_pw_input = gr.Textbox(
                                    value="",
                                    placeholder="16-char Gmail App Password",
                                    label="SMTP Password / App Password",
                                    type="password",
                                )
                                resend_key_input = gr.Textbox(
                                    value="",
                                    placeholder="re_…",
                                    label="Resend API Key (fallback)",
                                    type="password",
                                )
                                resend_from_input = gr.Textbox(
                                    value=EMAIL_CONFIG.get("resend_from",""),
                                    placeholder="Plant-AI <onboarding@resend.dev>",
                                    label="Resend From Address",
                                )
                            diagnose_btn = gr.Button(
                                "🔍 Diagnose Plant Disease",
                                variant="primary", size="lg",
                            )

                        with gr.Column(scale=2):
                            with gr.Row():
                                summary_out = gr.JSON(
                                    label="📋 Diagnosis Summary",
                                    elem_classes=["json-container"],
                                )
                                weather_out = gr.JSON(
                                    label="🌤️ Weather",
                                    elem_classes=["json-container"],
                                )
                                risk_out = gr.JSON(
                                    label="⚠️ GPR Risk Assessment",
                                    elem_classes=["json-container"],
                                )
                            evidence_out = gr.HTML(label="🔬 Evidence & Analysis")
                            gr.Markdown("---\n### 📊 Analysis Charts")
                            with gr.Row():
                                disease_chart = gr.Plot(label="🎯 Disease Candidates")
                                crop_chart    = gr.Plot(label="🌾 Crop Recommendations")
                            with gr.Row():
                                risk_chart = gr.Plot(label="📉 GPR Risk Meter & Mitigation")

                # ── Tab 2: Standard Evaluation ─────────────────────────────
                with gr.TabItem("📊 Model Evaluation"):
                    gr.Markdown("### 📈 Model Performance Metrics")
                    gr.JSON(
                        value=self.get_metrics_summary(),
                        label="Overall Metrics",
                        elem_classes=["json-container"],
                    )
                    with gr.Row():
                        gr.Image(
                            value=self.plot_paths["confusion_matrix"],
                            label="Confusion Matrix", interactive=False,
                        )
                        gr.Image(
                            value=self.plot_paths["f1_scores"],
                            label="F1 Scores per Class", interactive=False,
                        )
                    with gr.Row():
                        gr.Image(
                            value=self.plot_paths["roc_curves"],
                            label="ROC Curves", interactive=False,
                        )
                        gr.Image(
                            value=self.plot_paths["co2_emissions"],
                            label="CO₂ Emissions", interactive=False,
                        )
                    gr.File(
                        label="📥 Download Evaluation Report",
                        value=report_path, interactive=False,
                    )

                # ── Tab 3: GPR Evaluation ──────────────────────────────────
                with gr.TabItem("🧬 GPR Evaluation"):
                    gr.Markdown(
                        "### Gaussian Process Regression — Test-Set Analysis\n\n"
                        "Four panels: risk score distribution, uncertainty, "
                        "risk level breakdown, and spread-probability correlation."
                    )
                    gpr_ev = self.metrics.get("gpr_eval", {})
                    if gpr_ev:
                        gr.JSON(
                            value={
                                "Mean Risk Score":   f"{gpr_ev['mean_risk']:.4f}",
                                "Mean Uncertainty":  f"{gpr_ev['mean_uncertainty']:.4f}",
                                "Mean Spread Prob.": f"{gpr_ev['mean_spread']:.4f}",
                                "Risk Level Counts": gpr_ev["level_counts"],
                            },
                            label="GPR Summary Statistics",
                            elem_classes=["json-container"],
                        )
                    gr.Image(
                        value=self.plot_paths.get("gpr_evaluation", ""),
                        label="GPR Evaluation — Four-Panel Plot",
                        interactive=False,
                    )

                # ── Tab 4: Email Settings ──────────────────────────────────
                with gr.TabItem("📧 Email Settings"):
                    gr.Markdown(
                        f"### Email Configuration & Manual Send\n\n"
                        f"**Recipient:** `{config.NOTIFY_EMAIL}`\n\n"
                        "| | Method A — SMTP | Method B — Resend HTTP |\n"
                        "|---|---|---|\n"
                        "| Cost | Free (uses your Gmail/Brevo/Zoho) | Free tier: 3 000/month |\n"
                        "| Setup | App Password at [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords) | API key at [resend.com/api-keys](https://resend.com/api-keys) |\n"
                        "| SDK needed | None — pure `smtplib` | None — plain `requests` |\n\n"
                        "> The dispatcher tries **SMTP first**, then falls back to **Resend**."
                        " Both send the same full report with mitigation actions."
                    )
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Method A — SMTP")
                            smtp_sender_email = gr.Textbox(
                                value=EMAIL_CONFIG.get("sender_email",""),
                                placeholder="you@gmail.com",
                                label="Sender Email",
                            )
                            smtp_pw_email = gr.Textbox(
                                value="",
                                placeholder="16-char Gmail App Password",
                                label="App Password / SMTP Password",
                                type="password",
                            )
                        with gr.Column():
                            gr.Markdown("#### Method B — Resend HTTP")
                            resend_key_email = gr.Textbox(
                                value="",
                                placeholder="re_…",
                                label="Resend API Key",
                                type="password",
                            )
                            resend_from_email = gr.Textbox(
                                value=EMAIL_CONFIG.get("resend_from",""),
                                placeholder="Plant-AI <onboarding@resend.dev>",
                                label="Resend From Address",
                            )
                    email_btn    = gr.Button(
                        "📧 Send Evaluation Report Now",
                        variant="primary",
                    )
                    email_status = gr.Textbox(
                        label="Delivery Status", interactive=False
                    )
                    email_btn.click(
                        fn=self.send_eval_email_callback,
                        inputs=[smtp_pw_email, resend_key_email,
                                resend_from_email, smtp_sender_email],
                        outputs=[email_status],
                    )

                # ── Tab 5: About ───────────────────────────────────────────
                with gr.TabItem("ℹ️ About"):
                    gr.Markdown(f"""
## 🌿 Agentic-RAG Plant Disease Diagnosis System  ·  v3

### Architecture
| Agent | Technology | Purpose |
|-------|-----------|---------|
| DiseasePredictionAgent | CLIP ViT-B/32 + FAISS | Image embedding & label voting via RAG |
| WeatherCropAgent | OpenWeatherMap API | Live weather + seasonal crop scoring |
| GPRiskMitigationAgent | Gaussian Process Regression (RBF+White) | Risk score with 95% CI, spread prob., mitigation |

### Email Integration  (no Anthropic SDK required)
| | Method A — SMTP | Method B — Resend HTTP |
|---|---|---|
| Library | `smtplib` (Python stdlib) | `requests` |
| Auth | Gmail App Password | `re_…` API key |
| Free? | ✅ Yes | ✅ Yes (3 000/month) |
| When sent | Automatically on image upload (if ticked) + post-evaluation | Same |

### Email Contents — Per-Image Upload
The diagnosis email includes:
- **Diagnosis** (plant, disease, confidence, RAG candidates)
- **GPR Risk Assessment** (risk score, 95% CI, uncertainty, spread probability)
- **Mitigation Actions** (tailored to risk level)
- **Current Weather** (temperature, humidity, soil moisture, season)
- **Top Crop Recommendations** (suitability scores)

### Performance
| Metric | Score |
|--------|-------|
| Accuracy | `{self.metrics['accuracy']:.4f}` |
| F1 (Macro) | `{self.metrics['f1_score_macro']:.4f}` |
| F1 (Weighted) | `{self.metrics['f1_score_weighted']:.4f}` |
| ROC-AUC | `{self.metrics['roc_auc_macro']:.4f}` |
| CO₂ | `{self.metrics['co2_emissions_kg']:.8f}` kg |

### Stack
CLIP ViT-B/32 · FAISS · Gaussian Process Regression · CodeCarbon ·
smtplib · Resend HTTP API · Gradio
                    """)

            # ── Wire diagnosis button ──────────────────────────────────────
            diagnose_btn.click(
                fn=self.diagnose,
                inputs=[
                    image_input, use_rag, lat_input, lon_input,
                    smtp_pw_input, resend_key_input,
                    resend_from_input, smtp_sender_input,
                    send_email_flag,
                ],
                outputs=[
                    summary_out, weather_out, risk_out,
                    evidence_out, disease_chart, crop_chart, risk_chart,
                ],
            )

            gr.Markdown(
                "---\n"
                "**Dataset**: PlantVillage · "
                "**Embeddings**: CLIP ViT-B/32 · "
                "**VectorDB**: FAISS · "
                "**Risk**: Gaussian Process Regression · "
                "**Email**: smtplib + Resend HTTP (no Anthropic SDK)"
            )

        demo.launch(share=True, debug=False)


app = PlantDiagnosisApp(orchestrator, metrics, plot_paths)
print("\n🚀 Launching Gradio interface…")
app.launch()
