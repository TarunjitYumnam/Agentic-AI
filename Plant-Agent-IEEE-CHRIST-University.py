# SECTION 1: INSTALL DEPENDENCIES
# ──────────────────────────────────────────────────────────────

print("📦 Installing dependencies...")
import subprocess
import sys

packages = [
    "datasets",
    "transformers",
    "sentence-transformers",
    "faiss-cpu",
    "gradio",
    "torch",
    "torchvision",
    "Pillow",
    "tqdm",
    "requests",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "codecarbon",
    "langchain",
    "langchain-community",
    "numpy",
    "pandas"
]

for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

print("✅ All dependencies installed!\n")
# SECTION 2: IMPORTS & SETUP
# ──────────────────────────────────────────────────────────────

import os
import json
import time
import random
import base64
import hashlib
import requests
import numpy as np
import pandas as pd
import torch
from PIL import Image
from io import BytesIO
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from tqdm import tqdm
from collections import Counter
import logging
import warnings
warnings.filterwarnings("ignore")

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Metrics
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score,
    roc_curve, auc, roc_auc_score, accuracy_score,
    precision_recall_fscore_support
)
from sklearn.preprocessing import label_binarize

# CO2 Tracking
from codecarbon import EmissionsTracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Device: {device}")

# ──────────────────────────────────────────────────────────────
# SECTION 3: CONFIGURATION
# ──────────────────────────────────────────────────────────────

class Config:
    """Central configuration with updated sample sizes"""

    # ── OpenWeatherMap API ──────────────────────────────────
    OPENWEATHER_API_KEY: str = ""  # Optional: add your key

    # ── Dataset settings ────────────────────────────────────
    DATASET_NAME: str = "GVJahnavi/Plant_village_subset"
    TRAIN_SAMPLES: int = 9057      # Updated
    VAL_SAMPLES: int = 2265        # Updated
    TEST_SAMPLES: int = 2916       # Updated

    # ── Model settings ──────────────────────────────────────
    CLIP_MODEL: str = "clip-ViT-B-32"
    EMBEDDING_DIM: int = 512
    TOP_K_SIMILAR: int = 5

    # ── Default location ────────────────────────────────────
    DEFAULT_LAT: float = 28.6139   # New Delhi
    DEFAULT_LON: float = 77.2090
    DEFAULT_CITY: str = "New Delhi"

    # ── Output paths ────────────────────────────────────────
    OUTPUT_DIR: str = "/content/outputs"
    EMISSIONS_FILE: str = "/content/outputs/emissions.csv"

config = Config()

# Create output directory
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
# SECTION 4: LOAD DATASET
# ──────────────────────────────────────────────────────────────

from datasets import load_dataset

print("📥 Loading PlantVillage dataset...")

try:
    dataset = load_dataset(config.DATASET_NAME)
    label_names: List[str] = dataset["train"].features["label"].names
    num_classes = len(label_names)

    # Create train/val/test splits
    if "validation" not in dataset or "test" not in dataset:
        # Split the train set into train/val/test
        full_train = dataset["train"]
        total_samples = len(full_train)

        # Calculate split ratios
        train_ratio = config.TRAIN_SAMPLES / total_samples
        val_ratio = config.VAL_SAMPLES / total_samples
        test_ratio = config.TEST_SAMPLES / total_samples

        # First split: separate test set
        split1 = full_train.train_test_split(test_size=test_ratio, seed=42)
        test_set = split1["test"]

        # Second split: separate train and validation
        val_size = val_ratio / (train_ratio + val_ratio)
        split2 = split1["train"].train_test_split(test_size=val_size, seed=42)

        dataset["train"] = split2["train"]
        dataset["validation"] = split2["test"]
        dataset["test"] = test_set

    print(f"✅ Dataset loaded successfully!")
    print(f"   Classes: {num_classes}")
    print(f"   Train samples: {len(dataset['train'])}")
    print(f"   Validation samples: {len(dataset['validation'])}")
    print(f"   Test samples: {len(dataset['test'])}")
    print(f"   Label names (first 5): {label_names[:5]}...")

except Exception as exc:
    raise RuntimeError(f"❌ Dataset loading failed: {exc}")

# ──────────────────────────────────────────────────────────────
# SECTION 5: DATA STRUCTURES
# ──────────────────────────────────────────────────────────────

@dataclass
class PlantImageData:
    image_id: str
    image_pil: Image.Image
    label: int
    label_name: str
    disease_type: str
    plant_type: str
    is_healthy: bool = False
    embedding: Optional[np.ndarray] = None

@dataclass
class DiagnosisResult:
    disease: str
    plant_species: str
    confidence: float
    is_healthy: bool
    predicted_label: int = -1
    evidence: List[str] = field(default_factory=list)
    agent_name: str = ""

@dataclass
class WeatherData:
    city: str
    temperature_c: float
    humidity_pct: float
    condition: str
    wind_kmh: float
    soil_moisture_estimate: str
    season: str

# ──────────────────────────────────────────────────────────────
# SECTION 6: DATASET PROCESSOR
# ──────────────────────────────────────────────────────────────

class PlantVillageProcessor:
    """Parses PlantVillage label schema: <Plant>___<Disease_words>"""

    def __init__(self, dataset, label_names: List[str]):
        self.dataset = dataset
        self.label_names = label_names
        self.mapping = self._build_mapping()
        self.all_plants: List[str] = sorted({v["plant"] for v in self.mapping.values()})
        print(f"🌿 Plants in dataset ({len(self.all_plants)}): {self.all_plants}")

    def _build_mapping(self) -> Dict[str, Dict]:
        mapping: Dict[str, Dict] = {}
        for label in self.label_names:
            if "___" in label:
                plant_part, disease_part = label.split("___", 1)
            else:
                parts = label.split("_")
                plant_part = parts[0]
                disease_part = "_".join(parts[1:])

            disease_readable = disease_part.replace("_", " ").strip()
            mapping[label] = {
                "plant": plant_part,
                "disease": disease_readable if disease_readable else "Unknown",
                "healthy": "healthy" in disease_readable.lower(),
                "raw_label": label,
            }
        return mapping

    def load_split(self, split: str = "train", max_samples: int = None) -> List[PlantImageData]:
        items: List[PlantImageData] = []
        ds = self.dataset[split]
        n = min(max_samples, len(ds)) if max_samples else len(ds)

        for i in tqdm(range(n), desc=f"Loading {split}"):
            sample = ds[i]
            img: Image.Image = sample["image"]
            label: int = sample["label"]
            label_name: str = self.label_names[label]
            meta = self.mapping[label_name]

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
# SECTION 7: CLIP EMBEDDING GENERATOR
# ──────────────────────────────────────────────────────────────

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

# ──────────────────────────────────────────────────────────────
# SECTION 8: FAISS VECTOR DATABASE
# ──────────────────────────────────────────────────────────────

import faiss

class PlantVillageVectorDB:
    """Vector database with FAISS for fast similarity search"""

    def __init__(self, dim: int = config.EMBEDDING_DIM):
        self.dim = dim
        self.data: List[PlantImageData] = []
        self.index = faiss.IndexFlatIP(dim)

    def add(self, image_data: List[PlantImageData], embedder: CLIPEmbeddingGenerator):
        batch_embs: List[np.ndarray] = []

        for item in tqdm(image_data, desc="Embedding & indexing"):
            emb = embedder.encode_image(item.image_pil)
            item.embedding = emb
            batch_embs.append(emb)
            self.data.append(item)

        embs = np.array(batch_embs, dtype=np.float32)
        faiss.normalize_L2(embs)
        self.index.add(embs)

        print(f"✅ Indexed {len(batch_embs)} images (total: {len(self.data)})")

    def search(self, query_emb: np.ndarray, k: int = 5) -> List[Tuple[PlantImageData, float]]:
        q = query_emb.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q)
        sims, idxs = self.index.search(q, k)
        return [(self.data[i], float(sims[0][j])) for j, i in enumerate(idxs[0]) if i < len(self.data)]

vector_db = PlantVillageVectorDB(dim=clip_generator.dim)

# ──────────────────────────────────────────────────────────────
# SECTION 9: BUILD VECTOR DB INDEX
# ──────────────────────────────────────────────────────────────

print("\n📚 Building vector database from training data...")
train_images = processor.load_split("train", max_samples=config.TRAIN_SAMPLES)
vector_db.add(train_images, clip_generator)
# SECTION 10: RAG CORE SYSTEM
# ──────────────────────────────────────────────────────────────

class PlantDiseaseRAG:
    """Retrieval-Augmented Generation core"""

    def __init__(self, vector_db: PlantVillageVectorDB, clip_generator: CLIPEmbeddingGenerator):
        self.vector_db = vector_db
        self.clip_generator = clip_generator

    def find_similar_cases(self, image_pil: Image.Image, k: int = config.TOP_K_SIMILAR):
        query_emb = self.clip_generator.encode_image(image_pil)
        return self.vector_db.search(query_emb, k)

    def analyze_similar_cases(self, cases: List[Tuple[PlantImageData, float]]) -> Dict[str, Any]:
        if not cases:
            return {
                "top_diseases": [], "top_plants": [], "label_votes": {},
                "total_cases": 0, "avg_similarity": 0.0, "is_healthy": False
            }

        disease_scores: Dict[str, float] = {}
        plant_scores: Dict[str, float] = {}
        label_votes: Dict[str, float] = {}
        health_score = 0.0
        total_sim = sum(sim for _, sim in cases)

        for item, sim in cases:
            weight = sim / (total_sim + 1e-8)
            disease_scores[item.disease_type] = disease_scores.get(item.disease_type, 0) + sim
            plant_scores[item.plant_type] = plant_scores.get(item.plant_type, 0) + sim
            label_votes[item.label_name] = label_votes.get(item.label_name, 0) + weight
            if item.is_healthy:
                health_score += weight

        top_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        top_plants = sorted(plant_scores.items(), key=lambda x: x[1], reverse=True)[:2]

        return {
            "top_diseases": [{"disease": d, "score": s, "confidence": s / total_sim} for d, s in top_diseases],
            "top_plants": [{"plant": p, "score": s, "confidence": s / total_sim} for p, s in top_plants],
            "label_votes": label_votes,
            "total_cases": len(cases),
            "avg_similarity": total_sim / len(cases),
            "is_healthy": health_score > 0.5,
        }

    def create_rag_context(self, analysis: Dict[str, Any]) -> str:
        if analysis["total_cases"] == 0:
            return "No similar cases found."

        lines = [
            f"[RAG] Retrieved {analysis['total_cases']} similar cases "
            f"(avg similarity: {analysis['avg_similarity']:.3f})\n",
            "=== Top Diseases ===",
        ]
        for d in analysis["top_diseases"]:
            lines.append(f"  • {d['disease']} ({d['confidence']*100:.1f}%)")

        lines.append("\n=== Top Plants ===")
        for p in analysis["top_plants"]:
            lines.append(f"  • {p['plant']} ({p['confidence']*100:.1f}%)")

        return "\n".join(lines)

rag_system = PlantDiseaseRAG(vector_db, clip_generator)

# ──────────────────────────────────────────────────────────────
# SECTION 11: AGENT 1 - DISEASE PREDICTION
# ──────────────────────────────────────────────────────────────

class DiseasePredictionAgent:
    """Disease prediction using CLIP + label voting"""

    AGENT_NAME = "DiseasePredictionAgent"

    def __init__(self, rag_system: PlantDiseaseRAG, processor: PlantVillageProcessor):
        self.rag = rag_system
        self.processor = processor

    def run(self, image_pil: Image.Image, rag_analysis: Optional[Dict] = None) -> DiagnosisResult:
        similar_cases = self.rag.find_similar_cases(image_pil, k=config.TOP_K_SIMILAR)

        if rag_analysis is None:
            rag_analysis = self.rag.analyze_similar_cases(similar_cases)

        label_votes = rag_analysis.get("label_votes", {})
        if not label_votes:
            return DiagnosisResult(
                disease="Unknown", plant_species="Unknown",
                confidence=0.0, is_healthy=False,
                predicted_label=-1,
                agent_name=self.AGENT_NAME,
                evidence=["No similar cases found."],
            )

        # Get winner label
        best_label, best_weight = max(label_votes.items(), key=lambda x: x[1])
        meta = self.processor.mapping.get(best_label, {})
        disease = meta.get("disease", best_label)
        plant = meta.get("plant", "Unknown")
        is_healthy = meta.get("healthy", False)

        # Get predicted label index
        predicted_label = self.processor.label_names.index(best_label) if best_label in self.processor.label_names else -1

        # Calculate confidence
        top_sim = similar_cases[0][1] if similar_cases else 0.0
        avg_sim = rag_analysis.get("avg_similarity", 0.0)
        confidence = float(np.clip(0.4 * top_sim + 0.4 * avg_sim + 0.2 * best_weight, 0.0, 1.0))

        evidence = [
            f"Top label: '{best_label}' (weight: {best_weight:.3f})",
            f"Top similarity: {top_sim:.3f}",
        ]

        return DiagnosisResult(
            disease=disease,
            plant_species=plant,
            confidence=confidence,
            is_healthy=is_healthy,
            predicted_label=predicted_label,
            evidence=evidence,
            agent_name=self.AGENT_NAME,
        )
      # SECTION 12: AGENT 2 - WEATHER & CROP RECOMMENDATION
# ──────────────────────────────────────────────────────────────

class WeatherCropAgent:
    """Live weather + crop recommendation agent"""

    AGENT_NAME = "WeatherCropAgent"

    PLANT_PROFILES: Dict[str, Dict] = {
        "apple":      {"temp_range": (15, 25), "humidity": (50, 80), "seasons": ["spring", "autumn"]},
        "blueberry":  {"temp_range": (18, 28), "humidity": (60, 80), "seasons": ["spring", "summer"]},
        "cherry":     {"temp_range": (15, 25), "humidity": (50, 75), "seasons": ["spring", "summer"]},
        "corn":       {"temp_range": (20, 32), "humidity": (55, 75), "seasons": ["summer"]},
        "grape":      {"temp_range": (20, 30), "humidity": (40, 70), "seasons": ["summer", "autumn"]},
        "orange":     {"temp_range": (22, 35), "humidity": (55, 80), "seasons": ["autumn", "winter"]},
        "peach":      {"temp_range": (18, 28), "humidity": (50, 70), "seasons": ["spring", "summer"]},
        "pepper":     {"temp_range": (22, 32), "humidity": (50, 75), "seasons": ["summer"]},
        "potato":     {"temp_range": (15, 25), "humidity": (60, 80), "seasons": ["spring", "autumn"]},
        "raspberry":  {"temp_range": (15, 25), "humidity": (60, 80), "seasons": ["spring", "summer"]},
        "soybean":    {"temp_range": (20, 30), "humidity": (55, 75), "seasons": ["summer"]},
        "squash":     {"temp_range": (20, 35), "humidity": (50, 70), "seasons": ["summer"]},
        "strawberry": {"temp_range": (15, 25), "humidity": (60, 80), "seasons": ["spring"]},
        "tomato":     {"temp_range": (20, 30), "humidity": (50, 70), "seasons": ["summer", "autumn"]},
    }

    SEASON_MAP = {
        (12, 1, 2):  "winter",
        (3, 4, 5):   "spring",
        (6, 7, 8):   "summer",
        (9, 10, 11): "autumn",
    }

    def __init__(self, api_key: str = "", all_plants: Optional[List[str]] = None):
        self.api_key = api_key or config.OPENWEATHER_API_KEY
        self.all_plants = [p.lower() for p in (all_plants or processor.all_plants)]
        self._cache: Dict[str, WeatherData] = {}

    def _get_season(self, month: int) -> str:
        for months, season in self.SEASON_MAP.items():
            if month in months:
                return season
        return "unknown"

    def _estimate_soil_moisture(self, humidity: float, condition: str) -> str:
        condition_l = condition.lower()
        if any(w in condition_l for w in ["rain", "drizzle", "storm"]):
            return "High (wet)"
        if humidity > 75:
            return "Moderate-High"
        if humidity > 50:
            return "Moderate"
        return "Low (dry)"

    def fetch_weather(self, lat: float = config.DEFAULT_LAT, lon: float = config.DEFAULT_LON) -> WeatherData:
        cache_key = f"{lat:.2f},{lon:.2f}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self.api_key:
            return self._simulated_weather()

        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.api_key}&units=metric"
            resp = requests.get(url, timeout=8)
            resp.raise_for_status()
            data = resp.json()

            import datetime
            month = datetime.datetime.now().month
            season = self._get_season(month)

            weather = WeatherData(
                city=data.get("name", config.DEFAULT_CITY),
                temperature_c=data["main"]["temp"],
                humidity_pct=data["main"]["humidity"],
                condition=data["weather"][0]["description"],
                wind_kmh=data["wind"]["speed"] * 3.6,
                soil_moisture_estimate=self._estimate_soil_moisture(data["main"]["humidity"], data["weather"][0]["description"]),
                season=season,
            )
            self._cache[cache_key] = weather
            return weather

        except Exception as exc:
            logger.warning(f"Weather API error: {exc}")
            return self._simulated_weather()

    def _simulated_weather(self) -> WeatherData:
        import datetime
        month = datetime.datetime.now().month
        season = self._get_season(month)
        seasonal_temp = {"spring": 18, "summer": 28, "autumn": 20, "winter": 8}.get(season, 22)
        seasonal_hum = {"spring": 65, "summer": 60, "autumn": 70, "winter": 75}.get(season, 65)

        return WeatherData(
            city=config.DEFAULT_CITY,
            temperature_c=float(seasonal_temp),
            humidity_pct=float(seasonal_hum),
            condition="partly cloudy (simulated)",
            wind_kmh=12.0,
            soil_moisture_estimate=self._estimate_soil_moisture(float(seasonal_hum), ""),
            season=season,
        )

    def recommend_crops(self, weather: WeatherData) -> Dict[str, Any]:
        scored: List[Tuple[str, float, List[str]]] = []

        for plant in self.all_plants:
            profile = self.PLANT_PROFILES.get(plant)
            if profile is None:
                continue

            score = 0.0
            reasons: List[str] = []

            # Temperature
            t_min, t_max = profile["temp_range"]
            if t_min <= weather.temperature_c <= t_max:
                score += 40
                reasons.append(f"Temp {weather.temperature_c:.1f}°C ideal")
            else:
                delta = min(abs(weather.temperature_c - t_min), abs(weather.temperature_c - t_max))
                partial = max(0, 40 - delta * 3)
                score += partial

            # Humidity
            h_min, h_max = profile["humidity"]
            if h_min <= weather.humidity_pct <= h_max:
                score += 35
                reasons.append(f"Humidity {weather.humidity_pct:.0f}% ideal")
            else:
                delta = min(abs(weather.humidity_pct - h_min), abs(weather.humidity_pct - h_max))
                partial = max(0, 35 - delta * 2)
                score += partial

            # Season
            if weather.season in profile["seasons"]:
                score += 25
                reasons.append(f"Season '{weather.season}' ideal")

            scored.append((plant.capitalize(), score, reasons))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_crops = [
            {"plant": p, "suitability_score": round(s, 1), "max_score": 100, "reasons": r}
            for p, s, r in scored[:5]
        ]

        return {
            "weather": {
                "city": weather.city,
                "temperature_c": weather.temperature_c,
                "humidity_pct": weather.humidity_pct,
                "condition": weather.condition,
                "wind_kmh": weather.wind_kmh,
                "soil_moisture": weather.soil_moisture_estimate,
                "season": weather.season,
            },
            "top_recommended_crops": top_crops,
            "best_crop": top_crops[0]["plant"] if top_crops else "Unknown",
        }

    def run(self, lat: float = config.DEFAULT_LAT, lon: float = config.DEFAULT_LON) -> Dict[str, Any]:
        weather = self.fetch_weather(lat, lon)
        return self.recommend_crops(weather)

disease_agent = DiseasePredictionAgent(rag_system, processor)
weather_agent = WeatherCropAgent(all_plants=processor.all_plants)

# SECTION 13: AGENTIC ORCHESTRATOR
# ──────────────────────────────────────────────────────────────

class AgenticOrchestrator:
    """Coordinates multiple agents"""

    def __init__(self, rag_system, disease_agent, weather_agent):
        self.rag = rag_system
        self.disease_agent = disease_agent
        self.weather_agent = weather_agent

    def diagnose(self, image_pil: Image.Image, use_rag: bool = True,
                 lat: float = config.DEFAULT_LAT, lon: float = config.DEFAULT_LON) -> Dict[str, Any]:

        # RAG retrieval
        rag_analysis: Dict = {}
        if use_rag:
            _, rag_analysis = self._step_rag(image_pil)

        # Disease prediction
        diagnosis = self.disease_agent.run(image_pil, rag_analysis)

        # Weather & crop
        weather_crop = self.weather_agent.run(lat, lon)

        return {
            "final_diagnosis": {
                "disease": diagnosis.disease,
                "plant_species": diagnosis.plant_species,
                "is_healthy": diagnosis.is_healthy,
                "confidence": diagnosis.confidence,
                "predicted_label": diagnosis.predicted_label,
            },
            "disease_agent": {
                "name": diagnosis.agent_name,
                "result": {
                    "disease": diagnosis.disease,
                    "plant_species": diagnosis.plant_species,
                    "confidence": diagnosis.confidence,
                },
                "evidence": diagnosis.evidence,
            },
            "weather_agent": {
                "name": self.weather_agent.AGENT_NAME,
                "result": weather_crop,
            },
            "rag_analysis": rag_analysis,
        }

    def _step_rag(self, image_pil: Image.Image):
        similar_cases = self.rag.find_similar_cases(image_pil)
        analysis = self.rag.analyze_similar_cases(similar_cases)
        return similar_cases, analysis

orchestrator = AgenticOrchestrator(rag_system, disease_agent, weather_agent)
# SECTION 14: EVALUATION & METRICS
# ──────────────────────────────────────────────────────────────

class ModelEvaluator:
    """Comprehensive model evaluation with metrics and visualizations"""

    def __init__(self, orchestrator, processor, label_names):
        self.orchestrator = orchestrator
        self.processor = processor
        self.label_names = label_names
        self.num_classes = len(label_names)

    def evaluate_on_test_set(self, test_data: List[PlantImageData]) -> Dict[str, Any]:
        """Evaluate model on test set and generate all metrics"""

        print("\n" + "="*60)
        print("  📊 EVALUATING MODEL ON TEST SET")
        print("="*60)

        y_true = []
        y_pred = []
        y_scores = []
        predictions_list = []

        # Start CO2 tracking
        tracker = EmissionsTracker(
            project_name="PlantDiseaseRAG_Evaluation",
            output_dir=config.OUTPUT_DIR,
            output_file="emissions.csv",
            allow_multiple_runs=True
        )
        tracker.start()

        # Make predictions
        for item in tqdm(test_data, desc="Evaluating"):
            result = self.orchestrator.diagnose(item.image_pil, use_rag=True)

            true_label = item.label
            pred_label = result["final_diagnosis"]["predicted_label"]
            confidence = result["final_diagnosis"]["confidence"]

            y_true.append(true_label)
            y_pred.append(pred_label if pred_label != -1 else 0)

            # For multi-class ROC, create probability distribution
            prob_dist = np.zeros(self.num_classes)
            if pred_label != -1:
                prob_dist[pred_label] = confidence
            y_scores.append(prob_dist)

            predictions_list.append({
                "image_id": item.image_id,
                "true_label": true_label,
                "true_label_name": item.label_name,
                "pred_label": pred_label,
                "pred_label_name": self.label_names[pred_label] if pred_label != -1 else "Unknown",
                "confidence": confidence,
            })

        # Stop CO2 tracking
        emissions = tracker.stop()

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_scores = np.array(y_scores)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)

        # Classification Report
        report = classification_report(
            y_true, y_pred,
            target_names=self.label_names,
            output_dict=True,
            zero_division=0
        )

        # ROC-AUC (one-vs-rest)
        y_true_binarized = label_binarize(y_true, classes=range(self.num_classes))

        try:
            roc_auc = roc_auc_score(y_true_binarized, y_scores, average='macro', multi_class='ovr')
        except:
            roc_auc = 0.0

        metrics = {
            "accuracy": accuracy,
            "f1_score_macro": f1_macro,
            "f1_score_weighted": f1_weighted,
            "roc_auc_macro": roc_auc,
            "confusion_matrix": cm,
            "classification_report": report,
            "predictions": predictions_list,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_scores": y_scores,
            "y_true_binarized": y_true_binarized,
            "co2_emissions_kg": emissions if emissions else 0.0,
        }

        print(f"\n✅ Evaluation complete!")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1-Score (Macro): {f1_macro:.4f}")
        print(f"   F1-Score (Weighted): {f1_weighted:.4f}")
        print(f"   ROC-AUC (Macro): {roc_auc:.4f}")
        print(f"   CO2 Emissions: {emissions:.6f} kg" if emissions else "   CO2 Emissions: Not tracked")

        return metrics

    def plot_confusion_matrix(self, cm, save_path=None):
        """Plot confusion matrix heatmap with improved visibility"""

        # Create figure with appropriate size based on number of classes
        figsize = max(20, min(40, self.num_classes // 2))
        plt.figure(figsize=(figsize, figsize))

        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0

        # Create mask for zero values
        mask = cm_norm == 0

        # Plot heatmap
        sns.heatmap(
            cm_norm,
            annot=False,  # Don't annotate all cells to avoid clutter
            fmt='.2f',
            cmap='YlOrRd',
            xticklabels=self.label_names,
            yticklabels=self.label_names,
            cbar_kws={'label': 'Normalized Count', 'shrink': 0.8},
            mask=mask,
            vmin=0, vmax=1
        )

        # Add title and labels
        plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')

        # Rotate labels for better readability
        plt.xticks(rotation=90, ha='center', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Confusion matrix saved to {save_path}")

        return plt.gcf()

    def plot_f1_scores(self, report, save_path=None):
        """Plot F1 scores per class with improved visualization"""

        # Filter out non-class keys
        classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
        f1_scores = [report[c]['f1-score'] for c in classes]

        # Sort by F1 score for better visualization
        sorted_indices = np.argsort(f1_scores)[::-1]
        classes = [classes[i] for i in sorted_indices]
        f1_scores = [f1_scores[i] for i in sorted_indices]

        # Create figure
        fig_height = max(10, len(classes) * 0.3)
        plt.figure(figsize=(14, fig_height))

        # Create horizontal bar chart
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(classes)))
        bars = plt.barh(range(len(classes)), f1_scores, color=colors, edgecolor='black', linewidth=0.5)

        # Customize plot
        plt.xlabel('F1-Score', fontsize=14, fontweight='bold')
        plt.ylabel('Disease Class', fontsize=14, fontweight='bold')
        plt.title('F1-Score per Disease Class (Sorted)', fontsize=16, fontweight='bold', pad=20)
        plt.yticks(range(len(classes)), classes, fontsize=9)
        plt.xlim(0, 1.1)
        plt.grid(axis='x', alpha=0.3, linestyle='--')

        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, f1_scores)):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', ha='left', va='center', fontsize=8, fontweight='bold')

        # Add threshold line
        plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold (0.5)')
        plt.legend(loc='lower right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ F1-scores plot saved to {save_path}")

        return plt.gcf()

    def plot_roc_curves(self, y_true_bin, y_scores, save_path=None):
        """Plot ROC curves for multi-class classification with improved visualization"""

        n_classes = self.num_classes

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            try:
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            except:
                fpr[i] = [0, 1]
                tpr[i] = [0, 1]
                roc_auc[i] = 0.0

        # Plot all classes with different colors
        plt.figure(figsize=(14, 10))

        # Plot top 15 classes or all if less
        n_plot = min(15, n_classes)
        sorted_classes = sorted(roc_auc.items(), key=lambda x: x[1], reverse=True)[:n_plot]

        # Use a colormap for better distinction
        colors = plt.cm.tab20(np.linspace(0, 1, len(sorted_classes)))

        # Plot ROC curves for selected classes
        for idx, (i, auc_score) in enumerate(sorted_classes):
            plt.plot(fpr[i], tpr[i], color=colors[idx], lw=2,
                    label=f'{self.label_names[i][:40]}...' if len(self.label_names[i]) > 40 else self.label_names[i],
                    alpha=0.8)

        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.5)')

        # Calculate and plot macro-average ROC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes

        fpr_macro = all_fpr
        tpr_macro = mean_tpr
        roc_auc_macro = auc(fpr_macro, tpr_macro)

        plt.plot(fpr_macro, tpr_macro, color='deeppink', lw=3, linestyle=':',
                label=f'Macro-average (AUC = {roc_auc_macro:.3f})')

        # Customize plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        plt.title('ROC Curves - Multi-class Classification', fontsize=16, fontweight='bold', pad=20)

        # Place legend outside
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
        plt.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ ROC curves saved to {save_path}")

        return plt.gcf()

    def plot_co2_emissions(self, emissions_kg, save_path=None):
        """Plot CO2 emissions with contextual comparisons"""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # 1. Main emissions bar with gradient
        bars = ax1.bar(['Model Evaluation'], [emissions_kg],
                       color='#e74c3c', edgecolor='black', linewidth=2, alpha=0.8)

        # Add gradient effect
        for bar in bars:
            bar.set_facecolor(plt.cm.RdYlGn_r(emissions_kg / 0.05))  # Normalize for color scale

        ax1.set_ylabel('CO2 Emissions (kg)', fontsize=12, fontweight='bold')
        ax1.set_title(f'CO2 Emissions during Evaluation\n{emissions_kg:.6f} kg',
                     fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value label with background
        ax1.text(0, emissions_kg + emissions_kg*0.1, f'{emissions_kg:.6f} kg',
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        # 2. Comparison chart with real-world equivalents
        comparisons = {
            'This Model': emissions_kg,
            'Smartphone\n(1 hour)': 0.007,
            'LED Bulb\n(1 hour)': 0.008,
            'Car\n(1 km)': 0.120,
            'Air Travel\n(1 km)': 0.255,
            'Desktop PC\n(1 hour)': 0.055
        }

        # Sort by value
        sorted_items = sorted(comparisons.items(), key=lambda x: x[1])
        labels, values = zip(*sorted_items)

        # Create horizontal bar chart with different colors
        colors2 = ['#e74c3c' if 'Model' in l else '#3498db' for l in labels]
        bars2 = ax2.barh(labels, values, color=colors2, edgecolor='black', linewidth=1.5, alpha=0.8)

        ax2.set_xlabel('CO2 Emissions (kg)', fontsize=12, fontweight='bold')
        ax2.set_title('Emissions Comparison with Daily Activities',
                     fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3, linestyle='--')

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars2, values)):
            ax2.text(val + val*0.05, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}' if val < 0.01 else f'{val:.3f}',
                    ha='left', va='center', fontsize=10, fontweight='bold')

        # Add annotation about carbon footprint
        if emissions_kg > 0:
            trees_needed = emissions_kg / 0.021  # Average tree absorbs 21kg CO2 per year
            days_to_absorb = emissions_kg / 0.021 * 365

            fig.text(0.5, 0.02,
                    f'🌳 Carbon Footprint: Would take {trees_needed:.1f} trees {days_to_absorb:.0f} days to absorb this CO2',
                    ha='center', fontsize=11, style='italic',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

        plt.suptitle('CO₂ Emissions Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ CO2 emissions plot saved to {save_path}")

        return plt.gcf()

    def generate_all_plots(self, metrics):
        """Generate and save all evaluation plots"""

        print("\n📊 Generating evaluation plots...")

        # Ensure output directory exists
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        # Confusion Matrix
        print("   - Plotting confusion matrix...")
        cm_fig = self.plot_confusion_matrix(
            metrics['confusion_matrix'],
            save_path=os.path.join(config.OUTPUT_DIR, 'confusion_matrix.png')
        )
        plt.close(cm_fig)

        # F1 Scores
        print("   - Plotting F1 scores...")
        f1_fig = self.plot_f1_scores(
            metrics['classification_report'],
            save_path=os.path.join(config.OUTPUT_DIR, 'f1_scores.png')
        )
        plt.close(f1_fig)

        # ROC Curves
        print("   - Plotting ROC curves...")
        roc_fig = self.plot_roc_curves(
            metrics['y_true_binarized'],
            metrics['y_scores'],
            save_path=os.path.join(config.OUTPUT_DIR, 'roc_curves.png')
        )
        plt.close(roc_fig)

        # CO2 Emissions
        print("   - Plotting CO2 emissions...")
        co2_fig = self.plot_co2_emissions(
            metrics['co2_emissions_kg'],
            save_path=os.path.join(config.OUTPUT_DIR, 'co2_emissions.png')
        )
        plt.close(co2_fig)

        print("✅ All plots generated and saved!")

        return {
            'confusion_matrix': os.path.join(config.OUTPUT_DIR, 'confusion_matrix.png'),
            'f1_scores': os.path.join(config.OUTPUT_DIR, 'f1_scores.png'),
            'roc_curves': os.path.join(config.OUTPUT_DIR, 'roc_curves.png'),
            'co2_emissions': os.path.join(config.OUTPUT_DIR, 'co2_emissions.png'),
        }

    def save_metrics_report(self, metrics, save_path=None):
        """Save detailed metrics report with formatted output"""

        if save_path is None:
            save_path = os.path.join(config.OUTPUT_DIR, 'evaluation_report.txt')

        with open(save_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("  AGENTIC-RAG PLANT DISEASE DIAGNOSIS - EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")

            f.write("DATASET CONFIGURATION:\n")
            f.write("-"*40 + "\n")
            f.write(f"  Training samples: {config.TRAIN_SAMPLES:,}\n")
            f.write(f"  Validation samples: {config.VAL_SAMPLES:,}\n")
            f.write(f"  Test samples: {config.TEST_SAMPLES:,}\n")
            f.write(f"  Number of classes: {self.num_classes}\n\n")

            f.write("OVERALL METRICS:\n")
            f.write("-"*40 + "\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  F1-Score (Macro): {metrics['f1_score_macro']:.4f}\n")
            f.write(f"  F1-Score (Weighted): {metrics['f1_score_weighted']:.4f}\n")
            f.write(f"  ROC-AUC (Macro): {metrics['roc_auc_macro']:.4f}\n")
            f.write(f"  CO2 Emissions: {metrics['co2_emissions_kg']:.8f} kg\n\n")

            f.write("PER-CLASS METRICS:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Class Name':<50} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>8}\n")
            f.write("-"*80 + "\n")

            report = metrics['classification_report']
            for class_name in self.label_names:
                if class_name in report:
                    cls_metrics = report[class_name]
                    f.write(f"{class_name:<50} {cls_metrics['precision']:>10.4f} "
                           f"{cls_metrics['recall']:>10.4f} {cls_metrics['f1-score']:>10.4f} "
                           f"{cls_metrics['support']:>8.0f}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("  END OF REPORT\n")
            f.write("="*80 + "\n")

        print(f"✅ Evaluation report saved to {save_path}")
        return save_path

        # SECTION 15: RUN EVALUATION
# ──────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  🧪 LOADING TEST SET AND RUNNING EVALUATION")
print("="*60 + "\n")

# Load test data
test_images = processor.load_split("test", max_samples=config.TEST_SAMPLES)

# Create evaluator
evaluator = ModelEvaluator(orchestrator, processor, label_names)

# Run evaluation
metrics = evaluator.evaluate_on_test_set(test_images)

# Generate all plots
plot_paths = evaluator.generate_all_plots(metrics)

# Save report
report_path = evaluator.save_metrics_report(metrics)

print("\n" + "="*60)
print("  ✅ EVALUATION COMPLETE!")
print("="*60)
print(f"\n📁 All outputs saved to: {config.OUTPUT_DIR}")
print(f"   - Confusion Matrix: {plot_paths['confusion_matrix']}")
print(f"   - F1 Scores: {plot_paths['f1_scores']}")
print(f"   - ROC Curves: {plot_paths['roc_curves']}")
print(f"   - CO2 Emissions: {plot_paths['co2_emissions']}")
print(f"   - Evaluation Report: {report_path}\n")
# ──────────────────────────────────────────────────────────────
# SECTION 16: GRADIO APP (UPDATED - SEPARATE ROWS FOR PLOTS)
# ──────────────────────────────────────────────────────────────

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

class PlantDiagnosisApp:
    """Gradio UI for the Agentic-RAG system with improved visibility"""

    def __init__(self, orchestrator, metrics, plot_paths):
        self.orc = orchestrator
        self.metrics = metrics
        self.plot_paths = plot_paths

    def diagnose(self, image, use_rag, lat, lon, api_key_input):
        if image is None:
            return {}, {}, "<p>No image uploaded.</p>", None, None

        if api_key_input.strip():
            self.orc.weather_agent.api_key = api_key_input.strip()

        result = self.orc.diagnose(image, use_rag=use_rag, lat=lat, lon=lon)

        # Summary with black text styling
        fd = result["final_diagnosis"]
        summary = {
            "🌿 Plant": fd["plant_species"],
            "🦠 Disease": fd["disease"],
            "🩺 Healthy": "❌ No" if not fd["is_healthy"] else "✅ Yes",
            "📊 Confidence": f"{fd['confidence']*100:.1f}%",
        }

        # Weather with black text styling
        w = result["weather_agent"]["result"]["weather"]
        weather_card = {
            "📍 Location": w["city"],
            "🌡️ Temperature": f"{w['temperature_c']:.1f}°C",
            "💧 Humidity": f"{w['humidity_pct']}%",
            "☁️ Condition": w["condition"].title(),
            "🌱 Best Crop": result["weather_agent"]["result"]["best_crop"],
        }

        # Enhanced evidence with two-row layout and better styling
        evidence_html = self._create_evidence_display(result)

        # Charts with improved visibility
        disease_fig = self._make_disease_chart(result)
        crop_fig = self._make_crop_chart(result)

        return summary, weather_card, evidence_html, disease_fig, crop_fig

    def _create_evidence_display(self, result):
        """Create a two-row evidence display with better visibility"""

        evidence_items = result["disease_agent"]["evidence"]
        rag_analysis = result.get("rag_analysis", {})

        # Get top diseases for display
        top_diseases = rag_analysis.get("top_diseases", [])
        top_plants = rag_analysis.get("top_plants", [])

        # Create HTML with two rows
        html = """
        <div style='font-family: Arial, sans-serif; background-color: #f5f5f5; padding: 20px; border-radius: 10px; border: 2px solid #4CAF50; margin-top: 10px; margin-bottom: 20px;'>
            <h3 style='color: #000000; margin-top: 0; margin-bottom: 20px; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; font-size: 18px; font-weight: bold;'>🔬 Evidence & Analysis</h3>

            <!-- First Row: Evidence Items -->
            <div style='display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 25px;'>
        """

        # Add evidence items as cards
        for i, ev in enumerate(evidence_items):
            if "label" in ev.lower():
                bg_color = "#e3f2fd"  # Light blue for label info
                border_color = "#2196F3"
            elif "similarity" in ev.lower():
                bg_color = "#fff3e0"  # Light orange for similarity
                border_color = "#FF9800"
            else:
                bg_color = "#e8f5e8"  # Light green for other
                border_color = "#4CAF50"

            html += f"""
                <div style='flex: 1 1 calc(50% - 15px); background-color: {bg_color};
                           border-left: 5px solid {border_color}; padding: 15px;
                           border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <span style='color: #000000; font-weight: 600; font-size: 14px;'>{ev}</span>
                </div>
            """

        html += """
            </div>

            <!-- Second Row: Top Diseases and Plants -->
            <div style='display: flex; gap: 20px; margin-top: 10px;'>
        """

        # Top Diseases column
        html += """
                <div style='flex: 1; background-color: #ffffff; padding: 20px; border-radius: 8px;
                           border: 1px solid #e74c3c; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <h4 style='color: #000000; margin-top: 0; margin-bottom: 20px; border-bottom: 2px solid #e74c3c; padding-bottom: 10px; font-size: 16px; font-weight: bold;'>
                        🦠 Top Disease Candidates
                    </h4>
        """

        if top_diseases:
            for d in top_diseases:
                confidence_color = "#27ae60" if d['confidence'] > 0.7 else "#e67e22" if d['confidence'] > 0.4 else "#e74c3c"
                html += f"""
                    <div style='margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 6px;'>
                        <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                            <span style='color: #000000; font-weight: bold; font-size: 15px;'>{d['disease']}</span>
                            <span style='color: {confidence_color}; font-weight: bold; font-size: 15px;'>{d['confidence']*100:.1f}%</span>
                        </div>
                        <div style='width: 100%; height: 10px; background-color: #ecf0f1; border-radius: 5px; overflow: hidden;'>
                            <div style='width: {d['confidence']*100}%; height: 10px; background-color: {confidence_color}; border-radius: 5px;'></div>
                        </div>
                    </div>
                """
        else:
            html += "<p style='color: #000000; font-size: 14px;'>No disease candidates found</p>"

        html += """
                </div>

                <!-- Top Plants column -->
                <div style='flex: 1; background-color: #ffffff; padding: 20px; border-radius: 8px;
                           border: 1px solid #27ae60; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <h4 style='color: #000000; margin-top: 0; margin-bottom: 20px; border-bottom: 2px solid #27ae60; padding-bottom: 10px; font-size: 16px; font-weight: bold;'>
                        🌿 Top Plant Matches
                    </h4>
        """

        if top_plants:
            for p in top_plants:
                html += f"""
                    <div style='margin-bottom: 12px; padding: 12px; background-color: #f8f9fa; border-radius: 6px; display: flex; justify-content: space-between; align-items: center;'>
                        <span style='color: #000000; font-weight: bold; font-size: 15px;'>{p['plant']}</span>
                        <span style='background-color: #27ae60; color: white; padding: 5px 12px; border-radius: 20px; font-weight: bold; font-size: 14px;'>{p['confidence']*100:.1f}%</span>
                    </div>
                """
        else:
            html += "<p style='color: #000000; font-size: 14px;'>No plant matches found</p>"

        html += """
                </div>
            </div>
        </div>
        """

        return html

    def _make_disease_chart(self, result):
        """Create disease candidates chart with improved visibility"""
        rag = result.get("rag_analysis", {})
        diseases = rag.get("top_diseases", [])

        # Create figure with larger size for better visibility
        fig, ax = plt.subplots(figsize=(8, 5))

        if not diseases:
            ax.text(0.5, 0.5, "No disease data available", ha="center", va="center",
                   fontsize=14, color='#000000', transform=ax.transAxes, fontweight='bold')
            ax.axis("off")
            return fig

        names = [d["disease"] for d in diseases]  # Full names without truncation
        confs = [d["confidence"] * 100 for d in diseases]

        # Create horizontal bar chart with better spacing
        y_pos = np.arange(len(names))
        colors = ['#e74c3c', '#e67e22', '#3498db'][:len(names)]
        bars = ax.barh(y_pos, confs, color=colors, edgecolor='black', linewidth=1.5, alpha=0.9, height=0.5)

        # Add value labels with better positioning and black text
        for i, (bar, conf, name) in enumerate(zip(bars, confs, names)):
            # Add percentage label
            if conf > 15:  # If bar is wide enough, put label inside
                ax.text(conf - 3, bar.get_y() + bar.get_height()/2,
                       f'{conf:.1f}%', ha='right', va='center', fontsize=12,
                       fontweight='bold', color='white')
            else:  # If bar is narrow, put label outside
                ax.text(conf + 2, bar.get_y() + bar.get_height()/2,
                       f'{conf:.1f}%', ha='left', va='center', fontsize=12,
                       fontweight='bold', color='#000000')

        # Set labels and title with black text
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=12, color='#000000')
        ax.set_xlim(0, 100)
        ax.set_xlabel("Confidence (%)", fontsize=13, fontweight='bold', color='#000000', labelpad=10)
        ax.set_title("🎯 Disease Candidates", fontweight='bold', fontsize=16, color='#000000', pad=20)
        ax.invert_yaxis()

        # Add grid for better readability
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Set tick colors to black
        ax.tick_params(axis='x', colors='#000000', labelsize=11)
        ax.tick_params(axis='y', colors='#000000', labelsize=11)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#000000')
        ax.spines['left'].set_color('#000000')

        plt.tight_layout()
        return fig

    def _make_crop_chart(self, result):
        """Create crop recommendations chart with improved visibility"""
        crops = result.get("weather_agent", {}).get("result", {}).get("top_recommended_crops", [])

        # Create figure with larger size for better visibility
        fig, ax = plt.subplots(figsize=(8, 5))

        if not crops:
            ax.text(0.5, 0.5, "No crop data available", ha="center", va="center",
                   fontsize=14, color='#000000', transform=ax.transAxes, fontweight='bold')
            ax.axis("off")
            return fig

        names = [c["plant"] for c in crops]
        scores = [c["suitability_score"] for c in crops]

        # Create bar chart with better spacing
        x_pos = np.arange(len(names))
        bars = ax.bar(x_pos, scores, color='#27ae60', alpha=0.9, edgecolor='black',
                     linewidth=1.5, width=0.5)

        # Add value labels on top of bars with black text
        for bar, score, name in zip(bars, scores, names):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 2,
                   f'{score:.0f}', ha='center', va='bottom', fontsize=14,
                   fontweight='bold', color='#000000')

            # Add plant name below bar
            ax.text(bar.get_x() + bar.get_width()/2, -3,
                   name, ha='center', va='top', fontsize=13,
                   fontweight='bold', color='#000000', rotation=0)

        # Set labels and title with black text
        ax.set_xlim(-0.5, len(names) - 0.5)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Suitability Score", fontsize=13, fontweight='bold', color='#000000', labelpad=10)
        ax.set_title("🌾 Crop Recommendations", fontweight='bold', fontsize=16, color='#000000', pad=20)

        # Remove x-axis ticks since we added custom labels
        ax.set_xticks([])

        # Add grid for better readability
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Set tick colors to black
        ax.tick_params(axis='y', colors='#000000', labelsize=11)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#000000')
        ax.spines['left'].set_color('#000000')

        plt.tight_layout()
        return fig

    def get_metrics_summary(self):
        """Return metrics summary for display"""
        return {
            "Accuracy": f"{self.metrics['accuracy']:.4f}",
            "F1-Score (Macro)": f"{self.metrics['f1_score_macro']:.4f}",
            "F1-Score (Weighted)": f"{self.metrics['f1_score_weighted']:.4f}",
            "ROC-AUC (Macro)": f"{self.metrics['roc_auc_macro']:.4f}",
            "CO2 Emissions": f"{self.metrics['co2_emissions_kg']:.8f} kg",
            "Test Samples": f"{config.TEST_SAMPLES:,}",
        }

    def launch(self):
        """Launch the Gradio interface with improved styling"""

        # Custom CSS for better visibility and black text
        custom_css = """
        .json-container {
            color: #000000 !important;
            background-color: #ffffff !important;
            border: 2px solid #4CAF50 !important;
            border-radius: 8px !important;
            padding: 15px !important;
        }
        .json-container pre {
            color: #000000 !important;
            font-size: 14px !important;
        }
        .json-key {
            color: #000000 !important;
            font-weight: bold !important;
        }
        .json-value {
            color: #000000 !important;
        }
        .json-string {
            color: #000000 !important;
        }
        .json-boolean {
            color: #000000 !important;
        }
        .gradio-html {
            color: #000000 !important;
        }
        /* Fix plot spacing */
        .plot-container {
            margin-top: 20px;
            margin-bottom: 20px;
            padding: 10px;
            background-color: #ffffff;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        /* Style for all text in the app */
        body, p, h1, h2, h3, h4, h5, h6, label, span, div {
            color: #000000;
        }
        /* Style for markdown text */
        .markdown-text {
            color: #000000 !important;
        }
        /* Add spacing between sections */
        .section-spacing {
            margin-bottom: 30px;
        }
        """

        with gr.Blocks(title="🌿 Plant Disease Diagnosis",
                      theme=gr.themes.Soft(),
                      css=custom_css) as demo:
            gr.Markdown("""
            # 🌿 Agentic-RAG Plant Disease Diagnosis System
            **Two AI Agents**: Disease Prediction (CLIP) + Weather & Crop Recommendation
            """)

            with gr.Tabs():
                # Diagnosis Tab
                with gr.TabItem("🔍 Diagnosis"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            image_input = gr.Image(type="pil", label="📷 Upload Plant Image")
                            use_rag = gr.Checkbox(value=True, label="Use RAG for enhanced diagnosis")

                            with gr.Accordion("📍 Location Settings", open=False):
                                lat_input = gr.Number(value=config.DEFAULT_LAT, label="Latitude")
                                lon_input = gr.Number(value=config.DEFAULT_LON, label="Longitude")
                                api_key_input = gr.Textbox(
                                    value="", placeholder="OpenWeatherMap API Key (optional)",
                                    label="API Key", type="password"
                                )

                            diagnose_btn = gr.Button("🔍 Diagnose Plant Disease", variant="primary", size="lg")

                        with gr.Column(scale=2):
                            with gr.Row():
                                with gr.Column():
                                    # Apply custom CSS class for black text
                                    summary_out = gr.JSON(
                                        label="📋 Diagnosis Summary",
                                        elem_classes=["json-container"]
                                    )
                                with gr.Column():
                                    weather_out = gr.JSON(
                                        label="🌤️ Weather Information",
                                        elem_classes=["json-container"]
                                    )

                            # Evidence in two-row layout
                            evidence_out = gr.HTML(label="🔬 Evidence & Analysis")

                            # Add spacing before charts
                            gr.Markdown("---")
                            gr.Markdown("### 📊 Analysis Results")

                            # Disease Candidates Chart - First Row with proper spacing
                            with gr.Row(elem_classes=["plot-container", "section-spacing"]):
                                disease_chart = gr.Plot(label="🎯 Disease Candidates")

                            # Crop Recommendations Chart - Second Row with proper spacing
                            with gr.Row(elem_classes=["plot-container", "section-spacing"]):
                                crop_chart = gr.Plot(label="🌾 Crop Recommendations")

                # Evaluation Tab
                with gr.TabItem("📊 Model Evaluation"):
                    gr.Markdown("### 📈 Model Performance Metrics")

                    # Metrics summary with black text
                    metrics_display = gr.JSON(
                        value=self.get_metrics_summary(),
                        label="Overall Metrics",
                        elem_classes=["json-container"]
                    )

                    # Evaluation plots
                    with gr.Row():
                        with gr.Column():
                            cm_img = gr.Image(value=self.plot_paths['confusion_matrix'],
                                            label="Confusion Matrix", interactive=False)
                        with gr.Column():
                            f1_img = gr.Image(value=self.plot_paths['f1_scores'],
                                            label="F1 Scores per Class", interactive=False)

                    with gr.Row():
                        with gr.Column():
                            roc_img = gr.Image(value=self.plot_paths['roc_curves'],
                                            label="ROC Curves", interactive=False)
                        with gr.Column():
                            co2_img = gr.Image(value=self.plot_paths['co2_emissions'],
                                            label="CO2 Emissions Analysis", interactive=False)

                    # Add download button for evaluation report
                    with gr.Row():
                        report_btn = gr.File(label="📥 Download Evaluation Report",
                                            value=report_path, interactive=False)

                # About Tab
                with gr.TabItem("ℹ️ About"):
                    gr.Markdown(f"""
                    <div style='color: #000000; font-family: Arial, sans-serif;'>
                        <h2 style='color: #000000;'>🌿 Agentic-RAG Plant Disease Diagnosis System</h2>

                        <h3 style='color: #000000;'>System Architecture</h3>
                        <ul style='color: #000000;'>
                            <li><strong>Agent 1:</strong> Disease Prediction using CLIP embeddings + label matching</li>
                            <li><strong>Agent 2:</strong> Live weather data + crop recommendation</li>
                            <li><strong>RAG:</strong> FAISS vector database for similar case retrieval</li>
                        </ul>

                        <h3 style='color: #000000;'>Dataset Statistics</h3>
                        <ul style='color: #000000;'>
                            <li><strong>Training samples:</strong> {config.TRAIN_SAMPLES:,}</li>
                            <li><strong>Validation samples:</strong> {config.VAL_SAMPLES:,}</li>
                            <li><strong>Test samples:</strong> {config.TEST_SAMPLES:,}</li>
                            <li><strong>Number of classes:</strong> {num_classes}</li>
                        </ul>

                        <h3 style='color: #000000;'>Model Performance</h3>
                        <ul style='color: #000000;'>
                            <li><strong>Accuracy:</strong> {metrics['accuracy']:.4f}</li>
                            <li><strong>F1-Score (Macro):</strong> {metrics['f1_score_macro']:.4f}</li>
                            <li><strong>ROC-AUC:</strong> {metrics['roc_auc_macro']:.4f}</li>
                        </ul>

                        <h3 style='color: #000000;'>Environmental Impact</h3>
                        <ul style='color: #000000;'>
                            <li><strong>CO2 Emissions:</strong> {metrics['co2_emissions_kg']:.8f} kg</li>
                        </ul>

                        <h3 style='color: #000000;'>Technologies Used</h3>
                        <ul style='color: #000000;'>
                            <li>CLIP ViT-B/32 for embeddings</li>
                            <li>FAISS for vector similarity search</li>
                            <li>CodeCarbon for emissions tracking</li>
                            <li>Gradio for interactive UI</li>
                        </ul>
                    </div>
                    """)

            # Connect diagnosis button
            diagnose_btn.click(
                fn=self.diagnose,
                inputs=[image_input, use_rag, lat_input, lon_input, api_key_input],
                outputs=[summary_out, weather_out, evidence_out, disease_chart, crop_chart],
            )

            gr.Markdown("""
            ---
            **Dataset**: PlantVillage | **Model**: CLIP ViT-B/32 | **Vector DB**: FAISS
            *CO2 emissions tracked with CodeCarbon*
            """)

        # Launch with sharing enabled
        demo.launch(share=True, debug=False)

# Launch Gradio App
app = PlantDiagnosisApp(orchestrator, metrics, plot_paths)
print("\n🚀 Launching Gradio interface with improved visibility...")
app.launch()
