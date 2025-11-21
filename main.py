from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # ← AJOUT IMPORTA
from pydantic import BaseModel, Field
from typing import List, Optional
from rumor_engine import (
    QueryGenerator,
    WebSearcher,
    RumorExtractor,
    RumorDeduplicator,
    ViralityScorer,
    ImprovedFactChecker as FactChecker,
    Config
)

app = FastAPI(
    title="Détecteur de rumeurs Bénin",
    description="""
    API pour détecter et vérifier la véracité des rumeurs concernant le Bénin.
    
    **Fonctionnalités :**
    - Recherche de rumeurs sur internet selon des topics configurables.
    - Déduplication des rumeurs.
    - Évaluation de la viralité.
    - Vérification via sources fiables et Gemini AI.
    """,
    version="1.0.0"
)

# 🔥 Middleware CORS — À NE SURTOUT PAS OUBLIER
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou mets ton domaine pour limiter : ["https://monfrontend.com"]
    allow_credentials=True,
    allow_methods=["*"],  # Autorise OPTIONS (résout ton erreur 405)
    allow_headers=["*"],
)


# ----------------------------
# Models
# ----------------------------
class SearchRequest(BaseModel):
    topics: Optional[dict] = Field(
        None, description="Dictionnaire des topics à rechercher, e.g. {'politique': ['élections']}"
    )
    max_queries: Optional[int] = Field(
        10, description="Nombre maximal de requêtes de recherche à générer dans ce cycle"
    )

class RumorItem(BaseModel):
    rumor: str = Field(..., description="Texte complet de la rumeur détectée")
    virality: float = Field(..., description="Score de viralité (0.0 à 1.0)")
    link: str = Field(..., description="Lien de la source où la rumeur a été trouvée")
    verification: Optional[dict] = Field(None, description="Résultat de la vérification avec Gemini/FactChecker")

class VerifyRequest(BaseModel):
    rumors: List[str] = Field(..., description="Liste de textes de rumeurs à vérifier")

# ----------------------------
# Initialisation des outils partagés
# ----------------------------
searcher = WebSearcher(Config.GOOGLE_SEARCH_API_KEY, Config.GOOGLE_SEARCH_ENGINE_ID)
extractor = RumorExtractor()
dedup = RumorDeduplicator()
scorer = ViralityScorer()
fact_checker = FactChecker(Config.GEMINI_API_KEY, searcher)

# ----------------------------
# Endpoint véracité
# ----------------------------
@app.post("/verify", summary="Vérifier la véracité de rumeurs", response_description="Résultat de vérification pour chaque rumeur")
def verify_rumors(request: VerifyRequest):
    """
    Vérifie une ou plusieurs rumeurs en utilisant des sources fiables et Gemini AI.

    - **Input**: liste de rumeurs à vérifier
    - **Output**: dictionnaire où chaque rumeur est associée à son verdict JSON (VRAI/FAUX/INCERTAIN)
    """
    results = {}
    for rumor_text in request.rumors:
        trusted_sources = fact_checker.find_trusted_sources(rumor_text)
        verification = fact_checker.verify_with_gemini(rumor_text, trusted_sources)
        results[rumor_text] = verification
    return results

# ----------------------------
# Endpoint recherche
# ----------------------------
@app.post("/search", summary="Rechercher des rumeurs en ligne", response_description="Liste des rumeurs détectées")
def search_rumors(request: SearchRequest):
    """
    Recherche automatiquement des rumeurs sur Internet pour les topics spécifiés.

    - **Input**:
        - `topics`: dictionnaire de topics à rechercher (optionnel)
        - `max_queries`: nombre maximal de requêtes à générer
    - **Output**: liste de rumeurs détectées avec:
        - texte de la rumeur
        - score de viralité
        - lien source
        - résultat de la vérification (Gemini/FactChecker)
    """
    generator = QueryGenerator(request.topics)
    queries = generator.generate_queries(request.max_queries)
    detected_rumors = []

    for q in queries:
        results = searcher.search(q)
        for r in results:
            rumor_text = extractor.extract_rumor_text(r.get("title", ""), r.get("snippet", ""))
            if not rumor_text:
                continue
            if dedup.is_duplicate(rumor_text):
                continue
            virality = scorer.score(rumor_text, 1)

            # Vérification directe via FactChecker
            trusted_sources = fact_checker.find_trusted_sources(rumor_text)
            verification = fact_checker.verify_with_gemini(rumor_text, trusted_sources)

            detected_rumors.append({
                "rumor": rumor_text,
                "virality": virality,
                "link": r.get("link"),
                "verification": verification
            })

    return {"detected_rumors": detected_rumors}
