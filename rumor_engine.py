"""
DÉTECTEUR DE RUMEURS - VERSION AMÉLIORÉE
Améliorations:
1. Topics configurables (JSON ou paramètres API)
2. Score de viralité corrigé (5 apparitions = viral)
3. Filtrage strict sur le Bénin uniquement
4. Les rumeurs non-virales ne sont PAS enregistrées
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import requests
from dotenv import load_dotenv
from typing import Optional
import re

load_dotenv()

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY", "")
    GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    # Sites FIABLES (pour VÉRIFIER les rumeurs)
    TRUSTED_SOURCES = [
        # --- Médias béninois principaux ---
        "beninwebtv.com",
        "lematinal.bj",
        "ortb.bj",
        "lanouvelletribune.info",
        "lanation.bj",
        "24haubenin.info",
        "actubenin.com",
        "matinlibre.com",
        "lepoint.bj",
        "la-quotidienne.bj",
        "banouto.bj",
        "fraternitefj.com",
        "fraternitefm.bj",
        "jupiterinfo.bj",
        "firstafriquetv.bj",
        "leleaderinfobenin.bj",
        "gueritetvmonde.bj",
        "bjnews.bj",
        "diffo.net",
        "lepotentiel.bj",
        "eketinmagazine.com",

        # --- Médias panafricains fiables ---
        "rfi.fr",
        "bbc.com",
        "dw.com",
        "jeuneafrique.com",
        "africanews.com",
        "rtb.bf",   # Burkina Faso (souvent repris au Bénin)
        "lefaso.net",
        "linfodrome.com",   # Côte d’Ivoire sérieux
        "seneweb.com",      # Sénégal
        "lequotidien.sn",

        # --- Fact-checkers (très importants pour ton système !) ---
        "africacheck.org",
        "benincheck.info",      # (site existant dans certains projets)
        "dubawa.org",           # Afrique de l’Ouest
        "factcheck.org",
        "fullfact.org",
        "snopes.com",

        # --- Institutions béninoises ---
        "gouv.bj",
        "presidence.bj",
        "assemblee-nationale.bj",
        "justice.gouv.bj",
        "finances.bj",
        "msp.bj",               # Ministère de la Santé
        "interieur.gouv.bj",
        "police.bj",

        # --- Organisations internationales ---
        "who.int",
        "un.org",
        "worldbank.org",
        "imf.org",
        "unodc.org",
        "ecowas.int",
    ]


    # Sites à IGNORER complètement
    BLACKLIST = ["archive.org", "webcache.googleusercontent.com"]
    
    OUTPUT_FILE = "rumors_detected.json"
    TOPICS_FILE = "topics.json"  # Fichier de configuration des topics
    RECENT_DAYS = 360
    # Limites pour protéger l'API de recherche (par défaut raisonnable)
    MAX_SEARCH_REQUESTS = 10
    QUERY_DELAY = 2.0 # secondes entre requêtes
    
    # NOUVEAU: Score de viralité corrigé
    VIRALITY_THRESHOLD = 0.50  # Seuil minimal pour vérifier (50%)
    MIN_OCCURRENCES_FOR_VIRAL = 5  # 5 apparitions = réellement viral
    
    # NOUVEAU: Mots-clés Bénin (filtrage intelligent)
    BENIN_KEYWORDS = [
        # Pays
        "bénin", "benin", "béninois", "beninois", "béninoises",
        # Villes principales
        "cotonou", "cotonnou", "porto-novo", "porto novo", "parakou",
        "abomey", "bohicon", "natitingou", "djougou", "lokossa", "ouidah",
        # Régions
        "ouémé", "atlantique", "borgou", "zou", "mono", "couffo", 
        "collines", "plateau", "atacora", "donga", "alibori", "littoral",
        # Personnalités et institutions
        "patrice talon", "talon", "gouvernement béninois", "ceni bénin",
        "assemblée nationale bénin", "présidence bénin"
    ]
    
    # Topics par défaut si aucun fichier n'est fourni
    DEFAULT_TOPICS = {
        "politique": ["Patrice Talon", "CENI", "élections", "gouvernement béninois"],
        "santé": ["vaccination Bénin", "choléra Bénin", "paludisme Bénin"],
        "économie": ["carburant Bénin", "prix Bénin", "CFA"],
        "sécurité": ["sécurité Bénin", "braquage Cotonou"]
    }


def domain_of_url(url: str) -> str:
    """Extrait le domaine d'une URL"""
    if not url:
        return ""
    try:
        from urllib.parse import urlparse
        host = urlparse(url).hostname or ""
        host = host.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return url.lower()


def is_about_benin(text: str) -> bool:
    """
    NOUVEAU: Vérifie si le texte concerne vraiment le Bénin
    Approche flexible : cherche mentions du Bénin
    """
    if not text:
        return False
    
    text_lower = text.lower()
    
    # Enlever accents pour une meilleure détection
    import unicodedata
    text_normalized = unicodedata.normalize('NFD', text_lower)
    text_normalized = ''.join(c for c in text_normalized if unicodedata.category(c) != 'Mn')
    
    # Chercher dans le texte original ET normalisé
    for keyword in Config.BENIN_KEYWORDS:
        keyword_normalized = unicodedata.normalize('NFD', keyword.lower())
        keyword_normalized = ''.join(c for c in keyword_normalized if unicodedata.category(c) != 'Mn')
        
        if keyword in text_lower or keyword_normalized in text_normalized:
            return True
    
    return False


# ==========================================
# GEMINI EMBEDDINGS
# ==========================================
class GeminiEmbedder:
    def __init__(self, api_key: str):
        self.api_key = api_key
        if GEMINI_AVAILABLE and api_key:
            genai.configure(api_key=api_key)
    
    def embed_text(self, text: str) -> List[float]:
        if not GEMINI_AVAILABLE or not self.api_key:
            return [float(hash(text) % 1000) / 1000.0]
        
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logging.error(f"Embedding error: {e}")
            return [float(hash(text) % 1000) / 1000.0]
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        try:
            dot_prod = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(dot_prod) / (norm1 * norm2)
        except Exception:
            return 0.0


# ==========================================
# DÉDUPLICATEUR
# ==========================================
class RumorDeduplicator:
    def __init__(self, similarity_threshold: float = 0.85):
        self.embedder = GeminiEmbedder(Config.GEMINI_API_KEY)
        self.rumor_embeddings = []
        self.similarity_threshold = similarity_threshold
    
    def is_duplicate(self, rumor_text: str) -> bool:
        if not rumor_text.strip():
            return True
        
        new_emb = self.embedder.embed_text(rumor_text)
        
        for entry in self.rumor_embeddings:
            sim = self.embedder.cosine_similarity(new_emb, entry["embedding"])
            if sim >= self.similarity_threshold:
                logging.info(f"   Duplicate (sim: {sim:.2f})")
                return True
        
        self.rumor_embeddings.append({
            "text": rumor_text,
            "embedding": new_emb,
            "date_added": datetime.now().isoformat()
        })
        return False


# ==========================================
# EXTRACTEUR DE RUMEURS
# ==========================================
class RumorExtractor:
    """Détecte les rumeurs dans les résultats de recherche"""
    
    RUMOR_INDICATORS = [
        "rumeur", "info ou intox", "est-ce vrai", "circule",
        "on dit que", "selon des sources", "non confirmé",
        "aurait", "paraît que", "fake news"
    ]
    
    DEMENTI_KEYWORDS = [
        "dément", "démenti", "réfute", "fausse rumeur",
        "clarification", "mise au point", "infirme"
    ]
    
    @classmethod
    def is_rumor_candidate(cls, text: str) -> bool:
        """Vérifie si c'est une rumeur potentielle"""
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in cls.DEMENTI_KEYWORDS):
            return False
        
        return any(ind in text_lower for ind in cls.RUMOR_INDICATORS)
    

    @classmethod
    def extract_rumor_text(cls, title: str, snippet: str) -> Optional[str]:
        """Extrait le texte complet de la rumeur sans tronquer"""
        # Concatène titre + snippet
        full_text = f"{title} {snippet}".strip()

        # Vérifie que c'est une rumeur potentielle
        if not cls.is_rumor_candidate(full_text):
            return None

        # Nettoyage minimal : enlever dates et URLs, mais pas tronquer
        clean = re.sub(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', '', full_text)
        clean = re.sub(r'http\S+', '', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()

        # Retourne tout le texte disponible
        return clean if len(clean) > 20 else None


# ==========================================
# VIRALITY SCORER (CORRIGÉ)
# ==========================================
class ViralityScorer:
    """
    NOUVEAU: Score de viralité corrigé
    5 apparitions = 1.0 (100% viral)
    """
    def score(self, rumor_text: str, occurrences: int) -> float:
        """
        Calcule un score de viralité entre 0 et 1
        - occurrences: nombre de sites où la rumeur apparaît
        - 5 occurrences = 100% viral
        """
        # Score de base: 5 sources = 100%
        base = min(occurrences / Config.MIN_OCCURRENCES_FOR_VIRAL, 1.0)
        
        # Bonus selon mots viraux
        viral_words = ["circule", "buzz", "choc", "explose", "panique", "alerte"]
        bonus = 0.1 * sum(1 for w in viral_words if w in rumor_text.lower())
        
        return min(base + bonus, 1.0)


def jaccard_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity on word tokens"""
    import re
    a_set = set(re.findall(r"\w+", (a or "").lower()))
    b_set = set(re.findall(r"\w+", (b or "").lower()))
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


# ==========================================
# QUERY GENERATOR (AVEC TOPICS CONFIGURABLES)
# ==========================================
class QueryGenerator:
    """
    NOUVEAU: Génère des requêtes basées sur des topics configurables
    """
    def __init__(self, topics: Optional[Dict[str, List[str]]] = None):
        """
        Args:
            topics: Dictionnaire {categorie: [liste de topics]}
                   Si None, utilise les topics par défaut
        """
        if topics is None:
            # Essayer de charger depuis le fichier
            topics = self.load_topics_from_file()
        
        self.topics = topics or Config.DEFAULT_TOPICS
        
        # Indicateurs de rumeurs (pour certaines requêtes)
        self.rumor_indicators = ["rumeur", "info ou intox", "circule", "fake news"]
        
        # Mots-clés généraux pour trouver aussi de l'actualité récente
        self.general_indicators = ["actualité", "news", "dernière minute", "breaking"]
        
        # Compatibilité: regrouper tous les indicateurs dans `self.indicators`
        # (la méthode generate_queries utilisait `self.indicators`)
        self.indicators = self.rumor_indicators + self.general_indicators
        
        logging.info(f"📋 Topics chargés: {list(self.topics.keys())}")
    
    @staticmethod
    def load_topics_from_file(filename: str = Config.TOPICS_FILE) -> Optional[Dict]:
        """Charge les topics depuis un fichier JSON"""
        if not os.path.exists(filename):
            logging.info(f"⚠️  Fichier {filename} non trouvé, utilisation des topics par défaut")
            return None
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                topics = json.load(f)
            logging.info(f"✅ Topics chargés depuis {filename}")
            return topics
        except Exception as e:
            logging.error(f"❌ Erreur lecture {filename}: {e}")
            return None
    
    def generate_queries(self, max_queries: int = 10) -> List[str]:
        """Génère des requêtes de recherche"""
        queries = []
        
        # NOUVEAU: Générer VRAIMENT jusqu'à max_queries
        for category, topics in self.topics.items():
            for topic in topics:  # TOUS les topics, pas juste [:4]
                for indicator in self.indicators:  # TOUS les indicateurs
                    # Ajouter "Bénin" dans la requête
                    query = f'"{topic}" "{indicator}" Bénin -site:gouv.bj'
                    queries.append(query)
                    
                    if len(queries) >= max_queries:
                        return queries[:max_queries]
        
        return queries
    
    @staticmethod
    def save_default_topics(filename: str = Config.TOPICS_FILE):
        """Sauvegarde les topics par défaut dans un fichier (utilitaire)"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(Config.DEFAULT_TOPICS, f, ensure_ascii=False, indent=2)
        logging.info(f"✅ Topics par défaut sauvegardés dans {filename}")


# ==========================================
# WEB SEARCHER
# ==========================================
class WebSearcher:
    def __init__(self, api_key: str, engine_id: str, max_requests: Optional[int] = None, query_delay: float = 0.5):
        self.api_key = api_key
        self.engine_id = engine_id
        self._max_requests = max_requests
        self._requests_made = 0
        self._query_delay = query_delay
    
    def search(self, query: str, num_results: int = 10) -> List[Dict]:
        """Recherche sur le web"""
        if not self.api_key:
            return []
        params = {
            "key": self.api_key,
            "cx": self.engine_id,
            "q": query,
            "num": min(num_results, 10)
        }

        # Retry with exponential backoff on 429 or transient network errors
        max_retries = 3
        backoff = 1
        url = "https://www.googleapis.com/customsearch/v1"

        for attempt in range(1, max_retries + 1):
            # Respect global max requests budget
            if self._max_requests is not None and self._requests_made >= self._max_requests:
                logging.warning(f"Search budget exhausted ({self._requests_made}/{self._max_requests}); skipping query.")
                return []

            # Optional delay between queries to avoid burst
            if self._query_delay and attempt == 1:
                time.sleep(self._query_delay)
            try:
                resp = requests.get(url, params=params, timeout=12)
                # Count this request attempt
                self._requests_made += 1

                # Handle explicit 429 with Retry-After if provided
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    try:
                        wait = int(retry_after) if retry_after is not None else backoff
                    except Exception:
                        wait = backoff
                    logging.warning(f"Search 429 Too Many Requests; retrying after {wait}s (attempt {attempt}/{max_retries})")
                    time.sleep(wait)
                    backoff *= 2
                    continue

                resp.raise_for_status()
                data = resp.json()

                return [
                    {
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "link": item.get("link", ""),
                        "displayLink": item.get("displayLink", "")
                    }
                    for item in data.get("items", [])
                ]

            except requests.exceptions.RequestException as e:
                logging.error(f"Search error (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    logging.info(f"Retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                else:
                    logging.error("Max retries reached for search; giving up on this query.")
                    return []

        return []


# ==========================================
# FACT CHECKER
# ==========================================
from bs4 import BeautifulSoup
class ImprovedFactChecker:
    """Fact-checker avec fetch complet des pages et analyse Gemini intelligente"""
    
    def __init__(self, api_key: Optional[str], searcher):
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.searcher = searcher
        
        if not api_key:
            logging.error("❌ GEMINI_API_KEY manquante - Vérification impossible!")
            self.gemini_available = False
        elif not GEMINI_AVAILABLE:
            logging.error("❌ Module google.generativeai non installé!")
            self.gemini_available = False
        else:
            try:
                genai.configure(api_key=api_key)
                self.gemini_available = True
                logging.info("✅ Gemini configuré et prêt")
            except Exception as e:
                logging.error(f"❌ Erreur configuration Gemini: {e}")
                self.gemini_available = False
    
    # ==========================================
    # ÉTAPE 1: RECHERCHE DE SOURCES FIABLES
    # ==========================================
    
    def find_trusted_sources(self, rumor_text: str) -> List[Dict]:
        """Recherche multi-passes pour trouver des sources fiables"""
        logging.info(f"   🔍 Recherche sources fiables: {rumor_text[:80]}...")
        
        # Extraire mots-clés pertinents
        keywords = self._extract_keywords(rumor_text)
        all_sources = []
        
        # PASSE 1: Médias béninois + mots-clés
        benin_media = ["beninwebtv.com", "lematinal.bj", "lanation.bj", "24haubenin.info", "ortb.bj"]
        query1 = f'{keywords} ({" OR ".join([f"site:{d}" for d in benin_media])})'
        results1 = self.searcher.search(query1, num_results=8)
        all_sources.extend(results1)
        
        # PASSE 2: Sites officiels
        if len(all_sources) < 5:
            official = ["gouv.bj", "presidence.bj"]
            query2 = f'{keywords} ({" OR ".join([f"site:{d}" for d in official])})'
            results2 = self.searcher.search(query2, num_results=5)
            all_sources.extend(results2)
        
        # PASSE 3: Recherche large avec contexte
        if len(all_sources) < 3:
            query3 = f'{keywords} Bénin (officiel OR confirmé OR démenti OR annonce)'
            results3 = self.searcher.search(query3, num_results=10)
            all_sources.extend(results3)
        
        logging.info(f"   📊 {len(all_sources)} sources trouvées")
        return all_sources[:10]  # Max 10 sources
    
    # ==========================================
    # EXTRACTION DE MOTS-CLÉS
    # ==========================================

    def _extract_keywords(self, text: str) -> str:
        """
        Extraction intelligente des mots-clés.
        Utilise Gemini si disponible, sinon fallback simple.
        """
        # Si Gemini n’est pas dispo
        if not self.gemini_available:
            # Fallback simple : garder mots > 4 lettres
            tokens = re.findall(r"\b\w+\b", text.lower())
            filtered = [t for t in tokens if len(t) > 4]
            return " ".join(filtered[:6])  # max 6 mots

        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"""
            Extrait les mots-clés principaux de ce texte.
            Retourne UNIQUEMENT une liste de 3 à 7 mots-clés séparés par des espaces.

            Texte :
            {text}
            """
            response = model.generate_content(prompt)
            keywords = response.text.strip()


            # Nettoyage
            keywords = re.sub(r"[^a-zA-Z0-9À-ÿ \-]", "", keywords)
            return keywords

        except:
            # Si Gemini crashe, fallback simple
            tokens = re.findall(r"\b\w+\b", text.lower())
            filtered = [t for t in tokens if len(t) > 4]
            return " ".join(filtered[:6])


    # ==========================================
    # ÉTAPE 2: FETCH COMPLET DES PAGES
    # ==========================================
    
    def fetch_full_content(self, url: str) -> Optional[str]:
        """
        Récupère le contenu COMPLET d'une page web
        (pas juste le snippet Google)
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; FactCheckBot/1.0)',
                'Accept': 'text/html,application/xhtml+xml',
                'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8'
            }
            
            response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
            response.raise_for_status()
            
            # Parser avec BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extraire le texte de l'article (supprimer scripts, styles, etc.)
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Récupérer le texte principal
            text = soup.get_text(separator=' ', strip=True)
            
            # Nettoyer
            text = re.sub(r'\s+', ' ', text)
            text = text[:8000]  # Limiter à 8000 caractères (pour Gemini)
            
            logging.info(f"      ✅ Fetché: {len(text)} caractères de {url[:50]}...")
            return text
        
        except Exception as e:
            logging.warning(f"      ⚠️ Erreur fetch {url[:50]}: {e}")
            return None
    
    def fetch_all_sources(self, sources: List[Dict]) -> List[Dict]:
        """Fetch le contenu complet de toutes les sources"""
        enriched_sources = []
        
        for src in sources[:5]:  # Limiter à 5 pour éviter trop de requêtes
            url = src.get("link", "")
            if not url:
                continue
            
            full_content = self.fetch_full_content(url)
            
            enriched_sources.append({
                "url": url,
                "domain": src.get("displayLink", ""),
                "title": src.get("title", ""),
                "snippet": src.get("snippet", ""),
                "full_content": full_content or src.get("snippet", "")  # Fallback sur snippet
            })
        
        return enriched_sources
    

    def _fallback_verification(self, rumor_text: str, trusted_sources: List[Dict]) -> Dict:
        """
        Fallback simple si Gemini n'est pas dispo ou erreur
        Basé sur la présence de mots-clés dans les snippets
        """
        positive_indicators = ["confirmé", "vrai", "officiel", "annoncé"]
        negative_indicators = ["démenti", "faux", "infondé", "réfute"]
        
        score = 0
        for src in trusted_sources:
            snippet = src.get("snippet", "").lower()
            if any(word in snippet for word in positive_indicators):
                score += 1
            if any(word in snippet for word in negative_indicators):
                score -= 1
        
        if score > 0:
            verdict = "vrai"
            confidence = min(0.5 + 0.1 * score, 0.9)
        elif score < 0:
            verdict = "faux"
            confidence = min(0.5 + 0.1 * abs(score), 0.9)
        else:
            verdict = "non vérifiable"
            confidence = 0.5
        
        logging.info(f"   ✅ Fallback verdict: {verdict} (score: {confidence:.2f})")
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "reasoning": "Analyse basée sur les snippets des sources fiables.",
            "sources_used": trusted_sources,
        }
    


    def extract_json(self, text: str) -> dict | None:
        """
        Essaie d'extraire le JSON depuis un texte brut renvoyé par Gemini.
        Retourne None si aucun JSON valide n'est trouvé.
        """
        # Nettoyer guillemets typographiques si jamais
        text = text.replace("“", '"').replace("”", '"')
        
        # Extraire le premier bloc JSON { ... }
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return None
        
        json_text = match.group(0)
        
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            return None

    
    # ==========================================
    # ÉTAPE 3: VÉRIFICATION AVEC GEMINI
    # ==========================================
    
    def verify_with_gemini(self, rumor_text: str, trusted_sources: List[Dict]) -> Dict:
        """
        Vérification intelligente avec Gemini
        Analyse le CONTENU COMPLET des sources et utilise le résumé JSON de Gemini pour décider.
        """
        if not self.gemini_available:
            logging.warning("⚠️ Gemini non disponible, utilisation fallback")
            return self._fallback_verification(rumor_text, trusted_sources)
        
        # Étape 1: Fetch le contenu complet
        logging.info("📥 Fetch du contenu complet des sources...")
        enriched_sources = self.fetch_all_sources(trusted_sources)
        
        if not enriched_sources:
            logging.warning("⚠️ Aucun contenu récupéré")
            return self._fallback_verification(rumor_text, trusted_sources)
        
        # Étape 2: Préparer le contexte pour Gemini
        context = self._build_context(enriched_sources)
        
        # Étape 3: Construire le prompt intelligent
        prompt = self._build_intelligent_prompt(rumor_text, context)
        
        try:
            # Appel de Gemini 2.0
            model = genai.GenerativeModel("gemini-2.0-flash")
            logging.info("🤖 Appel Gemini pour analyse...")
            response = model.generate_content(prompt)
            text = response.text.strip()
            print(text)  # Pour debug
            # Étape 4: Parser le JSON renvoyé par Gemini
            try:
                gemini_result = self.extract_json(text)
            except json.JSONDecodeError:
                logging.warning("⚠️ Réponse Gemini non JSON, fallback utilisé")
                return self._fallback_verification(rumor_text, enriched_sources)
            

            score = gemini_result.get("score_veracite", 0.5)
            if score <= 0.49:
                verdict = "FAUX"
            elif score == 0.5:
                verdict = "INCERTAIN"
            else:  # 0.51 <= score <= 1
                verdict = "VRAI"
            
            # Étape 5: Construire le résultat final
            result = {
                "verdict": verdict,
                "score_veracite": gemini_result.get("score_veracite", 0.5),
                "explication": gemini_result.get("explication", ""),
                "sources_utilisees": gemini_result.get("sources_utilisees", []),
                "elements_cles": gemini_result.get("elements_cles", []),
                "recommandation": gemini_result.get("recommandation", "")
            }
            
            logging.info(f"✅ Gemini verdict: {result['verdict']} (score: {result['score_veracite']:.2f})")
            return result
        
        except Exception as e:
            logging.error(f"❌ Erreur Gemini: {e}")
            return self._fallback_verification(rumor_text, trusted_sources)

    
    # ==========================================
    # PROMPT INTELLIGENT POUR GEMINI
    # ==========================================
    
    def _build_intelligent_prompt(self, rumor_text: str, context: str) -> str:
        """
        Construit un prompt pour Gemini 2.0 Flash qui :
        - Résume le contenu complet des sources
        - Vérifie la véracité de la rumeur
        - Fournit un verdict clair et justifié
        """
        current_year = datetime.now().year

        return f"""Tu es un fact-checker expert spécialisé dans les rumeurs au Bénin. 

            RUMEUR À VÉRIFIER:
            "{rumor_text}"

            CONTEXTE (sources fiables analysées):
            {context}

            OBJECTIFS:
            1. Résumer le contenu principal des sources pour chaque point clé.
            2. Vérifier la véracité de la rumeur selon les informations disponibles.
            3. Tenir compte du contexte temporel et légal (dates, événements passés, Constitution, annonces officielles).
            4. Identifier tout élément contradictoire ou incertain.

            FORMAT STRICT:
            Renvoie uniquement un JSON avec les champs suivants :

            {{
            "verdict": "VRAI/FAUX/INCERTAIN",
            "score_veracite": 0.0-1.0,
            "resume_sources": ["Résumé clair de chaque source analysée"],
            "explication": "Analyse détaillée justifiant le verdict",
            "sources_utilisees": ["liste des URLs pertinentes"],
            "elements_cles": ["points clés extraits des sources"],
            "recommandation": "Conseil/action à prendre"
            }}

            EXEMPLE :
            {{
            "verdict": "FAUX",
            "score_veracite": 0.2,
            "resume_sources": ["Article 1: info 2021...", "Article 2: annonce démentie..."],
            "explication": "Aucune source ne confirme la rumeur pour 2026. Sources disponibles concernent 2021.",
            "sources_utilisees": ["URL1", "URL2"],
            "elements_cles": ["Articles 2021", "Pas d'annonce 2026"],
            "recommandation": "Rumeur infondée"
            }}
        """

    
    def _build_context(self, enriched_sources: List[Dict]) -> str:
        """Construit le contexte avec le CONTENU COMPLET des sources"""
        context_parts = []
        
        for i, src in enumerate(enriched_sources, 1):
            context_parts.append(f"""
                SOURCE {i}: {src['domain']}
                URL: {src['url']}
                TITRE: {src['title']}
                CONTENU:
                {src['full_content'][:2000]}...
                ---
            """)
        
        return

# ==========================================
# SYSTÈME PRINCIPAL
# ==========================================
class CorrectRumorDetectionSystem:
    def __init__(self, topics: Optional[Dict[str, List[str]]] = None, debug: bool = False, max_search_requests: Optional[int] = None, query_delay: float = 0.5):
        """
        Args:
            topics: Dictionnaire de topics personnalisés (optionnel)
            debug: Mode debug pour voir plus de détails
        """
        self.query_generator = QueryGenerator(topics)
        self.web_searcher = WebSearcher(
            Config.GOOGLE_SEARCH_API_KEY,
            Config.GOOGLE_SEARCH_ENGINE_ID,
            max_requests=(max_search_requests or Config.MAX_SEARCH_REQUESTS),
            query_delay=(query_delay or Config.QUERY_DELAY)
        )
        self.extractor = RumorExtractor()
        self.deduplicator = RumorDeduplicator()
        self.virality_scorer = ViralityScorer()
        self.fact_checker = ImprovedFactChecker(Config.GEMINI_API_KEY, self.web_searcher)
        self.debug = debug
    
    def run_detection_cycle(self, max_queries: int = 10) -> List[Dict]:
        """Cycle complet de détection

        Collecte TOUTES les rumeurs extraites (virales ou non). Les rumeurs
        non-virales auront un verdict `NON_VIRAL` et seront sauvegardées
        pour audit manuel.
        """
        queries = self.query_generator.generate_queries(max_queries)
        detected_rumors = []  # Toutes les rumeurs (virales ou NON)
        
        for i, query in enumerate(queries, 1):
            logging.info(f"\n{'='*70}")
            logging.info(f"🔍 Recherche {i}/{len(queries)}: {query}")
            
            search_results = self.web_searcher.search(query, num_results=10)
            
            if not search_results:
                logging.info("   Aucun résultat")
                continue
            
            # Filtrer sources non-vérifiées ET concernant le Bénin
            unverified_sources = []
            for result in search_results:
                domain = domain_of_url(result.get("link", ""))
                
                if domain in Config.BLACKLIST:
                    continue
                
                if domain not in Config.TRUSTED_SOURCES:
                    # Vérifier que le contenu concerne le Bénin
                    full_text = f"{result.get('title', '')} {result.get('snippet', '')}"
                    
                    if self.debug:
                        logging.info(f"   🔍 Analyse: {domain}")
                        logging.info(f"      Titre: {result.get('title', '')[:80]}")
                        logging.info(f"      Snippet: {result.get('snippet', '')[:80]}")
                    
                    if not is_about_benin(full_text):
                        logging.info(f"   ⏭️  Ignoré (pas sur le Bénin): {domain}")
                        if self.debug:
                            logging.info(f"      Texte vérifié: {full_text[:100]}")
                        continue
                    
                    unverified_sources.append(result)
                    logging.info(f"   📍 Source non-vérifiée (Bénin): {domain}")
            
            logging.info(f"   ➡️  {len(unverified_sources)} sources non-vérifiées (Bénin)")
            
            # Extraire les rumeurs
            for result in unverified_sources:
                rumor_text = self.extractor.extract_rumor_text(
                    result.get("title", ""),
                    result.get("snippet", "")
                )
                
                if not rumor_text:
                    continue
                
                if self.deduplicator.is_duplicate(rumor_text):
                    continue
                
                # Calculer viralité
                occurrences = 0
                for other in unverified_sources:
                    other_text = f"{other.get('title','')} {other.get('snippet','')}"
                    if jaccard_similarity(rumor_text, other_text) >= 0.25:
                        occurrences += 1

                virality_score = self.virality_scorer.score(rumor_text, occurrences)

                logging.info(f"   ⚠️  RUMEUR: {rumor_text[:100]}...")
                logging.info(f"   📌 Source: {result.get('displayLink')}")
                logging.info(f"   🔥 Viralité: {virality_score:.2f} ({occurrences} occurrences)")

                # Si la viralité est trop faible, on n'appelle pas le vérificateur
                # externe mais on ENREGISTRE quand même la rumeur pour audit.
                if virality_score < Config.VIRALITY_THRESHOLD:
                    logging.info(f"   ⏭️  Non-viral: viralité ({virality_score:.2f}) < seuil ({Config.VIRALITY_THRESHOLD})")

                    # On tente malgré tout une vérification heuristique légère
                    # en recherchant sur les sources fiables et en utilisant
                    # la méthode de fallback pour produire un verdict.
                    try:
                        trusted_sources = self.fact_checker.find_trusted_sources(rumor_text)
                        verification = self.fact_checker._fallback_verification(rumor_text, trusted_sources)
                        # Indiquer que c'est une vérification heuristique (non-Gemini)
                        verification["verification_method"] = "heuristic_fallback"
                    except Exception as e:
                        logging.error(f"Erreur verification heuristique: {e}")
                        verification = {
                            "verdict": "INCERTAIN",
                            "score_veracite": 0.0,
                            "explication": "Erreur lors de la vérification heuristique",
                            "sources_utilisees": [],
                            "recommandation": "Vérification manuelle",
                            "nb_sources_fiables": 0,
                            "verification_method": "heuristic_fallback"
                        }

                    record = {
                        "rumeur": rumor_text,
                        "virality_score": virality_score,
                        "occurrences": occurrences,
                        "source_non_verifiee": {
                            "domain": result.get("displayLink", ""),
                            "url": result.get("link", ""),
                            "titre": result.get("title", "")
                        },
                        "verification": verification,
                        "detected_at": datetime.now(timezone.utc).isoformat(),
                        "note": "NON_VIRAL"
                    }
                    detected_rumors.append(record)
                    continue

                # Vérifier avec sources fiables
                logging.info(f"   🔬 Vérification...")
                trusted_sources = self.fact_checker.find_trusted_sources(rumor_text)
                verification = self.fact_checker.verify_with_gemini(rumor_text, trusted_sources)
                
                verdict = verification.get("verdict", "?")
                score = verification.get("score_veracite", 0)
                nb_sources = verification.get("nb_sources_fiables", 0)
                
                if verdict == "FAUX":
                    logging.info(f"   ❌ FAUX (score: {score:.2f}, {nb_sources} sources)")
                elif verdict == "VRAI":
                    logging.info(f"   ✅ VRAI (score: {score:.2f}, {nb_sources} sources)")
                else:
                    logging.info(f"   ⚠️  {verdict} ({nb_sources} sources)")
                
                # Enregistrer SEULEMENT si viral
                record = {
                    "rumeur": rumor_text,
                    "virality_score": virality_score,
                    "occurrences": occurrences,
                    "source_non_verifiee": {
                        "domain": result.get("displayLink", ""),
                        "url": result.get("link", ""),
                        "titre": result.get("title", "")
                    },
                    "verification": verification,
                    "detected_at": datetime.now(timezone.utc).isoformat()
                }
                
                detected_rumors.append(record)
                time.sleep(0.5)
        
        return detected_rumors
    
    def save_results(self, records: List[Dict], filename: str = Config.OUTPUT_FILE):
        """Sauvegarde les résultats"""
        # Écrire le fichier à côté du script pour éviter les confusions de cwd
        base_dir = os.path.dirname(__file__) or os.getcwd()
        out_path = os.path.join(base_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        logging.info(f"\n💾 {len(records)} rumeurs sauvegardées: {out_path}")




