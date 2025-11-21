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
        "beninpolitique.org",
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
        "fr.news.yahoo.com",
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

    def _generate_search_queries(self, rumor_text: str) -> List[str]:
        """
        Transforme n'importe quelle rumeur en plusieurs requêtes Google.
        Utilise :
        - Extraction de mots-clés (lieux, sujets, personnes)
        - Reformulations simples
        - Optionnel : appel à Gemini pour reformuler
        """
        # Étape 1: extraction brute de mots importants
        tokens = re.findall(r"\b\w+\b", rumor_text.lower())
        keywords = [t for t in tokens if len(t) > 4]

        # Étape 2: requête brute
        queries = [" ".join(keywords[:6])]

        # Étape 3: créer 2-3 variantes simples
        queries.append(" ".join(keywords[:6]) + " actualité")
        queries.append(" ".join(keywords[:6]) + " info")
        queries.append(" ".join(keywords[:6]) + " rumeur")

        # Étape 4: Option Gemini pour générer 3-5 requêtes alternatives si dispo
        if self.gemini_available:
            try:
                model = genai.GenerativeModel("gemini-2.5-flash")
                prompt = f"""
                Reformule la rumeur suivante en 5 requêtes Google optimisées pour rechercher
                des articles fiables. Ne renvoie que les requêtes, pas de phrases.
                
                Rumeur :
                {rumor_text}
                """
                resp = model.generate_content(prompt)
                for line in resp.text.split("\n"):
                    q = line.strip("-• ").strip()
                    if len(q) > 5:
                        queries.append(q)
            except Exception:
                pass

        # Retirer doublons et limiter
        final_queries = list(dict.fromkeys(queries))
        return final_queries[:8]

    
    def find_trusted_sources(self, rumor_text: str) -> List[Dict]:
        logging.info(f"🔍 Recherche pour : {rumor_text[:80]}")
        queries = self._generate_search_queries(rumor_text)
        logging.info(f"🔑 Requêtes générées : {queries}")

        all_results = []

        for q in queries:
            try:
                logging.info(f"📡 Recherche Google : {q}")
                res = self.searcher.search(q, num_results=10)
                all_results.extend(res)
            except Exception as e:
                logging.warning(f"⚠️ Erreur requête {q}: {e}")

        # Filtrer par sources fiables
        trusted, partial = [], []
        for result in all_results:
            domain = self._extract_domain(result.get("link", ""))
            if domain in Config.TRUSTED_SOURCES:
                trusted.append(result)
            elif domain.endswith(".bj"):
                partial.append(result)

        final = trusted + partial
        logging.info(f"📊 Total fiables trouvés : {len(final)}")
        return final[:10]

    
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
            model = genai.GenerativeModel("gemini-2.5-flash")
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
        

    @staticmethod
    def _extract_domain(url: str) -> str:
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
            return ""

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
    # FONCTION 1: _build_context (CORRECTION)
    # ==========================================

    def _build_context(self, enriched_sources: List[Dict]) -> str:
        """
        Construit le contexte avec le CONTENU COMPLET des sources
        CORRECTION: Retourne maintenant le contexte assemblé
        """
        if not enriched_sources:
            return "Aucune source fiable trouvée."
        
        context_parts = []
        
        for i, src in enumerate(enriched_sources, 1):
            # Extraire le contenu (jusqu'à 2500 caractères par source)
            content = src.get('full_content', '')[:2500]
            domain = src.get('domain', 'inconnu')
            url = src.get('url', 'N/A')
            title = src.get('title', 'Sans titre')
            
            context_parts.append(f"""
    SOURCE {i}: {domain}
    URL: {url}
    TITRE: {title}
    CONTENU:
    {content}
    {"..." if len(src.get('full_content', '')) > 2500 else ""}
    ---
    """)
        
        # CORRECTION CRITIQUE: Joindre et retourner le contexte complet
        final_context = "\n".join(context_parts)
        
        # Log pour debug
        logging.info(f"📝 Contexte construit: {len(final_context)} caractères, {len(enriched_sources)} sources")
        
        return final_context


    # ==========================================
    # FONCTION 2: _build_intelligent_prompt (AMÉLIORATION)
    # ==========================================

    def _build_intelligent_prompt(self, rumor_text: str, context: str) -> str:
        """
        Prompt ULTIME combinant toutes les règles importantes
        """
        from datetime import datetime
        current_year = datetime.now().year
        num_sources = context.count("SOURCE ")
        context_length = len(context)

        return f"""Tu es un fact-checker expert spécialisé dans les rumeurs au Bénin.

        RUMEUR À VÉRIFIER:
        "{rumor_text}"

        SOURCES FIABLES DISPONIBLES ({num_sources} sources, {context_length} caractères):
        {context}

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ⚠️  RÈGLE ABSOLUE DE COHÉRENCE (PRIORITÉ MAXIMALE)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        🚫 INTERDICTIONS ABSOLUES:
        1. Ne JAMAIS mettre verdict "VRAI" si ton explication dit que la rumeur est FAUSSE
        2. Ne JAMAIS mettre verdict "FAUX" si ton explication dit que la rumeur est VRAIE
        3. Ne JAMAIS mettre score > 0.6 si tu écris "aucune source ne confirme" ou "rumeur est fausse"
        4. Ne JAMAIS mettre score < 0.4 si tu écris "sources confirment" ou "information vérifiée"
        5. Ne JAMAIS dire "aucune source fiable" alors que j'ai fourni {num_sources} sources avec contenu
        6. Ne JAMAIS laisser de listes vides (resume_sources, sources_utilisees, elements_cles)
        7. Ne JAMAIS mettre de balises markdown ```json dans la réponse
        8. Ne JAMAIS dire "je n'ai pas accès aux sources" - elles sont CI-DESSUS

        ✅ OBLIGATION: Verdict, score ET explication doivent être 100% COHÉRENTS

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        📋 ÉTAPES OBLIGATOIRES À SUIVRE
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        1. Lis le CONTENU COMPLET de chaque SOURCE ci-dessus ({num_sources} sources)
        2. Résume ce que dit CHAQUE source individuellement (champ "resume_sources")
        3. Note les URLs des sources pertinentes (champ "sources_utilisees")
        4. Extrais les faits clés du contenu (champ "elements_cles")
        5. Détermine le verdict basé sur le CONTENU RÉEL (pas sur des suppositions)
        6. Choisis un score COHÉRENT avec ton verdict et ton explication

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        📊 GUIDE DE SCORING PRÉCIS
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        ┌─────────────────────────────────────────────────────────────────────┐
        │ RUMEUR VRAIE → Verdict: "VRAI" + Score 0.70-1.0                    │
        │                                                                      │
        │ ✓ Sources CONFIRMENT explicitement la rumeur                        │
        │ ✓ Annonces officielles ou articles qui ATTESTENT le fait            │
        │ ✓ Événement mentionné dans plusieurs sources fiables                │
        │                                                                      │
        │ Exemple: Rumeur "Réforme électorale annoncée"                      │
        │          + Sources: "Gouvernement annonce réforme électorale"       │
        │          → Verdict: VRAI, Score: 0.85                              │
        └─────────────────────────────────────────────────────────────────────┘

        ┌─────────────────────────────────────────────────────────────────────┐
        │ RUMEUR FAUSSE → Verdict: "FAUX" + Score 0.0-0.30                   │
        │                                                                      │
        │ ✓ Sources DÉMENTENT explicitement la rumeur                         │
        │ ✓ ABSENCE TOTALE de mention dans toutes les sources fiables         │
        │ ✓ Sources parlent d'événements récents SANS mentionner la rumeur    │
        │ ✓ Sources contredisent directement la rumeur                        │
        │                                                                      │
        │ Exemple: Rumeur "Palais présidentiel a brûlé"                      │
        │          + Sources: Conférences au palais récentes, zéro mention feu│
        │          → Verdict: FAUX, Score: 0.12                              │
        │                                                                      │
        │ ⚠️  IMPORTANT: ABSENCE de confirmation = FAUX (pas INCERTAIN!)      │
        └─────────────────────────────────────────────────────────────────────┘

        ┌─────────────────────────────────────────────────────────────────────┐
        │ INCERTAIN → Verdict: "INCERTAIN" + Score 0.40-0.60                 │
        │                                                                      │
        │ ✓ Sources contradictoires (certaines confirment, d'autres démentent)│
        │ ✓ Informations partielles ou ambiguës                               │
        │ ✓ Sources insuffisantes pour trancher définitivement                │
        │ ✓ Besoin de sources supplémentaires                                 │
        │                                                                      │
        │ Exemple: Rumeur "Ministre va démissionner"                         │
        │          + Sources: Un média dit oui, gouvernement ne confirme pas  │
        │          → Verdict: INCERTAIN, Score: 0.50                         │
        └─────────────────────────────────────────────────────────────────────┘

        RÈGLES DE SCORE STRICTES:
        - Score 0.00-0.30 → verdict DOIT être "FAUX"
        - Score 0.31-0.49 → verdict DOIT être "FAUX" (rumeur probablement fausse)
        - Score 0.50 → verdict DOIT être "INCERTAIN"
        - Score 0.51-0.69 → verdict DOIT être "VRAI" (rumeur probablement vraie)
        - Score 0.70-1.00 → verdict DOIT être "VRAI"

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ⚠️  PIÈGES À ÉVITER ABSOLUMENT
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        ❌ ERREUR 1: "La rumeur est fausse" + score 0.9
        ✅ CORRECT: "La rumeur est fausse" + score 0.1-0.2

        ❌ ERREUR 2: "Sources confirment l'information" + score 0.2
        ✅ CORRECT: "Sources confirment l'information" + score 0.8-0.9

        ❌ ERREUR 3: "Aucune source ne mentionne cet événement" + verdict VRAI
        ✅ CORRECT: "Aucune source ne mentionne cet événement" + verdict FAUX

        ❌ ERREUR 4: "Sources démentent la rumeur" + score 0.85
        ✅ CORRECT: "Sources démentent la rumeur" + score 0.1-0.2

        ❌ ERREUR 5: Liste vide dans resume_sources alors que tu as {num_sources} sources
        ✅ CORRECT: Résumer CHAQUE source individuellement

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        🌍 CONTEXTE TEMPOREL
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        - Année actuelle: {current_year}
        - Vérifie TOUJOURS les DATES dans les sources
        - Un article de 2021 parlant d'élections 2021 NE concerne PAS une rumeur sur 2026
        - Une conférence de presse au palais en 2024 PROUVE que le palais n'a pas brûlé en 2024

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        📄 FORMAT JSON STRICT
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        {{
        "verdict": "VRAI/FAUX/INCERTAIN",
        "score_veracite": 0.0-1.0,
        "resume_sources": [
            "SOURCE 1 (nom du média): Résumé de ce que dit l'article...",
            "SOURCE 2 (nom du média): Résumé de ce que dit l'article...",
            "SOURCE 3 (nom du média): Résumé de ce que dit l'article..."
        ],
        "explication": "Analyse détaillée COHÉRENTE avec le verdict et le score",
        "sources_utilisees": ["URL1", "URL2", "URL3"],
        "elements_cles": [
            "Élément clé 1 extrait du contenu",
            "Élément clé 2 extrait du contenu",
            "Élément clé 3 extrait du contenu"
        ],
        "recommandation": "Action à prendre basée sur le verdict"
        }}

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ✅ EXEMPLE COMPLET (RUMEUR FAUSSE)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        Rumeur: "Le palais présidentiel du Bénin a pris feu récemment"

        Sources analysées:
        - SOURCE 1: Conférence de presse au Palais de la Marina le 8 février 2024
        - SOURCE 2: Discours du président au palais le 20 décembre 2024
        - SOURCE 3: Article Jeune Afrique janvier 2025 sur affaires politiques au palais
        - Résultat: Aucun article ne mentionne un incendie

        ✅ BONNE RÉPONSE:
        {{
        "verdict": "FAUX",
        "score_veracite": 0.12,
        "resume_sources": [
            "SOURCE 1 (Présidence Bénin): Transcription conférence de presse du Président Talon au Palais de la Marina le 8 février 2024. Aucune mention d'incendie.",
            "SOURCE 2 (Présidence Bénin): Message sur l'état de la Nation prononcé au palais le 20 décembre 2024. Le palais est opérationnel.",
            "SOURCE 3 (Jeune Afrique): Article du 16 janvier 2025 traitant d'affaires politiques. Mentionne des événements au palais sans aucune référence à un incendie."
        ],
        "explication": "La rumeur est fausse. Aucune des trois sources fiables ne mentionne d'incendie au palais présidentiel. Au contraire, plusieurs événements officiels se sont tenus au Palais de la Marina en 2024 et début 2025 (conférence de presse février 2024, discours décembre 2024, affaires politiques janvier 2025), ce qui confirme que le palais est pleinement opérationnel. Un incendie serait un événement majeur qui aurait été largement relayé par les médias.",
        "sources_utilisees": [
            "https://presidence.bj/actualite/point-presse/325/",
            "https://presidence.bj/actualite/discours-interviews/363/",
            "https://www.jeuneafrique.com/1648428/politique/"
        ],
        "elements_cles": [
            "Événements officiels récents au palais (février et décembre 2024)",
            "Aucune mention d'incendie dans aucune source fiable",
            "Palais utilisé normalement pour activités gouvernementales en 2025",
            "Absence de couverture médiatique d'un tel événement majeur"
        ],
        "recommandation": "Rumeur infondée - Ne pas relayer. Démentir si elle se propage."
        }}

        ❌ MAUVAISE RÉPONSE (INTERDITE):
        {{
        "verdict": "VRAI",
        "score_veracite": 0.9,
        "explication": "La rumeur est fausse. Aucune source ne confirme...",
        "resume_sources": [],
        "sources_utilisees": []
        }}
        ☝️ CECI EST STRICTEMENT INTERDIT: verdict/score/explication incohérents + listes vides!

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ✅ EXEMPLE COMPLET (RUMEUR VRAIE)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        Rumeur: "Réformes constitutionnelles annoncées au Bénin"

        Sources analysées:
        - SOURCE 1: Article BBC mars 2024 sur réformes constitutionnelles
        - SOURCE 2: Article La Nation confirmant les réformes
        - SOURCE 3: Communiqué présidence sur les amendements
        - Résultat: Toutes les sources confirment les réformes

        ✅ BONNE RÉPONSE:
        {{
        "verdict": "VRAI",
        "score_veracite": 0.88,
        "resume_sources": [
            "SOURCE 1 (BBC): Article du 15 mars 2024 annonçant que le gouvernement béninois a présenté des réformes constitutionnelles. Le président Talon déclare vouloir moderniser le système électoral.",
            "SOURCE 2 (La Nation): Article du 16 mars 2024 confirmant les annonces de réformes. Détails sur les amendements proposés concernant la CENI.",
            "SOURCE 3 (Présidence Bénin): Communiqué officiel détaillant les réformes constitutionnelles et le calendrier de mise en œuvre."
        ],
        "explication": "La rumeur est vraie. Trois sources fiables (BBC, La Nation, Présidence du Bénin) confirment explicitement que des réformes constitutionnelles ont été annoncées au Bénin en mars 2024. Les articles citent des déclarations officielles et détaillent les changements proposés, notamment concernant le système électoral et la CENI. Il s'agit d'une information vérifiée par des sources gouvernementales et des médias reconnus.",
        "sources_utilisees": [
            "https://www.bbc.com/afrique/articles/reformes-2024",
            "https://lanation.bj/actualites/reformes-constitutionnelles",
            "https://presidence.bj/communiques/reformes"
        ],
        "elements_cles": [
            "Réformes constitutionnelles officiellement annoncées mars 2024",
            "Confirmé par la Présidence et médias fiables internationaux",
            "Concerne le système électoral et la CENI",
            "Sources datées de 2024 (pertinent et récent)"
        ],
        "recommandation": "Information confirmée - Peut être relayée en citant les sources officielles"
        }}

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        🔍 AUTO-VÉRIFICATION FINALE (AVANT D'ENVOYER TA RÉPONSE)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        Pose-toi ces questions:

        1. ✓ Mon verdict correspond-il à mon explication?
        (Si j'écris "faux", est-ce que verdict = "FAUX"?)

        2. ✓ Mon score correspond-il à mon verdict?
        (Faux = 0.0-0.3, Incertain = 0.4-0.6, Vrai = 0.7-1.0)

        3. ✓ Ai-je résumé TOUTES les sources fournies?
        (resume_sources doit avoir {num_sources} éléments)

        4. ✓ Ai-je listé les URLs utilisées?
        (sources_utilisees ne doit PAS être vide)

        5. ✓ Ai-je extrait des éléments clés concrets?
        (elements_cles doit contenir des faits précis)

        6. ✓ Mon JSON est-il valide sans balises markdown?
        (Pas de ```json avant ni ``` après)

        SI UNE SEULE RÉPONSE EST "NON", CORRIGE AVANT D'ENVOYER!

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        MAINTENANT, ANALYSE LES {num_sources} SOURCES CI-DESSUS ET RÉPONDS EN JSON.
        """




    # ==========================================
    # FONCTION 3: verify_with_gemini (AMÉLIORATION DEBUG)
    # ==========================================

    def verify_with_gemini(self, rumor_text: str, trusted_sources: List[Dict]) -> Dict:
        """
        Vérification intelligente avec Gemini
        VERSION AMÉLIORÉE avec logs de debug
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
        
        # Étape 2: Construire le contexte
        context = self._build_context(enriched_sources)
        
        # DEBUG: Vérifier que le contexte n'est pas vide
        if not context or context == "Aucune source fiable trouvée.":
            logging.error("❌ ERREUR: Contexte vide ou invalide!")
            logging.error(f"   Enriched sources: {len(enriched_sources)}")
            logging.error(f"   Context: {context[:100]}...")
            return self._fallback_verification(rumor_text, trusted_sources)
        
        logging.info(f"✅ Contexte OK: {len(context)} chars, {len(enriched_sources)} sources")
        
        # Étape 3: Construire le prompt
        prompt = self._build_intelligent_prompt(rumor_text, context)
        
        # DEBUG: Afficher un extrait du prompt
        logging.info(f"📤 Prompt envoyé à Gemini ({len(prompt)} chars)")
        logging.info(f"   Extrait: {prompt[:200]}...")
        
        try:
            # Appel Gemini
            model = genai.GenerativeModel("gemini-2.5-flash")
            logging.info("🤖 Appel Gemini...")
            response = model.generate_content(prompt)
            text = response.text.strip()
            
            # DEBUG: Afficher la réponse brute
            logging.info("📥 Réponse Gemini reçue:")
            logging.info(f"   {text[:300]}...")
            
            # Parser le JSON
            try:
                gemini_result = self.extract_json(text)
                
                if not gemini_result:
                    logging.error("❌ Pas de JSON valide dans la réponse")
                    return self._fallback_verification(rumor_text, enriched_sources)
                
                # Vérifier que les champs ne sont pas vides
                if not gemini_result.get("resume_sources"):
                    logging.warning("⚠️ resume_sources vide dans la réponse Gemini!")
                if not gemini_result.get("sources_utilisees"):
                    logging.warning("⚠️ sources_utilisees vide dans la réponse Gemini!")
                if not gemini_result.get("elements_cles"):
                    logging.warning("⚠️ elements_cles vide dans la réponse Gemini!")
                
                # Déterminer le verdict final
                score = gemini_result.get("score_veracite", 0.5)
                if score <= 0.49:
                    verdict = "FAUX"
                elif score == 0.5:
                    verdict = "INCERTAIN"
                else:
                    verdict = "VRAI"
                
                # Construire le résultat
                result = {
                    "verdict": verdict,
                    "score_veracite": score,
                    "explication": gemini_result.get("explication", ""),
                    "sources_utilisees": gemini_result.get("sources_utilisees", []),
                    "elements_cles": gemini_result.get("elements_cles", []),
                    "resume_sources": gemini_result.get("resume_sources", []),
                    "recommandation": gemini_result.get("recommandation", "")
                }
                
                logging.info(f"✅ Gemini verdict: {result['verdict']} (score: {result['score_veracite']:.2f})")
                logging.info(f"   {len(result['sources_utilisees'])} sources, {len(result['elements_cles'])} éléments clés")
                
                return result
                
            except json.JSONDecodeError as e:
                logging.error(f"❌ Erreur parsing JSON: {e}")
                logging.error(f"   Texte reçu: {text[:500]}")
                return self._fallback_verification(rumor_text, enriched_sources)
        
        except Exception as e:
            logging.error(f"❌ Erreur Gemini: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_verification(rumor_text, trusted_sources)

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




