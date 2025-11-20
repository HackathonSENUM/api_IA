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
class FactChecker:
    """Vérifie les rumeurs avec des sources FIABLES"""
    
    def __init__(self, api_key: str, searcher: WebSearcher):
        self.api_key = api_key
        self.searcher = searcher
        if GEMINI_AVAILABLE and api_key:
            genai.configure(api_key=api_key)
    
    def find_trusted_sources(self, rumor_text: str) -> List[Dict]:
        """Cherche des infos sur la rumeur sur les sites fiables"""
        keywords = rumor_text.split()[:6]
        # Utiliser TOUS les domaines fiables (pas seulement les 5 premiers)
        trusted_domains = " OR ".join([f"site:{d}" for d in Config.TRUSTED_SOURCES])
        query = f'{" ".join(keywords)} ({trusted_domains})'

        logging.info(f"   🔍 Recherche sur sources fiables (requête): {query[:200]}...")

        # Requête principale: demander davantage de résultats pour augmenter les chances
        results = self.searcher.search(query, num_results=20)

        # Loguer les résultats reçus (quelques éléments) pour diagnostic
        if results:
            logging.info(f"   ℹ️  {len(results)} résultats reçus de l'API de recherche (examining up to 10)")
            for idx, r in enumerate(results[:10], 1):
                logging.debug(f"      Result {idx}: domain={domain_of_url(r.get('link',''))} title={r.get('title','')[:80]} snippet={r.get('snippet','')[:120]}")

        trusted_results = []
        for r in results:
            domain = domain_of_url(r.get("link", ""))
            if domain in Config.TRUSTED_SOURCES:
                trusted_results.append(r)

        logging.info(f"   📰 {len(trusted_results)} sources fiables trouvées (après filtrage)")

        # Si aucune source fiable trouvée, faire une recherche secondaire plus large
        if not trusted_results:
            logging.info("   🔎 Aucune source fiable directe trouvée — tentative de recherche secondaire élargie...")
            # Chercher indices explicites de confirmation/démenti dans des snippets
            intent_terms = "démenti OR dément OR faux OR " \
                           "confirmé OR officiel OR communiqué OR \"mise au point\""
            secondary_query = f'{" ".join(keywords)} ({intent_terms})'
            logging.info(f"   🔍 Secondary query: {secondary_query[:200]}...")
            secondary_results = self.searcher.search(secondary_query, num_results=20)

            if secondary_results:
                logging.info(f"   ℹ️  {len(secondary_results)} résultats secondaires reçus")
                for idx, r in enumerate(secondary_results[:10], 1):
                    logging.debug(f"      Sec {idx}: domain={domain_of_url(r.get('link',''))} title={r.get('title','')[:80]} snippet={r.get('snippet','')[:120]}")

                # Retourner ces résultats même s'ils ne proviennent pas de la whitelist
                # pour que la vérification heuristique puisse en tirer des indices.
                return secondary_results

        return trusted_results
    
    def verify_with_gemini(self, rumor_text: str, trusted_sources: List[Dict]) -> Dict:
        """Vérifie la rumeur avec Gemini"""
        if not GEMINI_AVAILABLE or not self.api_key:
            return self._fallback_verification(rumor_text, trusted_sources)
        
        evidence = "\n".join([
            f"- [{s.get('displayLink')}] {s.get('title')} : {s.get('snippet')}"
            for s in trusted_sources[:5]
        ])
        
        prompt = f"""Tu es un fact-checker expert au Bénin.

        RUMEUR À VÉRIFIER:
        \"{rumor_text}\"

        SOURCES FIABLES TROUVÉES:
        {evidence if evidence else "(Aucune source fiable trouvée)"}

        INSTRUCTIONS:
        - Si les sources montrent clairement que la rumeur est vraie → verdict: VRAI
        - Si les sources montrent clairement que la rumeur est fausse → verdict: FAUX
        - Si les sources ne contiennent pas assez d'information pour trancher, tu **dois utiliser tes connaissances récentes sur l'actualité politique au Bénin** pour décider si la rumeur est vraisemblable ou fausse.

        - Fournis un score de véracité de 0.0 à 1.0 (0 = totalement faux, 1 = totalement vrai)
        - Fournis les sources utilisées (même partielles) et une explication concise
        - Réponds uniquement en JSON

        EXEMPLE DE RÉPONSE JSON:
        {{
        "verdict": "VRAI",
        "score_veracite": 0.8,
        "explication": "La majorité des sources fiables confirment les faits, ou mes connaissances sur l'actualité récente corroborent la rumeur.",
        "sources_utilisees": ["url1", "url2"],
        "recommandation": "Vérification humaine recommandée"
        }}
        """

        
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            text = response.text.strip()
            
            if text.startswith("```json"):
                text = text.replace("```json", "").replace("```", "").strip()
            
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
                result["date_verification"] = datetime.now(timezone.utc).isoformat()
                result["nb_sources_fiables"] = len(trusted_sources)
                return result
            else:
                raise ValueError("No JSON in response")
        
        except Exception as e:
            logging.error(f"Gemini verification error: {e}")
            return self._fallback_verification(rumor_text, trusted_sources)
    
    def _fallback_verification(self, rumor_text: str, sources: List[Dict]) -> Dict:
        """Vérification simple sans Gemini"""
        if not sources:
            return {
                "verdict": "INCERTAIN",
                "score_veracite": 0.0,
                "explication": "Aucune source fiable trouvée.",
                "sources_utilisees": [],
                "recommandation": "Enquête manuelle nécessaire",
                "nb_sources_fiables": 0
            }
        
        combined_text = " ".join([s.get("snippet", "") + " " + s.get("title", "") for s in sources]).lower()

        # Mots-clés plus larges pour détecter confirmation/démenti
        negative_indicators = ["faux", "démenti", "refute", "réfute", "infirme"]
        positive_indicators = ["confirmé", "officiel", "communiqué", "mise au point", "confirmes"]

        if any(term in combined_text for term in negative_indicators):
            verdict = "FAUX"
            score = 0.2
        elif any(term in combined_text for term in positive_indicators):
            verdict = "VRAI"
            score = 0.8
        else:
            verdict = "INCERTAIN"
            score = 0.5
        
        return {
            "verdict": verdict,
            "score_veracite": score,
            "explication": f"Analyse basée sur {len(sources)} source(s) fiable(s).",
            "sources_utilisees": [s.get("link") for s in sources],
            "recommandation": "Vérification humaine recommandée",
            "nb_sources_fiables": len(sources)
        }


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
        self.fact_checker = FactChecker(Config.GEMINI_API_KEY, self.web_searcher)
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




