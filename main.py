import re
import urllib.parse
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import pickle
import json
import os

# ML imports
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# UI
import gradio as gr

# Optional: Gemini AI summarization 
try:
    import google.generativeai as genai
except Exception:
    genai = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CyberHunter")

# ==========================
# API KEYS (set via env vars or config)
# ==========================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ABUSEIPDB_KEY = os.getenv("ABUSEIPDB_KEY")
VT_API_KEY = os.getenv("VT_API_KEY")
OTX_API_KEY = os.getenv("OTX_API_KEY")
IPINFO_API_KEY = os.getenv("IPINFO_API_KEY")

if GEMINI_API_KEY and genai:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        logger.warning("Could not configure generative API (genai). Check the library and key.")

# ==========================
# Detection Rules (regex)
# ==========================
rules = {
    "Brute Force Login": r"(POST\s+/login.*\s401|Invalid password|Authentication failed)",
    "SQL Injection": r"(?i)(\bUNION\b.*\bSELECT\b|\bDROP\b.*\bTABLE\b|OR\s+['\"]?1['\"]?=.*)",
    "XSS": r"(?i)(onerror\s*=|onload\s*=|javascript:|alert\s*\()",
    "File Inclusion": r"(\.\./|\.\.\\|/etc/passwd|/etc/shadow|boot\.ini|c:\\windows)",
    "Command Injection": r"(?ix)([;&|]{1,2}|`|\$\(|\)\s*;|\b(cat|wget|curl|bash|python|sh)\b)",
    "Web Shell": r"(?i)(c99shell|r57shell|webshell|eval\(|assert\(|exec\()",
    "Path Traversal": r"(\.\./|\.\.\\|/etc/|/proc/|boot\.ini|/var/log/|/root/)",
    "SSRF": r"(http://169\.254\.169\.254|http://metadata\.google|http://localhost|http://127\.0\.0\.1)",
    "DoS / Flood": r"(Slowloris|ApacheBench|masscan|nmap|udp\s+flood|syn\s+flood|hping3)",
    "Auth Bypass": r"(?i)(login.*success|jwt\s*=.*|Authorization:\s*Bearer\s+)",
    "Admin Access Attempt": r"(/admin\b|/administrator\b|/wp-admin|/phpmyadmin|/console|/dashboard)",
    "Successful Login": r"(POST\s+/login.*\s200|Login successful|Authenticated user)"
}
compiled_rules = {k: re.compile(v, re.IGNORECASE) for k, v in rules.items()}

# ==========================
# Threat Intelligence helpers
# ==========================
ip_cache = {}

def is_private_ip(ip: str) -> bool:
    if not ip:
        return True

    private_prefixes = (
        "10.",
        "192.168.",
        "127.",
        "172.16.", "172.17.", "172.18.", "172.19.",
        "172.20.", "172.21.", "172.22.", "172.23.",
        "172.24.", "172.25.", "172.26.", "172.27.",
        "172.28.", "172.29.", "172.30.", "172.31."
    )
    return ip.startswith(private_prefixes)


def enrich_ip_single(ip: str) -> (str, dict):
    result = {
    "country": None,
    "city": None,
    "region": None,
    "org": None,
    "latitude": None,
    "longitude": None,
    "abuseScore": None,
    "otx_pulses": None,
    "vt_rep": None
    }
    if not ip or is_private_ip(ip):
        return ip, result
    if ip in ip_cache:
        return ip, ip_cache[ip]
    
    # AbuseIPDB
    if ABUSEIPDB_KEY:
        try:
            r = requests.get(f"https://api.abuseipdb.com/api/v2/check?ipAddress={ip}",
                             headers={"Key": ABUSEIPDB_KEY, "Accept": "application/json"}, timeout=4)
            if r.ok:
                data = r.json().get("data", {})
                result["abuseScore"] = data.get("abuseConfidenceScore")
        except Exception:
            pass
    # OTX
    if OTX_API_KEY:
        try:
            r = requests.get(f"https://otx.alienvault.com/api/v1/indicators/IPv4/{ip}/general",
                             headers={"X-OTX-API-KEY": OTX_API_KEY}, timeout=4)
            if r.ok:
                result["otx_pulses"] = len(r.json().get("pulse_info", {}).get("pulses", []))
        except Exception:
            pass
    # VirusTotal
    if VT_API_KEY:
        try:
            r = requests.get(f"https://www.virustotal.com/api/v3/ip_addresses/{ip}",
                             headers={"x-apikey": VT_API_KEY}, timeout=4)
            if r.ok:
                result["vt_rep"] = r.json().get("data", {}).get("attributes", {}).get("reputation")
        except Exception:
            pass

    # 🌍 GeoIP (IPinfo)
    if IPINFO_API_KEY:

        try:
            r = requests.get(
                f"https://ipinfo.io/{ip}/json",
                headers={"Authorization": f"Bearer {IPINFO_API_KEY}"},
                timeout=4
            )
            if r.ok:
                data = r.json()
                result["country"] = data.get("country")
                result["city"] = data.get("city")
                result["region"] = data.get("region")
                result["org"] = data.get("org")

                loc = data.get("loc")  # "lat,lon"
                if loc and "," in loc:
                    lat, lon = loc.split(",")
                    result["latitude"] = float(lat)
                    result["longitude"] = float(lon)
        except Exception:
            pass



    ip_cache[ip] = result
    return ip, result

def enrich_ips_parallel(ip_list):
    unique_ips = list(set(ip_list))
    results = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ip = {executor.submit(enrich_ip_single, ip): ip for ip in unique_ips}
        for future in as_completed(future_to_ip):
            try:
                ip, data = future.result()
                results[ip] = data
            except Exception:
                ip = future_to_ip[future]
                results[ip] = {}
    return results

# ==========================
# Feature extraction & lightweight embedding
# ==========================
class FeatureBuilder:
    def __init__(self, svd_dim=50, max_tfidf_features=2000):
        self.svd_dim = svd_dim
        self.max_tfidf_features = max_tfidf_features
        self.tfidf = None
        self.svd = None
        self.scaler = None

    def fit_transform(self, parsed_records):
        texts = [r["msg"] for r in parsed_records]
        self.tfidf = TfidfVectorizer(max_features=self.max_tfidf_features, ngram_range=(1,2))
        X_tfidf = self.tfidf.fit_transform(texts)
        n_comp = min(self.svd_dim, max(1, X_tfidf.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n_comp)
        X_emb = self.svd.fit_transform(X_tfidf)
        X_num = self._numeric_matrix(parsed_records)
        X = np.hstack([X_emb, X_num])
        self.scaler = StandardScaler().fit(X)
        return self.scaler.transform(X)

    def transform(self, parsed_records):
        if self.tfidf is None or self.svd is None or self.scaler is None:
            raise RuntimeError("FeatureBuilder not trained")
        texts = [r["msg"] for r in parsed_records]
        X_tfidf = self.tfidf.transform(texts)
        X_emb = self.svd.transform(X_tfidf)
        X_num = self._numeric_matrix(parsed_records)
        X = np.hstack([X_emb, X_num])
        return self.scaler.transform(X)

    def _numeric_matrix(self, parsed_records):
        rows = []
        for r in parsed_records:
            token_count = len(r["msg"].split())
            status = r.get("status") or 0
            has_ip = 1 if r.get("ip") else 0
            bytes_ = r.get("bytes") or 0
            rows.append([token_count, status, has_ip, bytes_])
        return np.array(rows, dtype=float)

# ==========================
# Hybrid ML detectors (IsolationForest + LOF + PCA)
# ==========================
class HybridDetectors:
    def __init__(self):
        self.iso = None
        self.lof = None
        self.pca = None

    def fit(self, X, contamination=0.01):
        logger.info("Fitting IsolationForest, LOF, PCA")
        self.iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
        self.iso.fit(X)
        self.lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
        self.lof.fit(X)
        n_comp = min(max(1, X.shape[1] // 2), 50)
        self.pca = PCA(n_components=n_comp, random_state=42)
        self.pca.fit(X)

    def score(self, X):
        iso_s = -self.iso.score_samples(X)  # higher = more anomalous
        lof_s = -self.lof.score_samples(X)
        X_rec = self.pca.inverse_transform(self.pca.transform(X))
        pca_s = np.mean((X - X_rec)**2, axis=1)
        return {"iso": iso_s, "lof": lof_s, "pca": pca_s}

def minmax_scale(arr):
    a = np.array(arr, dtype=float)
    mn = a.min(); mx = a.max()
    if abs(mx - mn) < 1e-12:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)

# ==========================
# Combined engine (fit/analyze/save/load)
# ==========================
class CyberHunterEngine:
    def __init__(self):
        self.fb = FeatureBuilder()
        self.det = HybridDetectors()
        self.trained = False
        self.train_combined = None
        self.quantile = 0.95

    def fit(self, parsed_records, contamination=0.01):
        X = self.fb.fit_transform(parsed_records)
        self.det.fit(X, contamination=contamination)
        scores = self.det.score(X)
        iso_n = minmax_scale(scores["iso"]); lof_n = minmax_scale(scores["lof"]); pca_n = minmax_scale(scores["pca"])
        combined = (iso_n + lof_n + 2.0 * pca_n) / 4.0
        self.train_combined = combined
        self.trained = True
        logger.info("Engine trained (hybrid detectors).")

    def analyze(self, parsed_records, fusion_weights=None, use_apis=True, quantile=None):
        if not self.trained:
            raise RuntimeError("Engine not trained. Call fit() first.")
        if fusion_weights is None:
            fusion_weights = {"iso": 1.0, "lof": 1.0, "pca": 2.0}
        X = self.fb.transform(parsed_records)
        scores = self.det.score(X)
        iso_n = minmax_scale(scores["iso"]); lof_n = minmax_scale(scores["lof"]); pca_n = minmax_scale(scores["pca"])
        w_iso = fusion_weights.get("iso", 1.0); w_lof = fusion_weights.get("lof", 1.0); w_pca = fusion_weights.get("pca", 2.0)
        denom = (w_iso + w_lof + w_pca) if (w_iso + w_lof + w_pca) != 0 else 1.0
        combined = (w_iso * iso_n + w_lof * lof_n + w_pca * pca_n) / denom
        q = quantile if quantile is not None else self.quantile
        base_thr = float(np.quantile(self.train_combined if self.train_combined is not None else combined, q))
        # regex hits
        regex_hits = []
        for r in parsed_records:
            matched = [name for name, cre in compiled_rules.items() if cre.search(r["raw"])]
            regex_hits.append("; ".join(matched) if matched else None)
        rows = []
        ips = []
        for i, r in enumerate(parsed_records):
            row = {
                "timestamp": datetime.utcnow().isoformat(),
                "line_no": r.get("line_no"),
                "ip": r.get("ip"),
                "log": r.get("raw"),
                "regex_detected": regex_hits[i],
                "iso_score": float(iso_n[i]),
                "lof_score": float(lof_n[i]),
                "pca_score": float(pca_n[i]),
                "final_score": float(combined[i]),
                "threshold": base_thr
            }
            rows.append(row)
            if r.get("ip"):
                ips.append(r.get("ip"))
        df = pd.DataFrame(rows)
        if use_apis and ips:
            info = enrich_ips_parallel(ips)
            df["country"] = df["ip"].apply(lambda ip: info.get(ip, {}).get("country") if ip else None)
            df["abuseScore"] = df["ip"].apply(lambda ip: info.get(ip, {}).get("abuseScore") if ip else None)
            df["otx_pulses"] = df["ip"].apply(lambda ip: info.get(ip, {}).get("otx_pulses") if ip else None)
            df["vt_rep"] = df["ip"].apply(lambda ip: info.get(ip, {}).get("vt_rep") if ip else None)
                # GeoIP coordinates & ISP info
            df["city"] = df["ip"].apply(lambda ip: info.get(ip, {}).get("city") if ip else None)
            df["region"] = df["ip"].apply(lambda ip: info.get(ip, {}).get("region") if ip else None)
            df["org"] = df["ip"].apply(lambda ip: info.get(ip, {}).get("org") if ip else None)
            df["latitude"] = df["ip"].apply(lambda ip: info.get(ip, {}).get("latitude") if ip else None)
            df["longitude"] = df["ip"].apply(lambda ip: info.get(ip, {}).get("longitude") if ip else None)

        df["severity"] = df["regex_detected"].apply(lambda x: "High" if x and any(k in x for k in ["SQL Injection","Web Shell","Command Injection","File Inclusion","DoS / Flood"]) else ("Medium" if x else "Low"))
        df["anomaly"] = (df["final_score"] > df["threshold"]).astype(int)
        # append regex-detected rows that may not be captured by anomaly scores (to preserve original logic)
        # original code appended regex findings earlier; here we already include regex_detected in rows
        return df.drop_duplicates().reset_index(drop=True)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "engine_fb.pkl"), "wb") as f:
            pickle.dump({"svd_dim": self.fb.svd_dim, "max_tfidf_features": self.fb.max_tfidf_features,
                         "tfidf": self.fb.tfidf, "svd": self.fb.svd, "scaler": self.fb.scaler}, f)
        with open(os.path.join(path, "engine_det.pkl"), "wb") as f:
            pickle.dump({"iso": self.det.iso, "lof": self.det.lof, "pca": self.det.pca}, f)
        with open(os.path.join(path, "engine_meta.json"), "w") as f:
            json.dump({"train_combined": self.train_combined.tolist() if self.train_combined is not None else None}, f)
        logger.info("Engine saved to %s", path)

    def load(self, path):
        with open(os.path.join(path, "engine_fb.pkl"), "rb") as f:
            fbobj = pickle.load(f)
        self.fb = FeatureBuilder(svd_dim=fbobj.get("svd_dim", 50), max_tfidf_features=fbobj.get("max_tfidf_features", 2000))
        self.fb.tfidf = fbobj["tfidf"]; self.fb.svd = fbobj["svd"]; self.fb.scaler = fbobj["scaler"]
        with open(os.path.join(path, "engine_det.pkl"), "rb") as f:
            detobj = pickle.load(f)
        self.det = HybridDetectors()
        self.det.iso = detobj["iso"]; self.det.lof = detobj["lof"]; self.det.pca = detobj["pca"]
        with open(os.path.join(path, "engine_meta.json"), "r") as f:
            meta = json.load(f)
        if meta.get("train_combined") is not None:
            self.train_combined = np.array(meta["train_combined"])
        self.trained = True
        logger.info("Engine loaded from %s", path)

# ==========================
# Helper: parse raw log text into records
# ==========================
def parse_lines_to_records(log_text):
    if not log_text or not log_text.strip():
        return []
    lines = [urllib.parse.unquote_plus(l) for l in log_text.strip().split("\n")]
    parsed = []
    for i, l in enumerate(lines):
        rec = {"raw": l, "line_no": i + 1, "ip": None, "status": None, "bytes": None, "msg": l}
        ip_match = re.match(r"(\d+\.\d+\.\d+\.\d+)", l)
        if ip_match:
            rec["ip"] = ip_match.group(1)
        st_match = re.search(r"\s(\d{3})\s", l)
        if st_match:
            try:
                rec["status"] = int(st_match.group(1))
            except Exception:
                rec["status"] = None
        bs_match = re.search(r"\s(\d{1,9})\s*$", l)
        if bs_match:
            try:
                rec["bytes"] = int(bs_match.group(1))
            except Exception:
                rec["bytes"] = None
        parsed.append(rec)
    return parsed

# ==========================
# Summarizer (Gemini) stub
# ==========================
def summarize_incidents(df, use_apis=True):
    if df is None or df.empty:
        return "No incidents detected."
    if not use_apis or not GEMINI_API_KEY or genai is None:
        # fallback simple summary
        total = len(df)
        n_anom = int(df["anomaly"].sum())
        top_rules = df["regex_detected"].dropna()
        s = [f"Total lines: {total}. Anomalies flagged: {n_anom}."]
        if not top_rules.empty:
            s.append(f"Example rule hits: {', '.join(list(top_rules.unique()[:5]))}")
        return " ".join(s)
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        text = df.to_string(index=False)
        prompt = f"Explain the following security incidents and suggest remediations:\n\n{text}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Summarization failed: {e}"
    
def save_summary_to_file(summary_text):
    if not summary_text or not summary_text.strip():
        return " No summary to save."

    # Windows Downloads folder
    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"cyber_hunter_report_{timestamp}.txt"
    filepath = os.path.join(downloads_path, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(summary_text)

    return f" Report downloaded to: {filepath}"


# ==========================
# Gradio UI (keeps same flow)
# ==========================
engine = CyberHunterEngine()

def process_logs(log_text, use_threat_intel, use_summarization):
    try:
        parsed = parse_lines_to_records(log_text)

        # If engine not trained, do a quick-train on incoming logs (fallback)
        if not engine.trained:
            logger.info("Engine not trained: performing quick-fit on provided logs (treating all lines as normal)")
            engine.fit(parsed, contamination=0.01)

        df = engine.analyze(parsed, fusion_weights=None, use_apis=bool(use_threat_intel))

        #  Filter out logs that have no regex detection and are not anomalous
        if not df.empty:
            df = df[(df['anomaly'] == 1) | (df['regex_detected'].notnull())]

            #  Sort so regex detections appear first, followed by AI-only anomalies
            df['sort_order'] = df['regex_detected'].notnull().astype(int) * -1
            df = df.sort_values(by='sort_order', ascending=False).drop(columns=['sort_order'])

        summary = summarize_incidents(df, use_apis=bool(use_summarization))

        # Automatically save summary to Downloads
        save_path = save_summary_to_file(summary)

        # Append saved file path to display in Gradio
        summary_with_path = f"{summary}\n\n✅ Report saved to: {save_path}"

        return df, summary_with_path


    except Exception as e:
        logger.exception("Error during process_logs")
        return pd.DataFrame([]), f"Error: {e}"
    
def send_summary_email(summary_text):
    return save_summary_to_file(summary_text)



with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## Cyber-Hunter ")
    log_input = gr.Textbox(label="Paste Logs Here", lines=15, placeholder="Paste web server logs...")
    use_threat_intel_toggle = gr.Checkbox(label="Enable Threat Intelligence APIs (ip-api/AbuseIPDB/OTX/VT)", value=False)
    use_summarization_toggle = gr.Checkbox(label="Enable Gemini Summarization API", value=False)
    analyze_btn = gr.Button("Analyze Logs")
    

    output_table = gr.Dataframe(label="Detection Results", interactive=False)
    output_summary = gr.Textbox(label="Incident Summary", lines=6)
    

    analyze_btn.click(process_logs, inputs=[log_input, use_threat_intel_toggle, use_summarization_toggle], outputs=[output_table, output_summary])
    

if __name__ == "__main__":
    demo.launch(share=True, inbrowser=True)  



# launch
