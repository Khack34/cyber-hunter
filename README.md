# Cyber-Hunter
### AI-Powered Agents for Cyber Log Data Investigation

Cyber-Hunter is an academic AI/ML and cybersecurity engineering project designed to investigate server log data, identify suspicious activities, detect potential security threats, and support incident reporting through automated analysis.

The project combines rule-based detection, machine learning anomaly detection, threat intelligence integrations, and AI-assisted report generation.

## Key Features

- **Regex-Based Threat Detection:** Identifies suspicious patterns associated with SQL injection, XSS, brute-force attempts, path traversal, command injection, and other web attacks.
- **Isolation Forest:** Detects unusual log activity using unsupervised machine learning.
- **Local Outlier Factor (LOF):** Identifies anomalous log observations.
- **Principal Component Analysis (PCA):** Supports dimensionality reduction and anomaly analysis.
- **Threat Intelligence Integration:** Enriches suspicious IP addresses using AbuseIPDB, AlienVault OTX, VirusTotal, and IPinfo.
- **AI-Assisted Reporting:** Uses Gemini to generate readable investigation summaries.


## Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| NLP / Feature Representation | TF-IDF, Truncated SVD |
| Threat Intelligence | AbuseIPDB, AlienVault OTX, VirusTotal, IPinfo |
| Generative AI | Google Gemini |
| Interface / Demonstration | Gradio |
| Reporting | 

## Detection Components

1. Regex-based threat detection
2. Isolation Forest
3. Local Outlier Factor
4. Principal Component Analysis
5. Threat intelligence enrichment
6. AI-assisted investigation reporting

## Project Workflow

1. Collect or provide server access logs.
2. Analyze log entries using rule-based detection.
3. Apply machine learning techniques to identify anomalous activity.
4. Enrich suspicious IP addresses using threat intelligence services.
5. Generate investigation summaries.

## Installation

Clone the repository:

```bash
git clone https://github.com/Khack34/cyber-hunter.git
cd cyber-hunter
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure the required API credentials using environment variables. Refer to `.env.example`.

## Configuration

The project integrates with external threat intelligence and AI services. API access and credentials may be required for their respective features.

Never commit API keys, passwords, or private log data.

## Disclaimer

Cyber-Hunter is intended for educational, research, and authorized defensive cybersecurity use. Test it only with logs and systems you own or have permission to analyze.

## Author

**Khack34**

GitHub: https://github.com/Khack34
