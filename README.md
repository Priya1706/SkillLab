# SkillLab

A collection of projects, practical implementations, and learning experiments developed as part of my technical skill training and independent development journey.

This repository brings together applications built using Python, machine learning, web development, and AI integration.

## Projects

### 1. SentinelAI – AI-Driven Anomaly Detection

An intelligent screening system designed to detect anomalous electronic components and predict performance drift during burn-in testing.

**Key Features**
- Dynamic anomaly detection using Isolation Forest.
- Time-series drift prediction using Random Forest Regression.
- Component-level risk classification.
- Explainable AI for inspection insights.
- Interactive Flask dashboard.
- CSV data upload and PDF report generation.

**Technologies:** Python, Flask, Scikit-learn, Pandas, NumPy, Plotly.

**Location:** `SentinelAI/`

### 2. Personal AI Chatbot

A web-based conversational chatbot built using Streamlit and the OpenRouter API.

**Key Features**
- Interactive chat interface.
- Configurable AI model selection.
- Custom system prompts.
- Adjustable temperature and token limits.
- Conversation management.

**Technologies:** Python, Streamlit, OpenAI Python SDK, OpenRouter API.

**Location:** `chatbot/`

**Run locally:**

```bash
pip install -r chatbot/requirements.txt
streamlit run chatbot/app.py
```

Configure the OpenRouter API key using Streamlit secrets before running the application.

### 3. Personal Portfolio Website

A personal portfolio website developed using Python and Flask to showcase projects, technical skills, and academic work.

**Technologies:** Python, Flask, HTML, CSS, JavaScript.

**Location:** `priyamvadhaa_portfolio/`

**Run locally:**

```bash
cd priyamvadhaa_portfolio
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## Learning and Training

### Skilium Certificate

Certificate obtained through skill training.

**File:** `certificate/Skilium-Certificate.pdf`

### Python Practice Notebook

A Jupyter Notebook containing Python learning exercises and practical implementations.

**File:** `notebook/skilllab.ipynb`

---

## Technologies and Tools

| Category | Technologies |
|---|---|
| Programming Languages | Python |
| Web Development | Flask, Streamlit, HTML, CSS, JavaScript |
| Machine Learning | Scikit-learn, Random Forest, Isolation Forest |
| Data Processing | Pandas, NumPy |
| AI Integration | OpenRouter API, OpenAI Python SDK |
| Visualization | Plotly |
| Development Tools | VS Code, Jupyter Notebook, Git, GitHub |

## Repository Structure

```text
SkillLab/
│
├── SentinelAI/
│   ├── data/
│   ├── models/
│   ├── static/
│   ├── templates/
│   ├── utils/
│   ├── app.py
│   ├── README.md
│   └── requirements.txt
│
├── chatbot/
│   ├── app.py
│   └── requirements.txt
│
├── priyamvadhaa_portfolio/
│   ├── static/
│   ├── templates/
│   ├── app.py
│   ├── Procfile
│   ├── README.md
│   └── requirements.txt
│
├── certificate/
│   └── Skilium-Certificate.pdf
│
├── notebook/
│   └── skilllab.ipynb
│
├── .gitignore
└── README.md
```

## Getting Started

Clone the repository:

```bash
git clone https://github.com/Priya1706/SkillLab.git
cd SkillLab
```

Each project has its own dependencies and setup instructions. Refer to the respective project folders for details.

## Security

- API keys and credentials are stored locally using environment variables or Streamlit secrets.
- Secret configuration files are excluded through `.gitignore`.
- Never commit API keys, passwords, or other sensitive credentials.

## Author

**Priyamvadhaa Hoskere**

- GitHub: [Priya1706](https://github.com/Priya1706)

---

This repository is continuously updated as new projects and learning experiences are added.
