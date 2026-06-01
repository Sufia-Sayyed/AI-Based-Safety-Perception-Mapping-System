# AI-Based Safety Perception Mapping System for Women

## Overview
The AI-Based Safety Perception Mapping System for Women is a web-based application that uses Artificial Intelligence (AI) and Natural Language Processing (NLP) to analyze crowdsourced safety experiences and generate location-based safety insights.

Unlike traditional women safety applications that focus on emergency response mechanisms such as SOS alerts and GPS tracking, this system emphasizes preventive awareness by helping users understand safety perceptions before entering a location.


## Features
- Anonymous safety feedback submission
- NLP-based text preprocessing
- TF-IDF feature extraction
- Logistic Regression sentiment classification
- Safety categorization:
  - Safe (Positive Sentiment)
  - Moderate (Neutral Sentiment)
  - Unsafe (Negative Sentiment)
- Interactive safety perception map
- Community-driven safety intelligence
- Privacy-friendly approach without user tracking
- 

## System Architecture
1. User submits safety feedback and location.
2. Text data is preprocessed:
   - Text cleaning
   - Lowercasing
   - Stopword removal
   - Tokenization
3. TF-IDF Vectorizer converts text into numerical features.
4. Logistic Regression model predicts sentiment.
5. Sentiment is mapped to safety levels.
6. Results are visualized on an interactive map.


## Technologies Used
### Frontend
- HTML
- CSS
- JavaScript
### Backend
- Python
- Flask
### Machine Learning
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
### Data Processing
- Pandas
- NumPy
- NLTK


## Project Structure

AI-Based-Safety-Perception-Mapping-System/
│
├── app.py
├── dataset.csv
├── sentiment_model.pkl
├── vectorizer.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   │
│   └── js/
│       └── script.js
│
└── README.md


## Dataset
The system was trained on a dataset containing **1,111 safety feedback entries** collected from **18 different locations**.
Each entry contains:
- Location information
- User safety experience
- Sentiment label


## Model Performance

| Metric | Score |
|----------|--------|
| Accuracy | 87% |
| Precision | 83% |
| Recall | 82% |
| F1-Score | 82% |


## Installation

### Clone the Repository

```bash
git clone https://github.com/Sufia-Sayyed/AI-Based-Safety-Perception-Mapping-System.git

cd AI-Based-Safety-Perception-Mapping-System


### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment
Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install flask pandas numpy scikit-learn nltk
```

---

## Run the Application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## Working Example

### Input Feedback

```text
Poor lighting and harassment issues make it unsafe to go alone.
```

### Prediction

```text
Sentiment: Negative
Safety Level: Unsafe
```



## Research Contribution
This project proposes a preventive safety awareness framework that leverages AI and crowdsourced experiences to generate safety perception maps. The system transforms unstructured textual feedback into actionable safety insights, helping users make informed decisions while promoting community-driven urban safety.


## License
This project is developed for academic and research purposes.

