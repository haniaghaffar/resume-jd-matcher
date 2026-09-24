# ResumeAI — Resume & Job Description Matcher

ResumeAI is an AI-powered resume analysis and job matching application that evaluates how well a resume aligns with a given job description.

The project combines **NLP, TF-IDF feature extraction, Logistic Regression, rule-based resume analysis, and Streamlit** to provide a practical resume screening experience.

Instead of only returning a match prediction, ResumeAI also analyzes the resume itself and highlights areas such as skills, sections, contact information, action verbs, measurable achievements, and potential content issues.

---

## Overview

A resume can be technically strong but still perform poorly against a specific job description if important skills, keywords, sections, or evidence are missing.

ResumeAI addresses this by providing two related capabilities:

### Resume Analysis

The application analyzes the uploaded resume and generates a structured profile containing:

* Overall resume score
* ATS-oriented score
* Resume section coverage
* Contact information
* Technical and soft skills
* Action verb usage
* Quantifiable achievements
* Resume strengths
* Potential issues
* Improvement priorities
* Executive summary

### Job Match

The application compares a resume with a job description and produces:

* Match signal
* Matching skills
* Missing skills
* Job-related skill coverage
* Model-based prediction
* Match probability when available

The goal is to turn a resume from a static document into something that can be evaluated against an actual job requirement.

---

# Machine Learning Pipeline

The core matching system follows a simple NLP classification pipeline.

```text
Resume + Job Description
          ↓
     Text Extraction
          ↓
    Text Preprocessing
          ↓
   TF-IDF Vectorization
          ↓
 Logistic Regression Model
          ↓
   Match / Mismatch
          ↓
 Match Probability
```

The Streamlit application then combines the model output with additional resume-analysis logic to provide a more useful result.

---

# Dataset

The model was trained using the Hugging Face dataset:

`vadimas22/resume-score-details`

The dataset contains resume and job-description examples with different evaluation categories.

The processed dataset used for this project contained:

| Category                |    Count |
| ----------------------- | -------: |
| Match                   |      648 |
| Mismatch                |      201 |
| Invalid Resume          |       55 |
| Invalid Job Description |       57 |
| Other                   |       70 |
| **Total**               | **1031** |

The training workflow separates valid matching examples from invalid or unusable records before building the classification dataset.

---

# Data Processing

The raw dataset consists of JSON files containing resume/job-related information.

The training workflow first identifies the relevant text fields and filters the records based on their categories.

The general preprocessing flow is:

```text
Raw JSON Dataset
      ↓
Load Records
      ↓
Identify Text Fields
      ↓
Validate Resume / Job Description
      ↓
Select Match & Mismatch Records
      ↓
Clean Text
      ↓
Create Training Dataset
```

Text preprocessing focuses on converting the available resume and job-description content into a format suitable for classical NLP models.

---

# Feature Extraction

Resume and job-description text cannot be directly passed into Logistic Regression.

The project uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert text into numerical features.

TF-IDF gives higher importance to words that are useful for distinguishing documents while reducing the influence of very common words.

The basic idea is:

```text
Text
 ↓
Tokenization
 ↓
Term Frequency
 ↓
Inverse Document Frequency
 ↓
TF-IDF Vector
```

This creates a numerical representation of the combined resume and job-description text.

---

# Machine Learning Model

The classification model used in the project is **Logistic Regression**.

The model receives TF-IDF features and learns to classify resume/job-description pairs into matching categories.

```text
TF-IDF Features
       ↓
Logistic Regression
       ↓
Match / Mismatch
```

Logistic Regression was selected because it is lightweight, interpretable, fast to train, and works well as a baseline model for high-dimensional sparse text features.

---

# Model Training

The training process can be summarized as:

```text
1. Load dataset
2. Read JSON records
3. Extract relevant resume/job-description text
4. Filter valid records
5. Create match/mismatch labels
6. Combine resume and job-description text
7. Fit TF-IDF Vectorizer
8. Transform text into numerical features
9. Train Logistic Regression
10. Save trained artifacts
```

After training, the two most important artifacts are exported:

```text
resume_jd_model.pkl
resume_jd_vectorizer.pkl
```

These files allow the Streamlit application to perform predictions without retraining the model every time the application starts.

---

# Saved Model Artifacts

### `resume_jd_model.pkl`

Contains the trained Logistic Regression classifier.

### `resume_jd_vectorizer.pkl`

Contains the fitted TF-IDF vectorizer used during training.

Both artifacts are loaded by the Streamlit application and must remain compatible with the preprocessing and model-training environment.

---

# Resume Analysis Engine

The project does more than simply run the machine learning model.

A separate rule-based analysis layer evaluates the uploaded resume across multiple dimensions.

## Resume Sections

The application checks for common resume sections such as:

* Summary / Profile
* Experience
* Education
* Skills
* Projects
* Certifications
* Achievements
* Volunteer / Activities

Section coverage contributes to the overall resume assessment.

## Contact Information

The application checks for commonly expected contact details, including:

* Email
* Phone
* LinkedIn
* GitHub
* Location

## Skills

The analysis engine maintains technical and soft-skill vocabularies and detects relevant skills from the resume.

It also supports skill aliases so that variations of common technologies can be recognized.

## Action Verbs

The application looks for strong resume action verbs such as:

* Developed
* Built
* Designed
* Implemented
* Created
* Integrated
* Optimized
* Automated
* Managed
* Led

This helps identify whether experience and project descriptions communicate actual contributions.

## Quantifiable Evidence

The application also checks for measurable evidence such as:

* Percentages
* Numbers
* Years
* Counts
* Other numerical achievements

For example:

```text
Improved response time by 35%
```

provides stronger measurable evidence than a generic statement without a result.

---

# Scoring System

ResumeAI combines multiple signals instead of relying on a single metric.

The main scoring components include:

```text
ATS Score
├── Resume Sections
├── Contact Information
├── Skills
└── Evidence

Overall Score
├── ATS Score
├── Content Quality
├── Skills
└── Evidence
```

The scoring layer is designed as a practical resume-analysis heuristic rather than a replacement for a professional recruiter or ATS platform.

---

# Job Matching Pipeline

When a user uploads a resume and enters a job description, the application follows this flow:

```text
Resume PDF / TXT
       +
Job Description
       ↓
Text Extraction
       ↓
Skill Extraction
       ↓
Skill Comparison
       ↓
TF-IDF Transformation
       ↓
Logistic Regression
       ↓
Match Signal
       ↓
Detailed Result
```

The application also identifies skills mentioned in the job description and compares them with the skills detected in the resume.

This makes it possible to distinguish between:

```text
Matched Skills
```

and

```text
Skill Gaps
```

---

# Application Features

## Resume Analyzer

Upload a resume and receive a structured analysis covering:

* Overall score
* ATS-oriented score
* Content score
* Skills
* Sections
* Contact information
* Action verbs
* Evidence
* Strengths
* Issues
* Improvement priorities
* Executive summary

## Job Match

Provide:

* Resume
* Job description

The application returns:

* Match signal
* Matching skills
* Missing skills
* Model prediction
* Match probability

## Supported Resume Formats

Currently supported:

* PDF
* TXT

PDF text is extracted using `pypdf`.

The application also handles PDF reading errors gracefully instead of failing silently.

---

# Tech Stack

### Programming

* Python

### Machine Learning

* scikit-learn
* Logistic Regression
* TF-IDF

### Data Processing

* pandas
* NumPy

### Application

* Streamlit

### File Processing

* pypdf

### Model Persistence

* joblib

### Development

* Jupyter Notebook
* Anaconda / Conda
* Git
* GitHub

---

# Project Structure

```text
resume-jd-matcher/
│
├── app.py
├── ResumeAnalyzer.ipynb
│
├── resume_jd_model.pkl
├── resume_jd_vectorizer.pkl
│
├── requirements.txt
│
└── README.md
```

### `app.py`

Main Streamlit application containing the UI, resume analysis logic, job matching workflow, scoring system, and model inference.

### `ResumeAnalyzer.ipynb`

Training and data-analysis notebook containing the model development workflow.

### `resume_jd_model.pkl`

Trained Logistic Regression model.

### `resume_jd_vectorizer.pkl`

Trained TF-IDF vectorizer.

### `requirements.txt`

Python dependencies required to run the application.

---

# Local Setup

## 1. Clone the repository

```text
git clone https://github.com/haniaghaffar/resume-jd-matcher.git
```

Move into the project directory:

```text
cd resume-jd-matcher
```

---

## 2. Create or activate a Conda environment

The project was developed using Anaconda/Conda.

Example:

```text
conda create -n resumeai python=3.11
```

Activate it:

```text
conda activate resumeai
```

If an existing compatible Conda environment is already available, it can be used instead.

---

## 3. Install dependencies

```text
pip install -r requirements.txt
```

---

## 4. Run the application

```text
streamlit run app.py
```

Streamlit will provide a local URL similar to:

```text
http://localhost:8501
```

Open the URL in a browser to use ResumeAI.

---

# Requirements

The project dependencies are listed in `requirements.txt`.

Core dependencies include:

```text
streamlit
joblib
pypdf
scikit-learn
numpy
pandas
```

The saved model and vectorizer should be loaded with compatible versions of the machine-learning libraries used during training.

---

# Example Workflow

A typical user workflow looks like this:

```text
Open ResumeAI
      ↓
Upload Resume
      ↓
Analyze Resume
      ↓
Review Resume Score
      ↓
Check Skills & Issues
      ↓
Enter Job Description
      ↓
Run Job Match
      ↓
Review Match Signal
      ↓
Identify Missing Skills
      ↓
Improve Resume
```

The same resume can then be analyzed again after improvements to compare the resulting analysis.

---

# Why This Project?

Traditional resume screening can be time-consuming, especially when a resume needs to be evaluated against multiple job descriptions.

ResumeAI explores how classical NLP and machine learning can be combined with rule-based analysis to build a practical career-oriented AI application.

The project demonstrates several important concepts:

* Text classification
* NLP feature engineering
* TF-IDF
* Logistic Regression
* Model serialization
* Resume information extraction
* Skill matching
* Rule-based scoring
* Streamlit application development
* End-to-end ML deployment

---

# Future Improvements

Possible next steps include:

* Transformer-based semantic similarity
* Sentence embeddings
* BERT-based resume/job matching
* Better entity extraction for skills and job titles
* Experience-level matching
* Education requirement matching
* Industry-specific scoring
* More robust PDF parsing
* Resume section classification
* Explainable ML predictions
* Recruiter dashboard
* Database-backed user history
* Resume improvement suggestions powered by an LLM
* Multi-language resume support
* Model evaluation dashboard

---

# Limitations

ResumeAI is a project-level resume analysis and matching system.

Its scores and match signals should be treated as indicators rather than definitive hiring decisions.

Resume formatting, terminology, dataset composition, model limitations, and differences between real-world job descriptions can all affect the output.

The current system primarily relies on TF-IDF-based text representation and rule-based analysis, so semantic similarity is more limited than what modern transformer-based systems can provide.

---

# Project Status

**Current status:** Functional ML-powered Streamlit application.

The project includes:

* Dataset processing
* Model training
* Trained model artifacts
* Resume analysis
* Job description matching
* Skill-gap detection
* Streamlit interface
* Local deployment workflow

---

## License

This project is intended for educational, portfolio, and experimental purposes.
