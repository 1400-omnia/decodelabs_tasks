# 🚀 Career Match AI

An AI-powered career recommendation system that recommends suitable career paths based on a user's skills.

The system uses Content-Based Filtering, TF-IDF Vectorization, Cosine Similarity, Skill Overlap, and Technical Relevance to rank the most relevant career paths.

## 🎯 Project Objective

The goal of this project is to help users discover career opportunities that match their technical and professional skills.

The user enters at least three skills, and the system analyzes them against job roles in the dataset and returns the Top 3 recommended career paths.

## 🧠 How It Works

1. User Input
   - The user enters at least three skills.

2. Skill Preprocessing
   - Skills are normalized and cleaned.
   - Duplicate skills are removed.
   - Common aliases and simple spelling mistakes are handled.

3. TF-IDF Vectorization
   - Job skills are converted into TF-IDF vectors.

4. Cosine Similarity
   - The similarity between the user's skills and available job roles is calculated.

5. Skill Matching
   - Direct skill overlap is calculated between the user's skills and each job role.

6. Technical Relevance
   - Technical skills and technical job characteristics are considered in the recommendation score.

7. Ranking
   - The different scores are combined to rank the most relevant career paths.
   - The system returns the Top 3 recommendations.

## 📊 Dataset

The project uses a job dataset containing:

- Job ID
- Job Category
- Job Title
- Job Description
- Job Skill Set

Dataset size:

- 1,167 job roles
- 5 career categories
- 4,967 unique skills

Categories include:

- Information Technology
- Business Development
- Finance
- Sales
- Human Resources

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- TF-IDF
- Cosine Similarity
- Content-Based Recommendation

## 📁 Project Structure

task-3-omnia/
├── data/
│   └── raw_skills.csv
├── task3.py
├── app.py
├── requirements.txt
└── README.md

## 🚀 How to Run

1. Install the required libraries:

pip install -r requirements.txt

2. Run the Streamlit application:

streamlit run app.py

3. Open the application in your browser using the local Streamlit address.

## 💡 Example

Example user skills:

Python, SQL, Data Analysis

The system analyzes the skills and returns the Top 3 matching career paths along with:

- Match Score
- Cosine Similarity
- Skill Overlap
- Technical Relevance
- Matched Skills

## 🎥 Demo

Watch the Demo Video:
https://youtu.be/OYTDEVjkctE

## 👩‍💻 Project

DecodeLabs - Task 3

Career Match AI | Content-Based Career Recommendation System