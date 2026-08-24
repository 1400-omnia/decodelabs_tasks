import pandas as pd
import ast

from difflib import get_close_matches

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/raw_skills.csv"

MIN_SKILLS = 3
TOP_N = 3

# Recommendation weights
COSINE_WEIGHT = 0.30
OVERLAP_WEIGHT = 0.50
TECHNICAL_WEIGHT = 0.20

# Minimum direct skill overlap required
MIN_OVERLAP_FOR_RECOMMENDATION = 0.33


# ============================================================
# TECHNICAL SKILLS
# ============================================================

TECH_KEYWORDS = {
    "python",
    "java",
    "javascript",
    "typescript",
    "c++",
    "cpp",
    "c#",
    "sql",
    "nosql",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "data science",
    "data analysis",
    "data engineering",
    "cloud computing",
    "aws",
    "azure",
    "google cloud",
    "gcp",
    "docker",
    "kubernetes",
    "devops",
    "automation",
    "linux",
    "networking",
    "cybersecurity",
    "information security",
    "tensorflow",
    "pytorch",
    "scikit learn",
    "git",
    "github",
    "api",
    "rest api",
    "software development",
    "software engineering",
    "programming",
    "database",
    "databases",
}


# ============================================================
# SKILL ALIASES
# ============================================================

SKILL_ALIASES = {
    "cpp": "c++",
    "c plus plus": "c++",
    "c-sharp": "c#",
    "c sharp": "c#",
    "js": "javascript",
    "ts": "typescript",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "dl": "deep learning",
    "scikit-learn": "scikit learn",
    "google cloud platform": "google cloud",
}


# ============================================================
# NORMALIZE SKILL
# ============================================================

def normalize_skill(skill):

    skill = str(skill).strip().lower()

    skill = skill.replace("_", " ")
    skill = skill.replace("-", " ")

    skill = " ".join(skill.split())

    if skill in SKILL_ALIASES:
        skill = SKILL_ALIASES[skill]

    return skill


# ============================================================
# PARSE SKILL SET
# ============================================================

def parse_skill_set(skill_set):

    try:

        skills = ast.literal_eval(skill_set)

        if isinstance(skills, list):
            return skills

        return []

    except (ValueError, SyntaxError, TypeError):

        return []


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(file_path):

    return pd.read_csv(file_path)


# ============================================================
# DATASET INFORMATION
# ============================================================

def inspect_dataset(df):

    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print(f"Dataset shape: {df.shape}")

    print("\nColumns:")

    for column in df.columns:
        print(f"- {column}")

    print("\nMissing values:")

    print(df.isnull().sum())


# ============================================================
# CATEGORY ANALYSIS
# ============================================================

def display_category_analysis(df):

    print("\n" + "=" * 60)
    print("DATASET CATEGORY ANALYSIS")
    print("=" * 60)

    counts = df["category"].value_counts()

    print("\nCategories and number of jobs:")

    for category, count in counts.items():

        percentage = count / len(df) * 100

        print(
            f"- {category}: "
            f"{count} jobs ({percentage:.2f}%)"
        )


# ============================================================
# PREPROCESS SKILLS
# ============================================================

def preprocess_skills(df):

    df = df.copy()

    df["parsed_skills"] = df["job_skill_set"].apply(
        parse_skill_set
    )

    df["parsed_skills"] = df["parsed_skills"].apply(

        lambda skills: list(
            dict.fromkeys(

                normalize_skill(skill)

                for skill in skills

                if str(skill).strip()
            )
        )
    )

    df["skill_document"] = df["parsed_skills"].apply(
        lambda skills: " ".join(skills)
    )

    return df


# ============================================================
# BUILD TF-IDF MODEL
# ============================================================

def build_tfidf_model(df):

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        token_pattern=r"(?u)\b[\w+#]+\b"
    )

    matrix = vectorizer.fit_transform(
        df["skill_document"]
    )

    return vectorizer, matrix


# ============================================================
# GET VOCABULARY
# ============================================================

def get_skill_vocabulary(df):

    vocabulary = set()

    for skills in df["parsed_skills"]:

        for skill in skills:
            vocabulary.add(skill)

    return vocabulary


# ============================================================
# CORRECT SKILL
# ============================================================

def correct_skill(skill, vocabulary):

    normalized = normalize_skill(skill)

    # Exact match
    if normalized in vocabulary:
        return normalized

    # Alias
    if normalized in SKILL_ALIASES:

        alias = SKILL_ALIASES[normalized]

        if alias in vocabulary:
            return alias

    # Typo correction
    matches = get_close_matches(
        normalized,
        vocabulary,
        n=1,
        cutoff=0.85
    )

    if matches:
        return matches[0]

    return normalized


# ============================================================
# USER INPUT
# ============================================================

def get_user_skills(vocabulary):

    print("\n" + "=" * 60)
    print("USER PROFILE")
    print("=" * 60)

    print(
        f"\nEnter at least {MIN_SKILLS} skills "
        "separated by commas."
    )

    print(
        "Example: Python, Cloud Computing, Automation"
    )

    while True:

        user_input = input("\nYour skills: ")

        raw_skills = [
            skill.strip()
            for skill in user_input.split(",")
            if skill.strip()
        ]

        processed_skills = []

        for skill in raw_skills:

            corrected = correct_skill(
                skill,
                vocabulary
            )

            if corrected not in processed_skills:

                processed_skills.append(
                    corrected
                )

        if len(processed_skills) < MIN_SKILLS:

            print(
                f"\nPlease enter at least "
                f"{MIN_SKILLS} different skills."
            )

            continue

        print("\nProcessed Skills:")

        for original, processed in zip(
            raw_skills,
            processed_skills
        ):

            if normalize_skill(original) != processed:

                print(
                    f"- {original} -> {processed}"
                )

            else:

                print(
                    f"- {processed}"
                )

        return processed_skills


# ============================================================
# USER VECTOR
# ============================================================

def create_user_vector(
    user_skills,
    vectorizer
):

    user_document = " ".join(user_skills)

    return vectorizer.transform(
        [user_document]
    )


# ============================================================
# COSINE SIMILARITY
# ============================================================

def calculate_cosine_similarity(
    user_vector,
    tfidf_matrix
):

    return cosine_similarity(
        user_vector,
        tfidf_matrix
    )[0]


# ============================================================
# SKILL OVERLAP
# ============================================================

def calculate_skill_overlap(
    user_skills,
    job_skills
):

    user_set = set(user_skills)
    job_set = set(job_skills)

    if not user_set:
        return 0.0

    matched = user_set.intersection(
        job_set
    )

    return len(matched) / len(user_set)


# ============================================================
# TECHNICAL RELEVANCE
# ============================================================

def calculate_technical_relevance(
    category,
    job_title,
    job_skills
):

    score = 0.0

    category_text = str(
        category
    ).lower()

    title_text = str(
        job_title
    ).lower()

    # IT category
    if category_text == "information-technology":
        score += 0.50

    # Technical job titles
    technical_title_terms = [
        "engineer",
        "developer",
        "software",
        "data",
        "cloud",
        "devops",
        "cyber",
        "network",
        "technology",
        "technical",
        "programmer",
        "architect",
        "machine learning",
        "artificial intelligence",
    ]

    if any(
        term in title_text
        for term in technical_title_terms
    ):

        score += 0.20

    # Technical skills
    technical_count = sum(
        1
        for skill in job_skills
        if skill in TECH_KEYWORDS
    )

    score += min(
        technical_count * 0.05,
        0.30
    )

    return min(score, 1.0)


# ============================================================
# OVERLAP PENALTY
# ============================================================

def calculate_overlap_penalty(
    overlap
):

    if overlap >= 1.0:

        return 1.00

    elif overlap >= 0.66:

        return 0.95

    elif overlap >= 0.33:

        return 0.75

    else:

        return 0.00


# ============================================================
# BUILD RECOMMENDATIONS
# ============================================================

def build_recommendations(
    df,
    user_skills,
    cosine_scores
):

    results = df.copy()

    # --------------------------------------------------------
    # Direct skill overlap
    # --------------------------------------------------------

    results["skill_overlap"] = results[
        "parsed_skills"
    ].apply(

        lambda skills:
        calculate_skill_overlap(
            user_skills,
            skills
        )
    )

    # --------------------------------------------------------
    # Technical relevance
    # --------------------------------------------------------

    results["technical_relevance"] = results.apply(

        lambda row:
        calculate_technical_relevance(
            row["category"],
            row["job_title"],
            row["parsed_skills"]
        ),

        axis=1
    )

    # --------------------------------------------------------
    # Cosine similarity
    # --------------------------------------------------------

    results["cosine_similarity"] = cosine_scores

    # --------------------------------------------------------
    # Require meaningful direct skill overlap
    # --------------------------------------------------------

    results = results[
        results["skill_overlap"]
        >= MIN_OVERLAP_FOR_RECOMMENDATION
    ].copy()

    # --------------------------------------------------------
    # Base score
    #
    # Direct skill overlap has the highest weight because
    # matching the user's actual skills is more important
    # than general textual similarity.
    # --------------------------------------------------------

    results["base_score"] = (

        results["cosine_similarity"]
        * COSINE_WEIGHT

        +

        results["skill_overlap"]
        * OVERLAP_WEIGHT

        +

        results["technical_relevance"]
        * TECHNICAL_WEIGHT

    )

    # --------------------------------------------------------
    # Apply overlap penalty
    # --------------------------------------------------------

    results["overlap_penalty"] = (
        results["skill_overlap"]
        .apply(calculate_overlap_penalty)
    )

    results["final_score"] = (
        results["base_score"]
        * results["overlap_penalty"]
    )

    # --------------------------------------------------------
    # Sort by final score
    # --------------------------------------------------------

    results = results.sort_values(
        by="final_score",
        ascending=False
    )

    # --------------------------------------------------------
    # Remove duplicate job titles
    #
    # Keep only the highest-scoring occurrence.
    # --------------------------------------------------------

    results = results.drop_duplicates(
        subset=["job_title"],
        keep="first"
    )

    return results


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_recommendations(
    recommendations,
    user_skills
):

    print("\n" + "=" * 60)
    print("RECOMMENDATION RESULTS")
    print("=" * 60)

    print("\nYour Skills:")

    for skill in user_skills:

        print(f"- {skill}")

    print("\n" + "=" * 60)
    print(
        f"TOP {TOP_N} RECOMMENDED CAREER PATHS"
    )
    print("=" * 60)

    top_results = recommendations.head(
        TOP_N
    )

    if top_results.empty:

        print(
            "\nNo matching career paths found."
        )

        return

    for index, (_, row) in enumerate(
        top_results.iterrows(),
        start=1
    ):

        final_score = (
            row["final_score"] * 100
        )

        cosine_score = (
            row["cosine_similarity"] * 100
        )

        overlap_score = (
            row["skill_overlap"] * 100
        )

        technical_score = (
            row["technical_relevance"] * 100
        )

        matched_skills = list(
            set(user_skills).intersection(
                set(row["parsed_skills"])
            )
        )

        print(
            f"\n{index}. "
            f"{row['job_title']}"
        )

        print(
            f"   Category: "
            f"{row['category']}"
        )

        print(
            f"   Match Score: "
            f"{final_score:.2f}%"
        )

        print(
            f"   Cosine Similarity: "
            f"{cosine_score:.2f}%"
        )

        print(
            f"   Skill Overlap: "
            f"{overlap_score:.2f}%"
        )

        print(
            f"   Technical Relevance: "
            f"{technical_score:.2f}%"
        )

        print(
            "   Matched Skills:"
        )

        for skill in matched_skills:

            print(
                f"   ✓ {skill}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    df = load_dataset(
        DATA_PATH
    )

    # --------------------------------------------------------
    # 2. Inspect dataset
    # --------------------------------------------------------

    inspect_dataset(df)

    # --------------------------------------------------------
    # 3. Category analysis
    # --------------------------------------------------------

    display_category_analysis(df)

    # --------------------------------------------------------
    # 4. Preprocess skills
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PROCESSING SKILL DATA")
    print("=" * 60)

    df = preprocess_skills(df)

    print(
        f"\nProcessed {len(df)} job records successfully."
    )

    # --------------------------------------------------------
    # 5. Build TF-IDF
    # --------------------------------------------------------

    vectorizer, tfidf_matrix = (
        build_tfidf_model(df)
    )

    print("\n" + "=" * 60)
    print("TF-IDF MODEL")
    print("=" * 60)

    print(
        f"Number of job roles: "
        f"{tfidf_matrix.shape[0]}"
    )

    print(
        f"Number of TF-IDF features: "
        f"{tfidf_matrix.shape[1]}"
    )

    # --------------------------------------------------------
    # 6. Vocabulary
    # --------------------------------------------------------

    vocabulary = get_skill_vocabulary(
        df
    )

    print(
        f"Number of unique skills: "
        f"{len(vocabulary)}"
    )

    # --------------------------------------------------------
    # 7. User input
    # --------------------------------------------------------

    user_skills = get_user_skills(
        vocabulary
    )

    # --------------------------------------------------------
    # 8. User vector
    # --------------------------------------------------------

    user_vector = create_user_vector(
        user_skills,
        vectorizer
    )

    # --------------------------------------------------------
    # 9. Cosine similarity
    # --------------------------------------------------------

    cosine_scores = (
        calculate_cosine_similarity(
            user_vector,
            tfidf_matrix
        )
    )

    # --------------------------------------------------------
    # 10. Recommendations
    # --------------------------------------------------------

    recommendations = build_recommendations(
        df,
        user_skills,
        cosine_scores
    )

    # --------------------------------------------------------
    # 11. Display
    # --------------------------------------------------------

    display_recommendations(
        recommendations,
        user_skills
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()