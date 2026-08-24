import streamlit as st

from task3 import (
    load_dataset,
    preprocess_skills,
    build_tfidf_model,
    get_skill_vocabulary,
    correct_skill,
    create_user_vector,
    calculate_cosine_similarity,
    build_recommendations,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Career Match AI",
    page_icon="🚀",
    layout="wide"
)


# ============================================================
# CUSTOM COLORS ONLY
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       MAIN BACKGROUND
       ======================================================== */

    .stApp {
        background-color: #F5EBDD;
    }

    /* Main content background */
    [data-testid="stAppViewContainer"] {
        background-color: #F5EBDD;
    }

    /* ========================================================
       TEXT COLOR
       ======================================================== */

    .stApp,
    .stApp p,
    .stApp span,
    .stApp label,
    .stApp div {
        color: #111111;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #111111 !important;
    }

    /* ========================================================
       SIDEBAR
       ======================================================== */

    [data-testid="stSidebar"] {
        background-color: #E8D5BD;
    }

    [data-testid="stSidebar"] * {
        color: #111111 !important;
    }

    /* ========================================================
       INPUT
       ======================================================== */

    div[data-baseweb="input"] {
        background-color: #FFFFFF;
    }

    div[data-baseweb="input"] input {
        color: #111111 !important;
    }

    /* ========================================================
       TEXT AREA / INPUT PLACEHOLDER
       ======================================================== */

    input::placeholder {
        color: #666666 !important;
    }

    /* ========================================================
       BUTTON
       ======================================================== */

    .stButton button {
        color: #FFFFFF !important;
    }

    /* ========================================================
       METRICS
       ======================================================== */

    [data-testid="stMetricValue"] {
        color: #111111 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #111111 !important;
    }

    /* ========================================================
       CONTAINERS
       ======================================================== */

    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFDF9;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    df = load_dataset(
        "data/raw_skills.csv"
    )

    df = preprocess_skills(df)

    return df


# ============================================================
# BUILD MODEL
# ============================================================

@st.cache_resource
def load_model(_df):

    vectorizer, tfidf_matrix = (
        build_tfidf_model(_df)
    )

    vocabulary = get_skill_vocabulary(
        _df
    )

    return (
        vectorizer,
        tfidf_matrix,
        vocabulary
    )


# ============================================================
# LOAD DATA + MODEL
# ============================================================

df = load_data()

vectorizer, tfidf_matrix, vocabulary = (
    load_model(df)
)


# ============================================================
# HEADER
# ============================================================

st.title("🚀 Career Match AI")

st.subheader(
    "AI-Powered Career Recommendation System"
)

st.write(
    """
    Discover career paths that match your skills using
    Content-Based Filtering, TF-IDF, Cosine Similarity,
    and direct Skill Matching.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🧠 About the System")

    st.write(
        """
        This recommendation engine uses:

        • Content-Based Filtering
        • TF-IDF Vectorization
        • Cosine Similarity
        • Skill Overlap
        • Technical Relevance
        • Top-N Ranking
        """
    )

    st.divider()

    st.metric(
        "Job Roles",
        f"{len(df):,}"
    )

    st.metric(
        "Unique Skills",
        f"{len(vocabulary):,}"
    )


# ============================================================
# USER INPUT
# ============================================================

st.markdown("## 🎯 Find Your Career Match")

st.write(
    "Enter at least 3 skills separated by commas."
)

skills_input = st.text_input(
    "Your Skills",
    placeholder="Python, Cloud Computing, Automation"
)


# ============================================================
# RECOMMEND BUTTON
# ============================================================

if st.button(
    "🔍 Get Career Recommendations",
    type="primary"
):

    # --------------------------------------------------------
    # Validate empty input
    # --------------------------------------------------------

    if not skills_input.strip():

        st.warning(
            "Please enter your skills first."
        )

        st.stop()


    # --------------------------------------------------------
    # Parse skills
    # --------------------------------------------------------

    raw_skills = [

        skill.strip()

        for skill in skills_input.split(",")

        if skill.strip()
    ]


    # --------------------------------------------------------
    # Normalize + correct skills
    # --------------------------------------------------------

    user_skills = []

    corrections = []

    for skill in raw_skills:

        normalized = skill.lower().strip()

        corrected = correct_skill(
            skill,
            vocabulary
        )

        if corrected not in user_skills:

            user_skills.append(
                corrected
            )

        if corrected != normalized:

            corrections.append(
                (
                    skill,
                    corrected
                )
            )


    # --------------------------------------------------------
    # Minimum skills
    # --------------------------------------------------------

    if len(user_skills) < 3:

        st.error(
            "Please enter at least 3 different skills."
        )

        st.stop()


    # ========================================================
    # USER PROFILE
    # ========================================================

    st.markdown("### 👤 Your Skill Profile")

    cols = st.columns(
        len(user_skills)
    )

    for col, skill in zip(
        cols,
        user_skills
    ):

        col.success(
            f"✓ {skill}"
        )


    # ========================================================
    # SKILL CORRECTIONS
    # ========================================================

    if corrections:

        st.info(
            "Some skills were normalized or corrected."
        )

        for original, corrected in corrections:

            st.write(
                f"**{original}** → `{corrected}`"
            )


    # ========================================================
    # USER VECTOR
    # ========================================================

    user_vector = create_user_vector(
        user_skills,
        vectorizer
    )


    # ========================================================
    # COSINE SIMILARITY
    # ========================================================

    cosine_scores = (
        calculate_cosine_similarity(
            user_vector,
            tfidf_matrix
        )
    )


    # ========================================================
    # BUILD RECOMMENDATIONS
    # ========================================================

    recommendations = build_recommendations(
        df,
        user_skills,
        cosine_scores
    )


    # ========================================================
    # RESULTS
    # ========================================================

    st.divider()

    st.markdown(
        "## 🏆 Top Career Matches"
    )


    top_results = recommendations.head(3)


    if top_results.empty:

        st.warning(
            "No matching career paths were found."
        )

        st.stop()


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

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

                set(
                    row["parsed_skills"]
                )
            )
        )


        # ----------------------------------------------------
        # Career Card
        # ----------------------------------------------------

        with st.container(
            border=True
        ):

            st.markdown(
                f"### {index}. {row['job_title']}"
            )

            st.write(
                f"**Category:** "
                f"{row['category']}"
            )


            # ------------------------------------------------
            # Main Match Score
            # ------------------------------------------------

            st.progress(
                min(final_score / 100, 1.0)
            )

            st.metric(
                "🎯 Match Score",
                f"{final_score:.2f}%"
            )


            # ------------------------------------------------
            # Score Breakdown
            # ------------------------------------------------

            col1, col2, col3 = st.columns(3)


            col1.metric(
                "Cosine Similarity",
                f"{cosine_score:.2f}%"
            )


            col2.metric(
                "Skill Overlap",
                f"{overlap_score:.2f}%"
            )


            col3.metric(
                "Technical Relevance",
                f"{technical_score:.2f}%"
            )


            # ------------------------------------------------
            # Matched Skills
            # ------------------------------------------------

            st.markdown(
                "**🔗 Matched Skills**"
            )


            if matched_skills:

                for skill in matched_skills:

                    st.write(
                        f"✓ {skill}"
                    )

            else:

                st.write(
                    "No direct skill matches."
                )


# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

st.markdown(
    "## ⚙️ How It Works"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        "### 1️⃣ Input"
    )

    st.write(
        "User provides at least three skills."
    )


with col2:

    st.markdown(
        "### 2️⃣ Vectorization"
    )

    st.write(
        "Skills are transformed into TF-IDF vectors."
    )


with col3:

    st.markdown(
        "### 3️⃣ Similarity"
    )

    st.write(
        "Cosine Similarity measures career alignment."
    )


with col4:

    st.markdown(
        "### 4️⃣ Ranking"
    )

    st.write(
        "The system returns the Top 3 career paths."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Career Match AI | DecodeLabs Project 3 | "
    "Content-Based Recommendation System"
)