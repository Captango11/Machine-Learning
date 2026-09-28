import streamlit as st
import re
import io

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Optional PDF extraction
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


# =========================================================
# 1. TRAINING DATA
# =========================================================

data = [
    # ---------------- DATA ANALYST ----------------
    ("python sql pandas numpy excel power bi tableau data visualization statistics", "Data Analyst"),
    ("sql excel powerbi dashboard reporting data cleaning pandas", "Data Analyst"),
    ("python pandas matplotlib seaborn sql business intelligence analytics", "Data Analyst"),
    ("data analysis excel sql tableau visualization reporting", "Data Analyst"),
    ("python numpy pandas exploratory data analysis statistics", "Data Analyst"),

    # ---------------- ML ENGINEER ----------------
    ("python machine learning tensorflow pytorch scikit learn deep learning", "ML Engineer"),
    ("python sklearn tensorflow neural networks model training deployment", "ML Engineer"),
    ("machine learning artificial intelligence python pytorch tensorflow", "ML Engineer"),
    ("deep learning NLP computer vision model deployment python", "ML Engineer"),
    ("scikit learn machine learning feature engineering model optimization", "ML Engineer"),

    # ---------------- WEB DEVELOPER ----------------
    ("html css javascript react node express frontend backend web development", "Web Developer"),
    ("react javascript typescript nodejs express mongodb full stack developer", "Web Developer"),
    ("html css javascript responsive websites frontend development", "Web Developer"),
    ("react tailwind javascript node mongodb REST API", "Web Developer"),
    ("frontend developer react javascript css html", "Web Developer"),

    # ---------------- DATA SCIENTIST ----------------
    ("python machine learning statistics pandas numpy data science", "Data Scientist"),
    ("python predictive modeling machine learning statistics data analysis", "Data Scientist"),
    ("data science python sklearn pandas statistics visualization", "Data Scientist"),
    ("machine learning regression classification clustering python", "Data Scientist"),
    ("python statistics machine learning predictive analytics", "Data Scientist"),

    # ---------------- CLOUD ENGINEER ----------------
    ("aws azure cloud docker kubernetes linux devops", "Cloud Engineer"),
    ("aws ec2 s3 lambda cloud infrastructure docker", "Cloud Engineer"),
    ("azure cloud computing kubernetes docker linux", "Cloud Engineer"),
    ("aws devops terraform docker kubernetes CI CD", "Cloud Engineer"),
    ("cloud engineer aws linux networking docker", "Cloud Engineer"),
]


texts = [item[0] for item in data]
labels = [item[1] for item in data]


# =========================================================
# 2. TEXT PREPROCESSING
# =========================================================

def clean_text(text):
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Keep letters/numbers
    text = re.sub(r"[^a-zA-Z0-9+#.\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


cleaned_texts = [clean_text(text) for text in texts]


# =========================================================
# 3. ML MODEL
# =========================================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=3000,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])


# Train model
model.fit(cleaned_texts, labels)


# =========================================================
# 4. MODEL ACCURACY
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    cleaned_texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

evaluation_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=3000
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])

evaluation_model.fit(X_train, y_train)

predictions = evaluation_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)


# =========================================================
# 5. PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_text(uploaded_file):

    if not PDF_AVAILABLE:
        return ""

    pdf_reader = PyPDF2.PdfReader(uploaded_file)

    text = ""

    for page in pdf_reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# =========================================================
# 6. PREDICTION
# =========================================================

def predict_roles(resume_text):

    cleaned_resume = clean_text(resume_text)

    probabilities = model.predict_proba([cleaned_resume])[0]

    classes = model.classes_

    results = list(zip(classes, probabilities))

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return results


# =========================================================
# 7. EXTRACT SKILLS
# =========================================================

SKILLS = [
    "python",
    "java",
    "c++",
    "javascript",
    "typescript",
    "sql",
    "mysql",
    "mongodb",
    "postgresql",
    "react",
    "node",
    "html",
    "css",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "machine learning",
    "deep learning",
    "nlp",
    "power bi",
    "tableau",
    "excel",
    "aws",
    "azure",
    "docker",
    "kubernetes",
    "linux",
    "git"
]


def extract_skills(text):

    text = text.lower()

    found = []

    for skill in SKILLS:

        if skill in text:
            found.append(skill)

    return sorted(set(found))


# =========================================================
# 8. STREAMLIT UI
# =========================================================

st.set_page_config(
    page_title="Resume Job Classifier",
    page_icon="💼",
    layout="wide"
)


st.title("💼 Resume → Job Role Classifier")

st.write(
    "An NLP + Machine Learning application that "
    "classifies resumes into suitable technology roles."
)


# Sidebar
with st.sidebar:

    st.header("📊 Model Information")

    st.write("Algorithm:")
    st.code("TF-IDF + Logistic Regression")

    st.write("NLP:")
    st.write("Text preprocessing + n-grams + TF-IDF")

    st.write("Training classes:")

    for role in sorted(set(labels)):
        st.write(f"• {role}")

    st.metric(
        "Test Accuracy",
        f"{accuracy * 100:.1f}%"
    )


# =========================================================
# INPUT SECTION
# =========================================================

st.subheader("📄 Upload Resume")

uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["pdf", "txt"]
)


resume_text = ""


if uploaded_file:

    if uploaded_file.type == "application/pdf":

        if PDF_AVAILABLE:
            resume_text = extract_pdf_text(uploaded_file)

        else:
            st.error(
                "PyPDF2 is not installed. "
                "Run: pip install PyPDF2"
            )

    else:

        resume_text = uploaded_file.read().decode(
            "utf-8",
            errors="ignore"
        )


st.subheader("✍️ Or paste your resume")

manual_text = st.text_area(
    "Resume text",
    value=resume_text,
    height=250,
    placeholder="""
Example:

Python developer with experience in pandas,
numpy, SQL, machine learning and Power BI.
Built data analysis dashboards and predictive
models using Python and scikit-learn.
"""
)


# =========================================================
# PREDICT BUTTON
# =========================================================

if st.button(
    "🚀 Analyze Resume",
    type="primary"
):

    if not manual_text.strip():

        st.warning(
            "Please upload a resume or paste resume text."
        )

    else:

        results = predict_roles(manual_text)

        skills = extract_skills(manual_text)


        # ---------------------------------------------
        # TOP PREDICTION
        # ---------------------------------------------

        top_role = results[0][0]
        top_probability = results[0][1]


        st.success(
            f"Predicted Role: {top_role}"
        )


        st.metric(
            "Confidence",
            f"{top_probability * 100:.1f}%"
        )


        # ---------------------------------------------
        # ROLE PROBABILITIES
        # ---------------------------------------------

        st.subheader("🎯 Job Role Predictions")


        for role, probability in results:

            st.write(
                f"**{role}** — "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                float(probability)
            )


        # ---------------------------------------------
        # SKILLS
        # ---------------------------------------------

        st.subheader("🛠️ Detected Skills")


        if skills:

            cols = st.columns(4)

            for i, skill in enumerate(skills):

                cols[i % 4].success(
                    skill
                )

        else:

            st.info(
                "No predefined skills detected."
            )


        # ---------------------------------------------
        # RECOMMENDATION
        # ---------------------------------------------

        st.subheader("💡 Recommendation")

        if top_role == "Data Analyst":

            st.write(
                "Your resume contains strong signals "
                "for data analysis, SQL, visualization "
                "and business intelligence."
            )

        elif top_role == "ML Engineer":

            st.write(
                "Your resume contains strong machine "
                "learning and Python-related skills."
            )

        elif top_role == "Web Developer":

            st.write(
                "Your resume contains strong web "
                "development and JavaScript signals."
            )

        elif top_role == "Data Scientist":

            st.write(
                "Your resume contains strong data science, "
                "statistics and predictive modeling signals."
            )

        elif top_role == "Cloud Engineer":

            st.write(
                "Your resume contains strong cloud, "
                "DevOps and infrastructure signals."
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "NLP Project | TF-IDF | Logistic Regression | Streamlit"
)