import pypdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


def extract_text_from_pdf(pdf_file):
  reader = pypdf.PdfReader(pdf_file)
  text = ""
  for page in reader.pages:
    text += page.extract_text() or ""
  return text


# Page Configuration
st.set_page_config(
    page_title="ATS Analysis & Resume Matching",
    page_icon="⚡",
    layout="centered",
)

# Custom CSS for Centered Text and Mobile/Desktop Responsiveness
st.markdown(
    """
    <style>
    .stApp {
        text-align: center;
    }
    h1, h2, h3, p, label, .stMarkdown, div {
        text-align: center !important;
    }
    .stTextArea textarea {
        text-align: left !important;
    }
    .stFileUploader {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    div.stButton > button {
        display: block;
        margin: 0 auto;
    }
    @media (max-width: 768px) {
        .stApp {
            padding: 10px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown(
    "<h1>ATS Analysis & Resume Matching</h1>", unsafe_allow_html=True
)
st.markdown(
    "<p>Compare your resume with any job description to check the match score"
    " and missing keywords.</p>",
    unsafe_allow_html=True,
)

st.write("")

# Inputs Section
job_description = st.text_area("Paste Job Description Here:")
uploaded_file = st.file_uploader(
    "Upload Resume (PDF format only):", type=["pdf"]
)

st.write("")

if st.button("Analyze Match"):
  if not job_description or not uploaded_file:
    st.warning("Please enter the job description and upload your resume first.")
  else:
    with st.spinner("Analyzing data and keywords..."):
      resume_text = extract_text_from_pdf(uploaded_file)

      # Match Score Calculation
      documents = [job_description, resume_text]
      vectorizer = TfidfVectorizer(stop_words="english")
      tfidf_matrix = vectorizer.fit_transform(documents)
      similarity_score = cosine_similarity(
          tfidf_matrix[0:1], tfidf_matrix[1:2]
      )[0][0]
      match_percentage = round(similarity_score * 100, 2)

      # Missing Keywords Extraction
      job_vec = TfidfVectorizer(
          stop_words="english", max_features=15
      ).fit([job_description])
      job_keywords = set(job_vec.get_feature_names_out())

      resume_lower = resume_text.lower()
      missing_keywords = [kw for kw in job_keywords if kw not in resume_lower]

      # Results Section
      st.success("Analysis Completed Successfully!")

      st.metric(label="ATS Match Score", value=f"{match_percentage}%")

      if match_percentage >= 75:
        st.info(
            "Excellent! Your resume is strongly aligned with the job"
            " requirements."
        )
      elif match_percentage >= 50:
        st.warning(
            "Acceptable, but it is recommended to add more keywords from the job"
            " description."
        )
      else:
        st.error(
            "Low match score. Please update your resume to include the required"
            " skills."
        )

      # Missing Keywords Display
      if missing_keywords:
        st.subheader("Missing Keywords in Your Resume:")
        st.write(
            "Adding these keywords (if applicable to your skills) will"
            " significantly improve your ATS screening results:"
        )
        st.code(", ".join(missing_keywords), language="text")
      else:
        st.success("Great job! No major key skills were found missing.")
