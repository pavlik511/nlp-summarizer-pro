import os
import re
import json
import math
import heapq
from collections import Counter

import streamlit as st
import docx
import pymupdf
from langdetect import detect, LangDetectException

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams

import spacy
from spacy.language import Language

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ==========================================
# 1. KONFIGURACE A TAJNÉ KLÍČE
# ==========================================
LIMITS = {
    "NLTK": 12000,
    "spaCy": 6000,
    "Gemma": 10000
}
MIN_WORDS_LIMIT = 30

# Načtení klíče primárně ze Streamlit Secrets (pro Cloud), fallback na lokální env
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# ==========================================
# 2. PARSOVÁNÍ A ČIŠTĚNÍ TEXTU
# ==========================================
def clean_text(text: str) -> str:
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'\[\d+(?:\s*,\s*\d+)*\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text(file_obj, file_extension: str) -> str:
    ext = file_extension.lower().replace('.', '')
    extracted_text = ""
    try:
        if ext == 'txt':
            file_bytes = file_obj.read()
            try:
                extracted_text = file_bytes.decode('utf-8')
            except UnicodeDecodeError:
                extracted_text = file_bytes.decode('cp1250')
        elif ext == 'docx':
            doc = docx.Document(file_obj)
            extracted_text = "\n".join([para.text for para in doc.paragraphs])
        elif ext == 'pdf':
            pdf_doc = pymupdf.open(stream=file_obj.read(), filetype="pdf")
            for page in pdf_doc:
                extracted_text += page.get_text() + "\n"
            pdf_doc.close()
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        return extracted_text.strip()
    except Exception as e:
        raise RuntimeError(f"Error parsing {ext} file: {str(e)}")

def count_sentences(text: str) -> int:
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    return len(sent_tokenize(text))

# ==========================================
# 3. NLP ENGINE: NLTK
# ==========================================
@st.cache_resource
def load_nltk_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    return set(stopwords.words('english'))

def generate_nltk_summary(text: str, sentence_count: int = 4) -> tuple[str, list[str]]:
    stop_words = load_nltk_resources()
    sentences = sent_tokenize(text)
    
    if len(sentences) <= 1:
        return ("Error: The text is too short. Cannot perform frequency analysis.", [])
    if sentence_count >= len(sentences):
        sentence_count = len(sentences) - 1
        
    tokens = word_tokenize(text.lower())
    word_frequencies = {}
    for word in tokens:
        if word not in stop_words and word.isalnum():
            word_frequencies[word] = word_frequencies.get(word, 0) + 1
            
    if not word_frequencies: 
        return None, []
    
    max_freq = max(word_frequencies.values())
    for word in word_frequencies:
        word_frequencies[word] /= max_freq
    
    all_sentences = []
    for paragraph in text.split('\n'):
        if paragraph.strip():
            all_sentences.extend(sent_tokenize(paragraph))

    sentence_scores = {}
    for sent in all_sentences:
        sentence_words = word_tokenize(sent.lower())
        if 5 < len(sentence_words) < 40:
            score = 0
            valid_words_count = 0
            for word in sentence_words:
                if word in word_frequencies:
                    score += word_frequencies[word]
                    valid_words_count += 1
            if valid_words_count > 0:
                sentence_scores[sent] = score / valid_words_count               

    top_sentences = heapq.nlargest(sentence_count, sentence_scores, key=sentence_scores.get)    
    top_sentences = sorted(top_sentences, key=all_sentences.index)
    summary = " ".join(top_sentences)

    valid_words = [w for w in tokens if w.isalnum() and w not in stop_words]
    bigrams = [' '.join(gram) for gram in ngrams(valid_words, 2)] if len(valid_words) >= 2 else []
    trigrams = [' '.join(gram) for gram in ngrams(valid_words, 3)] if len(valid_words) >= 3 else []
    
    candidates = list(word_frequencies.keys()) + bigrams + trigrams
    phrase_counts = {}
    text_lower = text.lower()
    
    for phrase in set(candidates):
        pattern = r'\b' + re.escape(phrase) + r'\b'
        freq = len(re.findall(pattern, text_lower))
        if len(phrase.split()) == 1 or freq > 1:
            phrase_counts[phrase] = freq
        
    sorted_phrases = sorted(phrase_counts.keys(), key=lambda x: len(x.split()), reverse=True)
    filtered_counts = phrase_counts.copy()
    
    for i, longer_phrase in enumerate(sorted_phrases):
        if filtered_counts[longer_phrase] > 0:
            for shorter_phrase in sorted_phrases[i+1:]:
                if re.search(r'\b' + re.escape(shorter_phrase) + r'\b', longer_phrase):
                    filtered_counts[shorter_phrase] -= phrase_counts[longer_phrase]
                    
    final_counts = {k: v for k, v in filtered_counts.items() if v > 0}
    keywords = heapq.nlargest(5, final_counts, key=final_counts.get)
    
    return summary, keywords

# ==========================================
# 4. NLP ENGINE: SPACY
# ==========================================
@Language.component("newline_boundary")
def newline_boundary(doc):
    for token in doc[:-1]:
        if "\n" in token.text:
            doc[token.i + 1].is_sent_start = True
    return doc

@st.cache_resource
def load_spacy_resources():
    try:
        model = spacy.load("en_core_web_md")
        model.add_pipe("newline_boundary", before="parser")
        return model
    except OSError:
        raise OSError("Missing model 'en_core_web_md'. Check requirements.txt")

def generate_spacy_summary(text: str, sentence_count: int = 4) -> tuple[str, list[str]]:
    nlp_model = load_spacy_resources()
    doc = nlp_model(text)
    sentences = list(doc.sents)
    total_sentences = len(sentences)

    if total_sentences <= 1:
        return ("Error: The text is too short. Cannot perform semantic analysis.", [])
    if sentence_count >= total_sentences:
        sentence_count = total_sentences - 1
    if not sentences:
        return None, []

    candidates = []
    for chunk in doc.noun_chunks:
        start = 1 if (chunk[0].is_stop or chunk[0].pos_ == "DET") and len(chunk) > 1 else 0
        cleaned_chunk = " ".join([t.lemma_.lower() for t in chunk[start:] if not t.is_punct]).strip()
        if cleaned_chunk and cleaned_chunk not in nlp_model.Defaults.stop_words and len(cleaned_chunk) > 2:
            candidates.append(cleaned_chunk)
            
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and not token.is_stop and not token.is_punct:
            candidates.append(token.lemma_.lower())
            
    tf_counts = Counter(candidates)
    if not tf_counts: 
        return None, []

    idf_scores = {}
    for word in tf_counts.keys():
        doc_freq = sum(1 for s in sentences if re.search(r'\b' + re.escape(word) + r'\b', s.text.lower()))
        idf_scores[word] = math.log(1 + (len(sentences) / (1 + doc_freq)))
        
    tf_idf_scores = {word: tf_counts[word] * idf_scores[word] for word in tf_counts}

    sorted_phrases = sorted(tf_idf_scores.keys(), key=lambda x: len(x.split()), reverse=True)
    filtered_tfidf = tf_idf_scores.copy()
    
    for i, longer_phrase in enumerate(sorted_phrases):
        if filtered_tfidf[longer_phrase] > 0:
            for shorter_phrase in sorted_phrases[i+1:]:
                if re.search(r'\b' + re.escape(shorter_phrase) + r'\b', longer_phrase):
                    filtered_tfidf[shorter_phrase] -= tf_idf_scores[longer_phrase]
                    
    final_tfidf = {k: v for k, v in filtered_tfidf.items() if v > 0}
    top_terms = sorted(final_tfidf, key=final_tfidf.get, reverse=True)[:10]
    
    theme_vector = nlp_model(" ") if not top_terms else nlp_model(" ".join(top_terms))

    base_scores = {}
    clean_sents = {}
    for sent in sentences:
        if len(sent.text.split()) < 5: 
            continue
        clean_sent = nlp_model(" ".join([t.lemma_.lower() for t in sent if not t.is_stop and not t.is_punct]))
        if clean_sent.vector_norm > 0 and theme_vector.vector_norm > 0:
            base_scores[sent] = clean_sent.similarity(theme_vector)
            clean_sents[sent] = clean_sent

    summary_nodes = []
    unselected_sentences = list(base_scores.keys())
    LAMBDA_PARAM = 0.7

    for _ in range(min(sentence_count, len(unselected_sentences))):
        best_sent = None
        max_mmr = -1.0 
        for sent in unselected_sentences:
            relevance = base_scores[sent]
            max_similarity_to_selected = 0.0
            if summary_nodes:
                for selected_sent in summary_nodes:
                    sim = clean_sents[sent].similarity(clean_sents[selected_sent])
                    if sim > max_similarity_to_selected:
                        max_similarity_to_selected = sim

            mmr_score = (LAMBDA_PARAM * relevance) - ((1 - LAMBDA_PARAM) * max_similarity_to_selected)
            if mmr_score > max_mmr:
                max_mmr = mmr_score
                best_sent = sent

        if best_sent:
            summary_nodes.append(best_sent)
            unselected_sentences.remove(best_sent)

    summary_nodes = sorted(summary_nodes, key=lambda x: x.start)
    return " ".join([s.text for s in summary_nodes]), top_terms[:5]

# ==========================================
# 5. NLP ENGINE: GEMMA (GOOGLE GENAI)
# ==========================================
class SummaryResponse(BaseModel):
    summary: str = Field(description="The summarized text.")
    keywords: list[str] = Field(description="List of exactly 5 keywords extracted from the text.")

def generate_gemma_summary(text: str, sentence_count: int = 4) -> tuple[str, list[str]]:
    if not GEMINI_API_KEY:
        return "Error: API key not found. Please set GEMINI_API_KEY in Streamlit Secrets.", []
    
    total_sentences = count_sentences(text)
    if total_sentences <= 1:
        return ("Error: The text is too short. Cannot generate a summary.", [])
    if sentence_count >= total_sentences:
        sentence_count = total_sentences - 1
    
    client = genai.Client(api_key=GEMINI_API_KEY)

    try:
        prompt = (
            f"Summarize the text below into exactly {sentence_count} sentences "
            "(or fewer if the source text is too short). Then extract up to 5 relevant keywords.\n"
            "You MUST return the result EXCLUSIVELY as a raw JSON object with the following structure:\n"
            "{\n"
            '  "summary": "your generated summary here",\n'
            '  "keywords": ["keyword1", "keyword2", "keyword3"]\n'
            "}\n"
            "Do NOT wrap the JSON in markdown blocks (like ```json). Just return the raw JSON string.\n\n"
            f"Text:\n{text}"
        )
        
        token_count = client.models.count_tokens(model='gemma-3-27b-it', contents=prompt).total_tokens
        if token_count > 14000:
            return f"Error: Text is too long ({token_count} tokens). Please shorten it.", []

        response = client.models.generate_content(
            model='gemma-3-27b-it',
            config=types.GenerateContentConfig(temperature=0.2),
            contents=prompt
        )

        try:
            p_tokens = response.usage_metadata.prompt_token_count
            c_tokens = response.usage_metadata.candidates_token_count
            if c_tokens is None:
                c_tokens = client.models.count_tokens(model='gemma-3-27b-it', contents=response.text).total_tokens
        except Exception:
            p_tokens, c_tokens = "N/A", "N/A"

        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:-3].strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:-3].strip()
            
        result = json.loads(raw_text)
        summary = result.get("summary", "")
        keywords = result.get("keywords", [])[:5] 
        summary += f"\n\n---\n📊 Token Economics:\n• Input (prompt): {p_tokens} tokens\n• Output (Completion): {c_tokens} tokens"
        
        return summary, keywords

    except json.JSONDecodeError:
        return "Error: Model did not return a valid JSON format.", []
    except Exception as e:
        return f"API Error: {str(e)}", []


# ==========================================
# 6. STREAMLIT FRONTEND (UŽIVATELSKÉ ROZHRANÍ)
# ==========================================
st.set_page_config(page_title="NLP Summarizer Pro", page_icon="🤖", layout="centered")

st.title("🤖 NLP Summarizer Pro")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ SELECT ENGINE")
    method = st.radio("Method", ["NLTK (Fast)", "spaCy (Semantic)", "Gemma 3 (LLM)"])
    
    st.header("📏 SUMMARY LENGTH")
    sentence_count = st.slider("Number of Sentences", 1, 10, 3)
    
    st.markdown("---")
    st.caption("Pro Edition | Web Demo")

st.markdown("### 📥 INPUT METHOD")

tab1, tab2 = st.tabs(["✍️ Paste Text", "📂 Upload File"])
text_to_analyze = ""

with tab1:
    pasted_text = st.text_area("Paste text for analysis:", height=300, placeholder="Paste your document text here...", key="pasted_area")
    if pasted_text.strip():
        text_to_analyze = pasted_text.strip()

with tab2:
    uploaded_file = st.file_uploader("Upload File (.txt, .docx, .pdf)", type=["txt", "docx", "pdf"])
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        try:
            text_to_analyze = extract_text(uploaded_file, file_extension)
        except Exception as e:
            st.error(f"Error reading file: {e}")

st.markdown("<br>", unsafe_allow_html=True) 

if st.button("🚀 ANALYZE AND SUMMARIZE", type="primary", use_container_width=True):
    if text_to_analyze:
        text_to_analyze = clean_text(text_to_analyze)
        
    if not text_to_analyze:
        st.warning("Please upload a document or paste text first.")
        st.stop()

    words = text_to_analyze.split()
    if len(words) < MIN_WORDS_LIMIT:
        st.warning(f"The text is too short ({len(words)} words). Please provide at least {MIN_WORDS_LIMIT} words.")
        st.stop() 
        
    engine_name = method.split()[0]
    current_limit = LIMITS.get(engine_name, 3000)

    if len(words) > current_limit:
        st.error(f"The text is too long ({len(words)} words). The {engine_name} engine has a limit of {current_limit} words.")
        st.stop()

    try:
        if detect(text_to_analyze) != 'en':
            st.warning("Warning: Non-English language detected. Results may be inaccurate.")
    except LangDetectException:
        pass

    actual_sentence_count = count_sentences(text_to_analyze)
    if actual_sentence_count <= sentence_count:
        if actual_sentence_count <= 1:
            st.error("The text is too short (contains only 1 sentence). Cannot generate a summary.")
            st.stop()
        else:
            st.info(f"💡 Info: The text only has {actual_sentence_count} sentences. Adjusting summary length to {actual_sentence_count - 1}.")
            sentence_count = actual_sentence_count - 1
            
    with st.spinner("✨ Running AI analysis..."):
        if "NLTK" in method:
            res, kw = generate_nltk_summary(text_to_analyze, sentence_count)
        elif "spaCy" in method:
            res, kw = generate_spacy_summary(text_to_analyze, sentence_count)
        else:
            res, kw = generate_gemma_summary(text_to_analyze, sentence_count)
            
    if res and not res.startswith(("Chyba", "Error", "API Error")):
        st.success("Analysis Complete!")
        engine_name = method.split()[0]
        st.markdown(f"**SUMMARY ({engine_name}):**\n\n> {res}")
        st.markdown(f"**KEYWORDS:** {', '.join(kw)}")
    else:
        st.error(res if res else "An unknown error occurred during generation.")