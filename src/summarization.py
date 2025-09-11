import re
import nltk
from nltk.tokenize import sent_tokenize

# نحاول نعمل download للـ punkt مرة واحدة
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

def summarize_review(text, num_sentences=3):
    """
    Summarize the input review text by extracting the first few sentences.
    لو الـ NLTK مش شغال هنستخدم regex كـ fallback.
    """
    try:
        sentences = sent_tokenize(text)
    except Exception:
        # fallback بسيط لو الـ NLTK مش متاح
        sentences = re.split(r'[.!?]', text)

    # لو عدد الجمل أقل من المطلوب نرجّع النص كله
    if len(sentences) <= num_sentences:
        return text.strip()

    return " ".join(sentences[:num_sentences]).strip()
