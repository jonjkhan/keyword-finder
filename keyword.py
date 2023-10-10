import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("stopwords")


def extract_keywords(text, num_keywords=5):
    # Tokenize the input text and remove stopwords and punctuation
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word.isalnum() and word not in stop_words]

    # Calculate word frequencies using Counter
    word_freq = Counter(words)

    # Calculate TF-IDF scores (for simplicity, you can assume a fixed IDF for each word)
    tfidf_scores = {word: word_freq[word] * 2.0 for word in word_freq}

    # Rank words by TF-IDF score and select the top N keywords
    keywords = [word for word, score in sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:num_keywords]]

    return keywords


# Example usage
if __name__ == "__main__":
    input_text = """
    My name is Jonathan Khan and I am a second-year Computer
    Science student with a passion for coding. I am Proficient
    in Python and experienced with tools like VSCode,
    Wing, and PyCharm, I also have some exposure into
    the world of web development with HTML and CSS, along with some
    exposure to Git and Netlify.
    I'm actively seeking an internship opportunity
    this summer to further expand my skills and contribute
    to exciting projects at your company
    """

    keywords = extract_keywords(input_text)
    print("Keywords:", keywords)
