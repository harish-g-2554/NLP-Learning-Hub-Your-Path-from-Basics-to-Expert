from flask import Flask, render_template, request, jsonify
import regex as reg
import numpy as np
import math
from collections import Counter
import nltk
import spacy
from nltk.util import ngrams
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from gensim.models import Word2Vec

# Load models once at startup
try:
    nlp_spacy = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp_spacy = spacy.load("en_core_web_sm")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('taggers/tagsets')
except LookupError:
    nltk.download('tagsets')


app = Flask(__name__)

# FIX: Added a dictionary for POS tag descriptions to make the app self-contained
POS_TAG_DESCRIPTIONS = {
    "CC": "Coordinating conjunction",
    "CD": "Cardinal number",
    "DT": "Determiner",
    "EX": "Existential there",
    "FW": "Foreign word",
    "IN": "Preposition or subordinating conjunction",
    "JJ": "Adjective",
    "JJR": "Adjective, comparative",
    "JJS": "Adjective, superlative",
    "LS": "List item marker",
    "MD": "Modal",
    "NN": "Noun, singular or mass",
    "NNS": "Noun, plural",
    "NNP": "Proper noun, singular",
    "NNPS": "Proper noun, plural",
    "PDT": "Predeterminer",
    "POS": "Possessive ending",
    "PRP": "Personal pronoun",
    "PRP$": "Possessive pronoun",
    "RB": "Adverb",
    "RBR": "Adverb, comparative",
    "RBS": "Adverb, superlative",
    "RP": "Particle",
    "SYM": "Symbol",
    "TO": "to",
    "UH": "Interjection",
    "VB": "Verb, base form",
    "VBD": "Verb, past tense",
    "VBG": "Verb, gerund or present participle",
    "VBN": "Verb, past participle",
    "VBP": "Verb, non-3rd person singular present",
    "VBZ": "Verb, 3rd person singular present",
    "WDT": "Wh-determiner",
    "WP": "Wh-pronoun",
    "WP$": "Possessive wh-pronoun",
    "WRB": "Wh-adverb",
    ".": "Punctuation",
    ",": "Punctuation",
    ":": "Punctuation",
    "(": "Punctuation",
    ")": "Punctuation",
    "``": "Opening quotation mark",
    "''": "Closing quotation mark",
}


# ---------- Pages ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/journey")
def journey():
    return render_template("modules.html")

@app.route("/modules")
def modules():
    return render_template("modules_overview.html")

@app.route("/modules/1-nlp")
def module_nlp():
    return render_template("module_nlp.html")

@app.route("/modules/2-text-regex")
def module_text_regex():
    return render_template("module_text_regex.html")

@app.route("/modules/3-edit-distance")
def module_edit_distance():
    return render_template("module_edit_distance.html")

@app.route("/modules/4-ngram-pos-ner")
def module_ngram_pos_ner():
    return render_template("module_ngram_pos_ner.html")

@app.route("/modules/5-wsd")
def module_wsd():
    return render_template("module_wsd.html")
    
@app.route("/setup-python-nlp")
def setup_python_nlp():
    return render_template("setup_python_nlp.html")

@app.route("/modules/6-sparse-vector")
def module_sparse_vector():
    return render_template("module_sparse_vector.html")

@app.route("/modules/7-dense-vector")
def module_dense_vector():
    return render_template("module_dense_vector.html")

@app.route("/modules/8-sparse-app")
def module_sparse_app():
    return render_template("module_sparse_app.html")

@app.route("/modules/9-dense-app")
def module_dense_app():
    return render_template("module_dense_app.html")


# ---------- APIs ----------

# Restored regex API endpoint
@app.route("/api/regex/find", methods=["POST"])
def api_regex_find():
    try:
        data = request.get_json(force=True) or {}
        pattern = data.get("pattern", "")
        text = data.get("text", "")
        rx = reg.compile(pattern)
        matches = [{"match": m.group(0), "start": m.start(), "end": m.end()} for m in rx.finditer(text)]
        return jsonify({"matches": matches})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Function to calculate Levenshtein and perform traceback
def calculate_levenshtein_with_traceback(word1, word2):
    m, n = len(word1), len(word2)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(m + 1): dp[i, 0] = i
    for j in range(n + 1): dp[0, j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if word1[i - 1] == word2[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1,        # Deletion
                           dp[i, j - 1] + 1,        # Insertion
                           dp[i - 1, j - 1] + cost) # Substitution / Match
    
    # Traceback
    path, operations = [], []
    align1, align2 = "", ""
    stats = {'matches': 0, 'substitutions': 0, 'insertions': 0, 'deletions': 0}
    i, j = m, n
    while i > 0 or j > 0:
        path.append((i, j))
        is_match = (i > 0 and j > 0 and word1[i-1] == word2[j-1])
        cost = 0 if is_match else 1
        
        diag_cost = dp[i-1, j-1] if i > 0 and j > 0 else float('inf')
        up_cost = dp[i-1, j] if i > 0 else float('inf')
        left_cost = dp[i, j-1] if j > 0 else float('inf')

        if i > 0 and j > 0 and dp[i,j] == diag_cost + cost:
            op_char = "Match" if cost == 0 else "Substitute"
            operations.append(f"{op_char} '{word1[i-1]}' with '{word2[j-1]}'")
            align1 = word1[i-1] + align1
            align2 = word2[j-1] + align2
            stats['matches' if cost == 0 else 'substitutions'] += 1
            i, j = i-1, j-1
        elif i > 0 and dp[i,j] == up_cost + 1:
            operations.append(f"Delete '{word1[i-1]}'")
            align1 = word1[i-1] + align1
            align2 = "-" + align2
            stats['deletions'] += 1
            i -= 1
        else:
            operations.append(f"Insert '{word2[j-1]}'")
            align1 = "-" + align1
            align2 = word2[j-1] + align2
            stats['insertions'] += 1
            j -= 1
    path.append((0,0))

    return {
        'distance': int(dp[m, n]),
        'dp_matrix': dp.tolist(),
        'path': path,
        'operations': operations[::-1],
        'aligned_word1': align1,
        'aligned_word2': align2,
        **stats
    }

# Restored edit distance API endpoint with full logic
@app.route("/api/edit-distance/calculate", methods=["POST"])
def api_edit_distance():
    try:
        data = request.get_json()
        word1, word2 = data['word1'], data['word2']
        results = calculate_levenshtein_with_traceback(word1, word2)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# N-gram Probability API
@app.route("/api/ngram/probability", methods=["POST"])
def api_ngram_probability():
    try:
        data = request.get_json()
        corpus = data.get("corpus", "").lower()
        n = data.get("n", 2)
        phrase = data.get("phrase", "").lower()

        tokens = word_tokenize(corpus)
        vocab_size = len(set(tokens))
        
        n_grams = list(ngrams(tokens, n))
        n_minus_1_grams = list(ngrams(tokens, n - 1))
        
        n_gram_counts = Counter(n_grams)
        n_minus_1_gram_counts = Counter(n_minus_1_grams)

        test_tokens = word_tokenize(phrase)
        if len(test_tokens) < n:
             return jsonify({"error": f"Phrase must be at least {n} words long to calculate {n}-gram probability."}), 400

        test_n_gram = tuple(test_tokens[-(n):])
        prefix = test_n_gram[:-1]

        numerator = n_gram_counts[test_n_gram] + 1
        denominator = n_minus_1_gram_counts[prefix] + vocab_size
        
        prob = numerator / denominator
        
        explanation = f"Calculating the probability of the last word '{test_n_gram[-1]}' given the previous {n-1} word(s) '{' '.join(prefix)}'.<br>"
        explanation += "Using Add-1 (Laplace) smoothing to handle unseen n-grams.<br><br>"
        explanation += f"P({test_n_gram[-1]} | {' '.join(prefix)}) = (Count({', '.join(test_n_gram)}) + 1) / (Count({', '.join(prefix)}) + Vocabulary Size)<br>"
        explanation += f" = ({n_gram_counts[test_n_gram]} + 1) / ({n_minus_1_gram_counts[prefix]} + {vocab_size}) = {prob:.4f}"
        
        return jsonify({ "probability": prob, "explanation": explanation })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Perplexity API
@app.route("/api/perplexity", methods=["POST"])
def api_perplexity():
    try:
        data = request.get_json()
        corpus = data.get("corpus", "").lower()
        sentence = data.get("sentence", "").lower()
        n = 2 # Using bigrams for perplexity calculation

        # Train model
        tokens = word_tokenize(corpus)
        n_grams = list(ngrams(tokens, n))
        n_minus_1_grams = list(ngrams(tokens, n - 1))
        n_gram_counts = Counter(n_grams)
        n_minus_1_gram_counts = Counter(n_minus_1_grams)
        vocab_size = len(set(tokens))

        # Calculate perplexity
        test_tokens = word_tokenize(sentence)
        test_n_grams = list(ngrams(test_tokens, n))
        
        if not test_n_grams:
            return jsonify({"perplexity": "Infinity"})

        log_prob_sum = 0
        for ng in test_n_grams:
            prefix = ng[:-1]
            numerator = n_gram_counts[ng] + 1
            denominator = n_minus_1_gram_counts[prefix] + vocab_size
            prob = numerator / denominator
            if prob == 0:
                log_prob_sum = float('-inf')
                break
            log_prob_sum += math.log2(prob)

        if log_prob_sum == float('-inf'):
            return jsonify({"perplexity": "Infinity"})

        perplexity = 2 ** (-log_prob_sum / len(test_n_grams))
        
        return jsonify({"perplexity": perplexity})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# POS Tagging API
@app.route("/api/pos-tag", methods=["POST"])
def api_pos_tag():
    try:
        data = request.get_json()
        sentence = data.get("sentence", "")
        tokens = word_tokenize(sentence)
        tags = nltk.pos_tag(tokens)
        
        detailed_tags = []
        for word, tag in tags:
            # FIX: Use the internal dictionary for descriptions
            description = POS_TAG_DESCRIPTIONS.get(tag, "N/A")
            detailed_tags.append({"word": word, "tag": tag, "description": description})

        return jsonify({"tags": detailed_tags})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# NER API
@app.route("/api/ner", methods=["POST"])
def api_ner():
    try:
        data = request.get_json()
        text = data.get("text", "")
        doc = nlp_spacy(text)
        
        entities = []
        for token in doc:
            tag = f"{token.ent_iob_}-{token.ent_type_}" if token.ent_type_ else "O"
            
            description = ""
            if token.ent_type_:
                base_desc = spacy.explain(token.ent_type_)
                if token.ent_iob_ == 'B':
                    description = f"Beginning of a '{base_desc}' entity."
                elif token.ent_iob_ == 'I':
                    description = f"Inside a '{base_desc}' entity."
            else:
                description = "Outside any named entity."

            entities.append({"word": token.text, "tag": tag, "description": description})
            
        return jsonify({"entities": entities})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Restored Sequence Alignment API
@app.route("/api/sequence-alignment/align", methods=["POST"])
def api_sequence_alignment():
    try:
        data = request.get_json()
        seq1, seq2 = data['seq1'], data['seq2']
        match_score, mismatch_penalty, gap_penalty = 1, 0, 0
        
        m, n = len(seq1), len(seq2)
        dp = np.zeros((m + 1, n + 1), dtype=int)
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                score = match_score if seq1[i-1] == seq2[j-1] else mismatch_penalty
                dp[i,j] = max(dp[i-1, j-1] + score,
                              dp[i-1, j] + gap_penalty,
                              dp[i, j-1] + gap_penalty)
        
        align1, align2, comparison = "", "", ""
        i, j = m, n
        while i > 0 or j > 0:
            score = match_score if i > 0 and j > 0 and seq1[i-1] == seq2[j-1] else mismatch_penalty
            
            if i > 0 and j > 0 and dp[i,j] == dp[i-1,j-1] + score:
                align1 = seq1[i-1] + align1
                align2 = seq2[j-1] + align2
                comparison = ("|" if score == match_score else " ") + comparison
                i -= 1
                j -= 1
            elif i > 0 and dp[i,j] == dp[i-1,j] + gap_penalty:
                align1 = seq1[i-1] + align1
                align2 = "-" + align2
                comparison = " " + comparison
                i -= 1
            else:
                align1 = "-" + align1
                align2 = seq2[j-1] + align2
                comparison = " " + comparison
                j -= 1

        return jsonify({
            'aligned_seq1': align1,
            'aligned_seq2': align2,
            'comparison': comparison,
            'score': int(dp[m, n])
        })
    except Exception as e:
        app.logger.error(f"Alignment error: {e}")
        return jsonify({"error": str(e)}), 400

# Word Sense Disambiguation API
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'): return wn.ADJ
    elif treebank_tag.startswith('V'): return wn.VERB
    elif treebank_tag.startswith('N'): return wn.NOUN
    elif treebank_tag.startswith('R'): return wn.ADV
    else: return None

@app.route("/api/wsd/disambiguate", methods=["POST"])
def api_wsd_disambiguate():
    try:
        data = request.get_json()
        sentence = data.get("sentence", "")
        
        tokens = word_tokenize(sentence)
        tagged_sent = pos_tag(tokens)
        lemmatizer = WordNetLemmatizer()
        
        results = []
        for word, tag in tagged_sent:
            wn_pos = get_wordnet_pos(tag)
            if not wn_pos: continue

            lemma = lemmatizer.lemmatize(word, pos=wn_pos)
            synsets = wn.synsets(lemma, pos=wn_pos)

            if not synsets: continue

            all_senses = []
            for sense in synsets:
                all_senses.append({"name": sense.name(), "definition": sense.definition()})
            
            results.append({
                "word": word, 
                "pos": tag, 
                "senses_count": len(synsets),
                "all_senses": all_senses
            })
            
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Sparse Vector API
@app.route("/api/sparse-vectors", methods=["POST"])
def api_sparse_vectors():
    try:
        data = request.get_json()
        corpus = data.get("corpus", "").split('\n')
        
        # Bag-of-Words
        count_vectorizer = CountVectorizer()
        bow_matrix = count_vectorizer.fit_transform(corpus)
        
        # TF-IDF
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
        
        # Get vocabulary
        vocab = count_vectorizer.get_feature_names_out().tolist()
        
        return jsonify({
            "vocabulary": vocab,
            "bow": bow_matrix.toarray().tolist(),
            "tfidf": tfidf_matrix.toarray().tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Sparse Vector Lab API
@app.route("/api/sparse-vectors/lab", methods=["POST"])
def api_sparse_vectors_lab():
    try:
        data = request.get_json()
        query = data.get("query", "").lower()
        query_terms = word_tokenize(query)

        # Data from the lab image
        tf_data = {
            'car': [27, 4, 24],
            'auto': [3, 33, 0],
            'insurance': [0, 33, 29],
            'best': [14, 0, 17]
        }
        docs = ['Doc1', 'Doc2', 'Doc3']
        tf = pd.DataFrame(tf_data, index=docs)

        idf_data = {
            'car': 1.65,
            'auto': 2.08,
            'insurance': 1.62,
            'best': 1.5
        }
        idf = pd.Series(idf_data)

        # Calculate TF-IDF
        tfidf = tf.copy()
        for term in tfidf.columns:
            tfidf[term] = tf[term] * idf[term]

        # Score documents based on the query
        scores = {}
        for doc in docs:
            score = 0
            for term in query_terms:
                if term in tfidf.columns:
                    score += tfidf.loc[doc, term]
            scores[doc] = score
        
        return jsonify({
            "tfidf_weights": tfidf.to_dict('index'),
            "query_scores": scores
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Dense Vector Lab API
@app.route("/api/dense-vectors/lab", methods=["POST"])
def api_dense_vectors_lab():
    try:
        data = request.get_json()
        corpus_text = data.get("corpus", "")
        target_word = data.get("target_word", "").lower()
        model_type = data.get("model_type", "cbow") # cbow or skipgram

        # Preprocess the corpus
        sentences = [word_tokenize(sent.lower()) for sent in corpus_text.split('\n')]
        
        # Train Word2Vec model
        sg = 1 if model_type == 'skipgram' else 0
        model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, sg=sg)
        
        # Find most similar words
        if target_word in model.wv:
            similar_words = model.wv.most_similar(target_word)
        else:
            return jsonify({"error": f"Word '{target_word}' not in vocabulary."}), 400

        return jsonify({
            "similar_words": similar_words
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Sparse Vector Application (Classification) API
@app.route("/api/sparse-vectors/classify", methods=["POST"])
def api_sparse_vectors_classify():
    try:
        data = request.get_json()
        training_data_raw = data.get("training_data", "")
        query = data.get("query", "")

        # Parse training data
        lines = training_data_raw.strip().split('\n')
        labels = []
        texts = []
        for line in lines:
            parts = line.split(',', 1)
            if len(parts) == 2:
                labels.append(parts[0].strip())
                texts.append(parts[1].strip())

        if not texts or not labels:
            return jsonify({"error": "Invalid or empty training data."}), 400

        # Create a simple classification pipeline
        model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        
        # Train the model
        model.fit(texts, labels)
        
        # Predict the query
        prediction = model.predict([query])[0]

        return jsonify({
            "prediction": prediction
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Dense Vector Application (Semantic Search) API
@app.route("/api/dense-vectors/search", methods=["POST"])
def api_dense_vectors_search():
    try:
        data = request.get_json()
        corpus_text = data.get("corpus", "")
        query = data.get("query", "")

        documents = [doc for doc in corpus_text.strip().split('\n') if doc]
        
        # Simple averaging of word vectors to get sentence embeddings
        tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
        tokenized_query = word_tokenize(query.lower())
        
        all_sentences = tokenized_docs + [tokenized_query]
        
        # Train a Word2Vec model on the provided documents + query
        model = Word2Vec(all_sentences, vector_size=100, window=5, min_count=1, workers=4)
        
        def get_sentence_vector(tokens):
            vectors = [model.wv[word] for word in tokens if word in model.wv]
            if len(vectors) == 0:
                return np.zeros(model.vector_size)
            return np.mean(vectors, axis=0)

        doc_vectors = np.array([get_sentence_vector(doc) for doc in tokenized_docs])
        query_vector = get_sentence_vector(tokenized_query).reshape(1, -1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, doc_vectors)[0]
        
        # Rank documents
        results = sorted(zip(documents, similarities), key=lambda item: item[1], reverse=True)
        
        return jsonify({
            "results": [{"doc": doc, "score": float(score)} for doc, score in results]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
