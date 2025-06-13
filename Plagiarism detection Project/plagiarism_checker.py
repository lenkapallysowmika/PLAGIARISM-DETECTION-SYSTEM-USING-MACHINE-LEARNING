
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Get all .txt files in the directory
student_files = [doc for doc in os.listdir() if doc.endswith('.txt')]

# Read file contents
student_notes = [open(file, encoding='utf-8').read() for file in student_files]

# Vectorize the text documents using TF-IDF
vectorize = lambda Text: TfidfVectorizer().fit_transform(Text).toarray()
vectors = vectorize(student_notes)

# Function to compute cosine similarity between two vectors
similarity = lambda doc1, doc2: cosine_similarity([doc1, doc2])

# Pair each filename with its corresponding vector
s_vectors = list(zip(student_files, vectors))

# Function to check plagiarism between all document pairs
def check_plagiarism():
    plagiarism_results = set()
    global s_vectors
    for student_a, text_vector_a in s_vectors:
        new_vectors = s_vectors.copy()
        current_index = new_vectors.index((student_a, text_vector_a))
        del new_vectors[current_index]
        for student_b, text_vector_b in new_vectors:
            sim_score = similarity(text_vector_a, text_vector_b)[0][1]
            student_pair = sorted((student_a, student_b))
            score = (student_pair[0], student_pair[1], sim_score)
            plagiarism_results.add(score)
    return plagiarism_results

# Print the results
for data in check_plagiarism():
    print(data)
