def chunk_text(text, size, overlap):
    words = text.split()
    step = size - overlap
    chunks = []
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i+ size]))
        if i + size>= len(words):
            break
    
    return chunks

text = "the cat sat on the mat near the door"
print(chunk_text(text, 4, 2))

text = ("Retrieval augmented generation grounds a language model by fetching relevant "
        "documents before it answers which reduces hallucination and lets the model use "
        "knowledge that was never part of its original training data at all")

print(chunk_text(text, size=10, overlap=3))