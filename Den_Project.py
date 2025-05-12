#!/usr/bin/env python
# coding: utf-8

# In[1]:


from openai import OpenAI
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import re
import unicodedata

# ------------------------
# 1. Initialize OpenAI client
# ------------------------
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# ------------------------


# In[ ]:





# In[ ]:


# 2. Clean smart quotes and fix encoding issues
# ------------------------
def clean_text(text):
    replacements = {
        'â€œ': '"', 'â€': '"',
        'â€˜': "'", 'â€™': "'",
        'â€”': "--", 'â€“': "-", 'â€¦': "...",
        'Â': '', 'Ã©': 'é'
    }
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)
    return unicodedata.normalize("NFKD", text).replace("�", "")

# ------------------------
# 3. Read and clean patient notes
# ------------------------
with open("patient_notes.txt", "r", encoding="utf-8", errors="replace") as f:
    raw = f.read()

raw = clean_text(raw)

# Split into individual entries
entries = re.split(r'\n\s*\n|(?<=\.”)\s*(?=\d{2})|(?<=\.”)\s*(?=Pt\s)', raw)
entries = [e.strip().replace("\n", " ") for e in entries if e.strip()]

# ------------------------
# 4. Behavioral Knowledge Base
# ------------------------
kb_docs = [
    "Patients who delay visits until pain arises may be fear-avoidant. Acknowledge effort to show up and avoid guilt.",
    "Busy professionals who reference work need efficiency and control. Respect their time and offer clarity.",
    "Patients with fear of procedures appreciate gentle pacing and choice in how things proceed.",
    "Withdrawn or passive patients may need structured options and confirmation of consent to feel safe.",
]


# In[ ]:





# In[2]:


# 2. Embedding Function            client.chat.completions.create         openai.Completion.create
# ------------------------
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    )
    return response.data[0].embedding

# Get embeddings for knowledge base
kb_embeddings = [get_embedding(doc) for doc in kb_docs]
kb_embeddings_np = np.array(kb_embeddings)

# Fit NearestNeighbors model
nn_model = NearestNeighbors(n_neighbors=2, metric='cosine')
nn_model.fit(kb_embeddings_np)

# ------------------------


# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:


# 5. Analyze and Collect Results
# ------------------------
results = []

for i, patient_note in enumerate(entries, start=1):
    query_embedding = get_embedding(patient_note)
    distances, indices = nn_model.kneighbors([query_embedding])
    retrieved_docs = [kb_docs[i] for i in indices[0]]

    # System prompt for SynapseSmiles.AI
    system_prompt = """You are SynapseSmiles.AI, a behavioral intelligence assistant for dental professionals and students.
    Your purpose is to help users understand patient behavior and tailor communication to improve trust, compliance, and treatment acceptance.
    You are trained on principles from neuroscience, behavioral psychology, cognitive bias theory, motivational interviewing, and dental communication science.
    When a user provides a description of a patient (e.g., age, tone, occupation, concern, reaction, etc.), you will:
    Identify possible behavioral traits (e.g., anxious avoider, logical thinker, present-biased)
    Recommend the best tone, framing, and language to use with that patient
    Suggest possible communication pitfalls to avoid
    Optionally, provide a sample conversation phrase or chairside script
    You should not give medical advice or clinical diagnosis. Only focus on behavior, psychology, communication, and emotional insight. Always prioritize empathy and clarity.
    """

    user_prompt = f"""
    Patient input:
    {patient_note}

    Related behavioral insights:
    - {retrieved_docs[0]}
    - {retrieved_docs[1]}

    What is the best tone and phrasing to use with this patient?
    """

    # Call GPT-4
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    communication_strategy = response.choices[0].message.content

    # Extract age from note (first number found)
    age_match = re.search(r'\b(\d{2})\b', patient_note)
    age = int(age_match.group(1)) if age_match else None

    results.append({
        "patient_id": i,
        "age": age,
        "communication_strategy": communication_strategy.strip()
    })

# ------------------------


# In[ ]:





# In[6]:


# 6. Export to CSV
# ------------------------
df = pd.DataFrame(results)
df.to_csv("behavioral_communication_output.csv", index=False)
print("✅ Exported to behavioral_communication_output.csv")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




