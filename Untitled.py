#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
from sklearn.neighbors import NearestNeighbors
from openai import OpenAI
import os

# ------------------------
# 1. Initialize OpenAI client securely
# ------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------------
# 2. Behavioral Knowledge Base
# ------------------------
kb_docs = [
    "Patients who delay visits until pain arises may be fear-avoidant. Acknowledge effort to show up and avoid guilt.",
    "Busy professionals who reference work need efficiency and control. Respect their time and offer clarity.",
    "Patients with fear of procedures appreciate gentle pacing and choice in how things proceed.",
    "Withdrawn or passive patients may need structured options and confirmation of consent to feel safe.",
]

# ------------------------
# 3. Clean smart quotes and corrupted characters
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
# 4. Get OpenAI Embedding
# ------------------------
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    )
    return response.data[0].embedding

# ------------------------
# 5. Get GPT-4 Communication Strategy
# ------------------------
def get_behavioral_guidance(note, context):
    system_prompt = """You are SynapseSmiles.AI, a behavioral intelligence assistant for dental professionals and students.
Your purpose is to help users understand patient behavior and tailor communication to improve trust, compliance, and treatment acceptance.
You are trained on principles from neuroscience, behavioral psychology, cognitive bias theory, motivational interviewing, and dental communication science.
When a user provides a description of a patient (e.g., age, tone, occupation, concern, reaction, etc.), you will:
- Identify possible behavioral traits (e.g., anxious avoider, logical thinker, present-biased)
- Recommend the best tone, framing, and language to use with that patient
- Suggest possible communication pitfalls to avoid
- Optionally, provide a sample conversation phrase or chairside script
You should not give medical advice or clinical diagnosis. Only focus on behavior, psychology, communication, and emotional insight. Always prioritize empathy and clarity.
"""

    user_prompt = f"""
Patient input:
{note}

Related behavioral insights:
- {context[0]}
- {context[1]}

What is the best tone and phrasing to use with this patient?
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# ------------------------
# 6. Streamlit UI
# ------------------------
st.title("🦷 SynapseSmiles.AI – Behavioral Dental Assistant")
st.write("Upload a `patient_notes.txt` file to generate patient-specific communication strategies using behavioral psychology and GPT-4.")

uploaded_file = st.file_uploader("📄 Upload patient_notes.txt", type=["txt"])

if uploaded_file:
    raw = uploaded_file.read().decode("utf-8", errors="replace")
    raw = clean_text(raw)

    # Split patient entries
    entries = re.split(r'\n\s*\n|(?<=\.”)\s*(?=\d{2})|(?<=\.”)\s*(?=Pt\s)', raw)
    entries = [e.strip().replace("\n", " ") for e in entries if e.strip()]

    with st.spinner("🔍 Analyzing notes and generating strategies..."):
        kb_embeddings = [get_embedding(doc) for doc in kb_docs]
        nn_model = NearestNeighbors(n_neighbors=2, metric='cosine')
        nn_model.fit(np.array(kb_embeddings))

        results = []
        for i, note in enumerate(entries, start=1):
            embed = get_embedding(note)
            _, indices = nn_model.kneighbors([embed])
            context = [kb_docs[j] for j in indices[0]]
            advice = get_behavioral_guidance(note, context)
            age_match = re.search(r'\b(\d{2})\b', note)
            age = int(age_match.group(1)) if age_match else None

            results.append({
                "patient_id": i,
                "age": age,
                "communication_strategy": advice
            })

        df = pd.DataFrame(results)
        st.success("✅ Done! See below:")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", data=csv, file_name="behavioral_communication_output.csv", mime="text/csv")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




