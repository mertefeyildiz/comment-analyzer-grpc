# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 22:02:19 2024

@author: 90531
"""

import grpc
from concurrent import futures
import grpc_service_pb2
import grpc_service_pb2_grpc
from sentence_transformers import SentenceTransformer, util
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, BertTokenizer, BertForSequenceClassification
import pandas as pd
import torch

# Veri setlerini yükle
topics_df = pd.read_csv('topics.csv')
opinions_df = pd.read_csv('opinions.csv')

# Sentence similarity modeli (grupla için kullanılıyor)
similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# GPT-Neo modeli ve tokenizer'ı
gpt_model_name = "EleutherAI/gpt-neo-1.3B"
gpt_model = GPTNeoForCausalLM.from_pretrained(gpt_model_name)
gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)

# BERT sınıflandırıcı modeli ve tokenizer'ı
bert_model_name = "bert-base-uncased"
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=4)  # Claim, Counterclaim, Evidence, Rebuttal

# Modeli GPU'da çalıştırmak için ayar yapalım (CUDA varsa)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt_model.to(device)
bert_model.to(device)

# Yorumları sınıflandırma fonksiyonu (BERT)
def classify_opinion(opinion_text):
    inputs = bert_tokenizer(opinion_text, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = bert_model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).item()

    # Sınıf etiketleri
    labels = ["Claim", "Counterclaim", "Evidence", "Rebuttal"]
    
    # Sınıflandırılamayan yorumlar için "Unknown" ekleyelim
    try:
        return labels[predictions]
    except IndexError:
        return "Unknown"  # Eğer sınıflandırılamazsa "Unknown" dönecek

# GPT-Neo ile sonuç üretme fonksiyonu
def generate_conclusion(topic_text, claims, counterclaims, evidence, rebuttals):
    prompt = (
        f"Topic: {topic_text}\n"
        f"Supportive Claims: {claims}\n"
        f"Opposing Claims: {counterclaims}\n"
        f"Evidence: {evidence}\n"
        f"Rebuttals: {rebuttals}\n\n"
        "Generate a detailed and well-reasoned conclusion based on the arguments above."
    )

    inputs = gpt_tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = gpt_model.generate(
        inputs, max_length=450, num_return_sequences=1, no_repeat_ngram_size=2,
        top_k=50, top_p=0.9, attention_mask=torch.ones(inputs.shape, device=device),
        pad_token_id=gpt_tokenizer.eos_token_id
    )

    conclusion = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return conclusion.strip()

# GRPC Sunucusu için Servis Sınıfı
class CommentAnalyzerServicer(grpc_service_pb2_grpc.CommentAnalyzerServicer):
    def AnalyzeComment(self, request, context):
        # Gelen topic_id'ye göre topic_text ve ilgili opinionları bulalım
        topic_id = request.topic_id
        topic_row = topics_df[topics_df['topic_id'] == topic_id]

        # Eğer topic_id bulunmazsa hata döndür
        if topic_row.empty:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Topic ID '{topic_id}' bulunamadı.")
            return grpc_service_pb2.AnalyzeResponse(conclusion="")

        topic_text = topic_row['text'].values[0]

        # İlgili opinionları bulalım
        opinions = opinions_df[opinions_df['topic_id'] == topic_id]
        claims, counterclaims, evidence, rebuttals = [], [], [], []

        # Her bir opinion'ı BERT modeli ile sınıflandır
        for opinion in opinions['text']:
            classification = classify_opinion(opinion)
            if classification == "Claim":
                claims.append(opinion)
            elif classification == "Counterclaim":
                counterclaims.append(opinion)
            elif classification == "Evidence":
                evidence.append(opinion)
            elif classification == "Rebuttal":
                rebuttals.append(opinion)
            else:
                # Eğer sınıflandırılamıyorsa "Unknown" kategorisi olarak işaretle
                print(f"Opinion '{opinion}' sınıflandırılamadı, 'Unknown' kategorisine alındı.")

        # Boş kalan sınıflandırmaları kontrol edelim
        if not claims:
            claims = ["No claims found"]
        if not counterclaims:
            counterclaims = ["No counterclaims found"]
        if not rebuttals:
            rebuttals = ["No rebuttals found"]

        # GPT-Neo ile sonuç cümlesi üret
        conclusion = generate_conclusion(
            topic_text,
            claims=" ".join(claims),
            counterclaims=" ".join(counterclaims),
            evidence=" ".join(evidence),
            rebuttals=" ".join(rebuttals)
        )

        # Sonucu geri döndür
        return grpc_service_pb2.AnalyzeResponse(conclusion=conclusion)

# GRPC Sunucusunu başlat
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    grpc_service_pb2_grpc.add_CommentAnalyzerServicer_to_server(CommentAnalyzerServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("GRPC sunucusu başlatıldı.")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
