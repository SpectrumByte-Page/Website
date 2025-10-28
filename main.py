import pandas as pd
import random
import time
from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load dataset percakapan (CSV)
data = pd.read_csv("faq_dataset.csv")

faq_questions = data["question"].tolist()
faq_answers = data["answer"].tolist()

# Encode FAQ
faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)

# Variasi fallback
fallback_responses = [
    "Hmm, saya belum begitu paham maksud Anda. Bisa dijelaskan lebih detail ya? 🙂",
    "Boleh dijelaskan lebih lengkap supaya saya bisa bantu lebih tepat 👍",
    "Sepertinya saya kurang mengerti, bisa coba diulang dengan cara lain? 🤔",
]

# Variasi salam awal
greetings = [
    "Halo 👋, selamat datang di SpectrumByte CS! Ada yang bisa kami bantu?",
    "Hai! 🙌 Terima kasih sudah menghubungi SpectrumByte, ada yang bisa kami perbaiki?",
    "Selamat datang di layanan SpectrumByte 📱💻, kami siap membantu Anda.",
]

# Variasi penutup
goodbyes = [
    "Terima kasih sudah menghubungi kami 🙏",
    "Senang bisa membantu 🤝, sampai jumpa!",
    "Kami siap membantu kapan saja 👍, have a nice day!",
]

# Intent sederhana
intents = {
    "salam": ["hai", "halo", "assalamualaikum", "pagi", "siang", "sore", "malam"],
    "terimakasih": ["makasih", "terimakasih", "thanks", "thank you"],
    "goodbye": ["bye", "dadah", "sampai jumpa", "quit", "exit"]
}

def detect_intent(user_input):
    text = user_input.lower()
    for intent, keywords in intents.items():
        if any(kw in text for kw in keywords):
            return intent
    return None

def simulate_typing():
    dots = "..."
    for d in dots:
        print(d, end="", flush=True)
        time.sleep(0.4)
    print()

def get_answer(user_question, threshold=0.65):
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    similarities = util.cos_sim(user_embedding, faq_embeddings)[0]
    best_match = similarities.argmax().item()
    score = similarities[best_match].item()

    if score >= threshold:
        return faq_answers[best_match]
    else:
        return random.choice(fallback_responses)

# Chat loop
def chat():
    print("Bot:", random.choice(greetings))
    chat_history = []

    while True:
        user_input = input("Customer: ").strip()
        if not user_input:
            continue

        intent = detect_intent(user_input)

        if intent == "goodbye":
            simulate_typing()
            print("Bot:", random.choice(goodbyes))
            break
        elif intent == "salam":
            simulate_typing()
            print("Bot:", random.choice(greetings))
            continue
        elif intent == "terimakasih":
            simulate_typing()
            print("Bot: Sama-sama 😊 Senang bisa membantu!")
            continue

        simulate_typing()
        answer = get_answer(user_input)
        chat_history.append((user_input, answer))
        print("Bot:", answer)

if __name__ == "__main__":
    chat()