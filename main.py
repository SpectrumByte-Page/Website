from flask import Flask, request, jsonify
import pandas as pd
import random
from sentence_transformers import SentenceTransformer, util

# Load model & data (sekali saat startup)
model = SentenceTransformer("all-MiniLM-L6-v2")
data = pd.read_csv("faq_dataset.csv")
faq_questions = data["question"].tolist()
faq_answers = data["answer"].tolist()
faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)

# Intent
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

def get_answer(user_question, threshold=0.65):
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    similarities = util.cos_sim(user_embedding, faq_embeddings)[0]
    best_match = similarities.argmax().item()
    score = similarities[best_match].item()
    if score >= threshold:
        return faq_answers[best_match]
    return random.choice([
        "Hmm, saya belum begitu paham maksud Anda. Bisa dijelaskan lebih detail ya? 🙂",
        "Boleh dijelaskan lebih lengkap supaya saya bisa bantu lebih tepat 👍",
        "Sepertinya saya kurang mengerti, bisa coba diulang dengan cara lain? 🤔",
    ])

# Flask app
app = Flask(__name__)

@app.route("/api/chat", methods=["POST"])
def chat_api():
    try:
        user_msg = request.json.get("message", "").strip()
        if not user_msg:
            return jsonify({"error": "Pesan tidak boleh kosong"}), 400

        intent = detect_intent(user_msg)
        if intent == "goodbye":
            reply = random.choice([
                "Terima kasih sudah menghubungi kami 🙏",
                "Senang bisa membantu 🤝, sampai jumpa!",
                "Kami siap membantu kapan saja 👍, have a nice day!",
            ])
        elif intent == "salam":
            reply = random.choice([
                "Halo 👋, selamat datang di SpectrumByte CS! Ada yang bisa kami bantu?",
                "Hai! 🙌 Terima kasih sudah menghubungi SpectrumByte, ada yang bisa kami perbaiki?",
                "Selamat datang di layanan SpectrumByte 📱💻, kami siap membantu Anda.",
            ])
        elif intent == "terimakasih":
            reply = "Sama-sama 😊 Senang bisa membantu!"
        else:
            reply = get_answer(user_msg)

        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "OK", "model": "all-MiniLM-L6-v2"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
