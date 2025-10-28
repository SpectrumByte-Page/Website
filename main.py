import pandas as pd
import random
import time
from sentence_transformers import SentenceTransformer, util

# ========== CONFIGURATION ==========
MODEL_NAME = "all-MiniLM-L6-v2"
DATASET_PATH = "customer_service_hp_dataset.csv"  # Ganti sesuai dataset kamu
THRESHOLD = 0.60  # toleransi kesamaan (lebih fleksibel biar tangkep maksud user)

# ========== LOAD MODEL & DATA ==========
print("🔧 Loading model dan dataset...")
model = SentenceTransformer(MODEL_NAME)
data = pd.read_csv(DATASET_PATH)
faq_questions = data["question"].tolist()
faq_answers = data["answer"].tolist()
faq_topics = data["topic"].tolist()

faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)
print(f"✅ Dataset loaded ({len(faq_questions)} entries)")

# ========== PREDEFINED RESPONSES ==========
fallback_responses = [
    "Hmm, aku belum yakin maksudnya Kak 😅 bisa dijelasin sedikit lebih detail?",
    "Aku belum nangkep sepenuhnya, Kak. Boleh dijelasin ulang biar aku bantu lebih tepat ya 😊",
    "Kayaknya perlu konteks tambahan deh, Kak. Ceritain dikit lagi dong ✨",
]

greetings = [
    "Halo 👋, selamat datang di *SpectrumByte Service Center*! Ada yang bisa aku bantu hari ini? 💬",
    "Hai Kak! 🙌 Gimana kabarnya? Lagi ada kendala di HP atau laptop ya?",
    "Selamat datang di *SpectrumByte CS* 📱💻 Siap bantuin masalah device kamu nih 😄",
]

goodbyes = [
    "Makasih banyak udah ngobrol sama aku hari ini 🙏 semoga HP/Laptop-nya cepet beres ya!",
    "Seneng bisa bantu Kak 🤝 Jangan sungkan hubungi kami lagi kapan aja 💬",
    "Oke Kak, jaga gadget-nya baik-baik ya ✨ Sampai jumpa lagi!",
]

empathetic_replies = [
    "Waduh, pasti nyebelin banget ya 😢",
    "Aduh, maaf banget ya Kak 🙏 semoga cepet kelar deh",
    "Pasti capek banget ya nunggu 😔 tenang, kita bantu semaksimal mungkin ya 💪",
]

# Intent dasar
intents = {
    "salam": ["hai", "halo", "assalamualaikum", "pagi", "siang", "sore", "malam"],
    "terimakasih": ["makasih", "terimakasih", "thanks", "thank you"],
    "goodbye": ["bye", "dadah", "sampai jumpa", "quit", "exit"],
    "curhat": ["cape", "kesel", "stress", "stres", "sedih", "bete", "nyerah", "frustasi", "frustrasi"]
}

# ========== FUNCTIONS ==========
def detect_intent(user_input):
    """Deteksi intent dasar dari kata kunci"""
    text = user_input.lower()
    for intent, keywords in intents.items():
        if any(kw in text for kw in keywords):
            return intent
    return None


def simulate_typing(delay=0.3):
    """Simulasi bot mengetik"""
    for d in "...":
        print(d, end="", flush=True)
        time.sleep(delay)
    print()


def contextual_response(user_input, chat_history):
    """Analisis konteks obrolan terakhir untuk menyesuaikan nada"""
    if len(chat_history) == 0:
        return ""
    last_topic = chat_history[-1]["topic"]
    if "Garansi" in last_topic and "garansi" in user_input.lower():
        return "Oh iya, ngomong-ngomong soal garansi, udah sempet kirim bukti pembeliannya ke WhatsApp kami belum Kak? 😊"
    elif "Service Lama" in last_topic and "kapan" in user_input.lower():
        return "Aku ngerti banget kalau nunggu itu bikin gak sabar 😅 tapi biar cepet aku bantu follow up ya~"
    elif "Curhat" in last_topic or "cape" in user_input.lower():
        return "Hehe, gapapa Kak curhat aja dulu, aku dengerin kok 🤗"
    return ""


def get_answer(user_question, chat_history):
    """Cari jawaban paling mirip dari dataset dengan mempertimbangkan konteks"""
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    similarities = util.cos_sim(user_embedding, faq_embeddings)[0]
    best_match = similarities.argmax().item()
    score = similarities[best_match].item()

    if score >= THRESHOLD:
        base_answer = faq_answers[best_match]
        topic = faq_topics[best_match]
    else:
        base_answer = random.choice(fallback_responses)
        topic = "Fallback"

    # Tambah konteks dari obrolan sebelumnya
    context_extra = contextual_response(user_question, chat_history)
    empathetic = random.choice(empathetic_replies) if random.random() < 0.4 else ""

    return f"{empathetic} {base_answer} {context_extra}".strip(), topic


# ========== MAIN CHAT LOOP ==========
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
            print("Bot: Sama-sama Kak 😊 seneng bisa bantu!")
            continue

        elif intent == "curhat":
            simulate_typing()
            print("Bot: 😢 Aduh, gapapa Kak. Kadang emang capek sih kalo device rusak terus. Ceritain aja dulu, aku dengerin kok 💬")
            continue

        simulate_typing()
        answer, topic = get_answer(user_input, chat_history)
        chat_history.append({"user": user_input, "bot": answer, "topic": topic})
        print("Bot:", answer)


if __name__ == "__main__":
    chat()
