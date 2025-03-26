from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
from pymongo import MongoClient
from celery import Celery
import numpy as np
from bson import ObjectId

app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')

# MongoDB Setup
client = MongoClient('mongodb+srv://websupport:RQHkN9PJJZ4uCHDP@cluster0.k0hjp.mongodb.net/bwebevents')
db = client['bwebevents']
collection = db['exhibitorparticipants']

# Celery Configuration (for async processing)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)


@app.route('/', methods=['GET'])
def home():
    return "Company Recommendation System is Running 🚀"


@app.route('/register', methods=['POST'])
def register():
    """Registers a new company and asynchronously updates recommendations."""
    data = request.get_json()
    
    if not data or "companyName" not in data or "companyProfile" not in data:
        return jsonify({"error": "companyName and companyProfile are required"}), 400

    # Generate and store vector
    company_vector = model.encode(data["companyProfile"]).tolist()
    data["vector"] = company_vector

    # Insert into MongoDB
    collection.insert_one(data)

    # Run recommendation update in the background
    update_recommendations_task.delay(data["companyName"])

    return jsonify({
        "message": "Company registered successfully!",
        "recommendations": compute_recommendations(data)
    }), 201


@app.route('/recommend', methods=['POST'])
def recommend():
    """Fetches recommendations for an existing company."""
    data = request.get_json()
    
    if "companyProfile" not in data:
        return jsonify({"error": "companyProfile is required"}), 400

    recommendations = compute_recommendations(data)

    return jsonify(recommendations[:5])


def compute_recommendations(company_data):
    """Computes recommendations and returns full company details, including _id."""
    user_keywords = set(company_data.get("businessKeywords", []))
    user_vector = np.array(company_data.get("vector") or model.encode(company_data.get("companyProfile", "")).tolist())

    all_companies = list(collection.find({"companyName": {"$ne": company_data["companyName"]}}))

    recommendations = []
    for company in all_companies:
        company_vector = np.array(company.get("vector") or model.encode(company.get("companyProfile", "")).tolist())
        bio_similarity = util.cos_sim(user_vector, company_vector).item()

        company_keywords = set(company.get("businessKeywords", []))
        common_keywords = len(user_keywords & company_keywords)
        max_keywords = max(len(user_keywords), len(company_keywords)) or 1
        keyword_score = common_keywords / max_keywords

        gives_score = 0.1 if company_data.get("gives") and company.get("gives") else 0
        top10_score = 0.1 if company_data.get("top10Customers") and company.get("top10Customers") else 0

        score = 0.4 * bio_similarity + 0.3 * keyword_score + gives_score + top10_score

        company["_id"] = str(company["_id"])
        company["country"] = str(company["country"])
        company["registeredByEventPartner"] = str(company["registeredByEventPartner"])
        company["exhibitionId"] = str(company["exhibitionId"])
        company["buisnessCategory"] = str(company["buisnessCategory"])
        company["bwebCategory"] = str(company["bwebCategory"])

        recommendations.append({"score": score, **company})

    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return recommendations[:5]  # Return top 5 recommendations


@celery.task
def update_recommendations_task(new_company_name):
    """Updates recommendations for all registered companies asynchronously."""
    new_company = collection.find_one({"companyName": new_company_name})
    
    if not new_company:
        return

    all_companies = list(collection.find({"companyName": {"$ne": new_company_name}}))

    for company in all_companies:
        recommendations = compute_recommendations(company)
        collection.update_one(
            {"companyName": company["companyName"]},
            {"$set": {"recommendations": recommendations}}
        )


if __name__ == '__main__':
    app.run(debug=True)
