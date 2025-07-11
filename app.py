from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS

# ✅ Load & Prepare Dataset
df = pd.read_csv("https://drive.google.com/uc?id=1jVmVG960OoBLxewXtJ9tleU_HF4_jKBY")
df = df.rename(columns={
    'ProductID': 'product_id',
    'ProductName': 'productname',
    'ProductBrand': 'productbrand',
    'Gender': 'gender',
    'Price (INR)': 'price',
    'PrimaryColor': 'primary_color'
})

df = df[['product_id', 'productname', 'productbrand', 'gender', 'price', 'primary_color']].dropna()
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df.dropna(inplace=True)

# ✅ Simulate "on sale" status
df['is_sale'] = np.random.choice([True, False], size=len(df), p=[0.6, 0.4])
df['discount_price'] = df.apply(
    lambda row: row['price'] * np.random.uniform(0.5, 0.9) if row['is_sale'] else row['price'], axis=1
)

# ✅ Filter only sale items
df = df[df['is_sale'] == True].reset_index(drop=True)

# ✅ TF-IDF Content-Based Setup
df['text'] = df['productname'].fillna('') + ' ' + df['productbrand'].fillna('')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_mat = tfidf.fit_transform(df['text'])
cos_sim = cosine_similarity(tfidf_mat, tfidf_mat)
indices = pd.Series(df.index, index=df['product_id'])

# ✅ Clustering Setup
features = pd.get_dummies(df[['gender', 'primary_color']])
features['price'] = df['discount_price']
scaled = StandardScaler().fit_transform(features)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(scaled)

# ✅ Home Route for Testing
@app.route("/")
def home():
    return "<h2>🛍️ Welcome to the Product Recommendation API!<br>Try <code>/recommend?keyword=bag</code> in the URL.</h2>"

# ✅ Recommendation Route
@app.route('/recommend', methods=['GET'])
def recommend():
    keyword = request.args.get('keyword', '').lower().strip()

    matched_products = df[df['text'].str.lower().str.contains(keyword)]

    if matched_products.empty:
        tfidf_keyword = tfidf.transform([keyword])
        sim_scores = cosine_similarity(tfidf_keyword, tfidf_mat)[0]
        best_idx = sim_scores.argmax()
        matched_product = df.iloc[best_idx]
    else:
        matched_product = matched_products.iloc[0]

    idx = matched_product.name
    gender_target = matched_product['gender'].lower()
    recs = []

    # 🔹 Content-Based Recommendations
    sim_scores = list(enumerate(cos_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:15]
    for i, _ in sim_scores:
        item = df.iloc[i]
        if item['gender'].lower() == gender_target and item['product_id'] != matched_product['product_id']:
            recs.append({
                "type": "content",
                "product": item['productname'],
                "brand": item['productbrand'],
                "discount_price": int(item['discount_price']),
                "original_price": int(item['price'])
            })
        if len([r for r in recs if r['type'] == 'content']) >= 3:
            break

    # 🔹 Cluster-Based Recommendations
    cluster_id = matched_product['cluster']
    cluster_df = df[(df['cluster'] == cluster_id) & (df['product_id'] != matched_product['product_id'])]
    cluster_df = cluster_df[cluster_df['gender'].str.lower() == gender_target]

    if len(cluster_df) > 0:
        cluster_sample = cluster_df.sample(min(3, len(cluster_df)), random_state=42)
        for _, row in cluster_sample.iterrows():
            recs.append({
                "type": "cluster",
                "product": row['productname'],
                "brand": row['productbrand'],
                "discount_price": int(row['discount_price']),
                "original_price": int(row['price'])
            })

    return jsonify({
        "search_keyword": keyword,
        "matched_product": {
            "product": matched_product['productname'],
            "brand": matched_product['productbrand'],
            "original_price": int(matched_product['price']),
            "discount_price": int(matched_product['discount_price']),
        },
        "recommendations": recs
    })

# ✅ Run the app
if __name__ == '__main__':
    app.run(debug=True)
