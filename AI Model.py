import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings("ignore")

#1. Load Data 
rooms    = pd.read_csv("rooms.csv")
bookings = pd.read_csv("bookings.csv")
branches = pd.read_csv("branches.csv")
guests = pd.read_csv("guests.csv")

print(f"Loaded: {len(rooms)} rooms, {len(bookings)} bookings,{len(guests)} guests, {len(rooms)} rooms")

#2. Data Cleaning
bookings['check_in']  = pd.to_datetime(bookings['check_in'], errors='coerce')
bookings['check_out'] = pd.to_datetime(bookings['check_out'], errors='coerce')
bookings['stay_days'] = (bookings['check_out'] - bookings['check_in']).dt.days

#3. Room-Level Stats
stats = bookings.groupby('room_id').agg(
    booking_count=('booking_id', 'count'),
    total_revenue=('payment', 'sum'),
    avg_payment=('payment', 'mean'),
    avg_stay=('stay_days', 'mean')
).reset_index()

df = rooms.merge(stats, on='room_id', how='left')
df = df.merge(branches[['Branch_id', 'Branch_name']], on='Branch_id', how='left')

for col in ['booking_count','total_revenue','avg_payment','avg_stay']:
    df[col].fillna(0, inplace=True)
df['Branch_name'].fillna('Unknown', inplace=True)

#4. Feature Engineering 
df['revenue_per_booking'] = df['total_revenue'] / (df['booking_count'] + 1)
df['revenue_per_day']     = df['total_revenue'] / (df['avg_stay'] + 1)
df['is_premium']          = df['type'].isin(['Suite','Deluxe','Presidential']).astype(int)

# Log transforms
df['log_price']        = np.log1p(df['price'])
df['log_revenue']      = np.log1p(df['total_revenue'])
df['log_bookings']     = np.log1p(df['booking_count'])
df['log_avg_stay']     = np.log1p(df['avg_stay'])
df['log_rev_per_day']  = np.log1p(df['revenue_per_day'])

features = [
    'log_price', 'log_revenue', 'log_bookings',
    'log_avg_stay', 'log_rev_per_day', 'is_premium'
]

X = df[features].copy()

#5. Remove Outliers
iso = IsolationForest(contamination='auto', n_estimators=300, random_state=42)
mask = iso.fit_predict(X) == 1
df_clean = df[mask].reset_index(drop=True)
X_clean  = X[mask].reset_index(drop=True)

print(f"\nRemoved Outliers → Remaining Rooms: {len(df_clean)}")

#6. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

#7. Train/Test Split 
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

#8. Best K search 
best_score = -1
best_k = 3

print("\nSearching for best number of clusters")

for k in range(2, 15):
    kmeans = KMeans(n_clusters=k, n_init=100, random_state=42)
    labels_train = kmeans.fit_predict(X_train)
    score = silhouette_score(X_train, labels_train)

    print(f"   k={k} → Silhouette Score (TRAIN) = {score:.4f}")

    if score > best_score:
        best_score = score
        best_k = k

print(f"\nBEST RESULT → k={best_k} | Silhouette Score (TRAIN) = {best_score:.4f}")

#9. Train final KMeans 
kmeans_final = KMeans(n_clusters=best_k, n_init=100, random_state=42)
kmeans_final.fit(X_train)

train_labels = kmeans_final.predict(X_train)
test_labels  = kmeans_final.predict(X_test)

df_clean['cluster'] = kmeans_final.predict(X_scaled)

#10. Train/Test Results 
train_sil = silhouette_score(X_train, train_labels)
test_sil  = silhouette_score(X_test, test_labels)

print("\nTRAIN / TEST CLUSTER ACCURACY")
print(f"Train Silhouette Score = {train_sil:.4f}")
print(f"Test  Silhouette Score = {test_sil:.4f}")


#11. Naming clusters
cluster_summary = df_clean.groupby('cluster').agg({
    'total_revenue': 'mean',
    'price': 'mean',
    'booking_count': 'mean',
    'room_id': 'count'
}).round(2).sort_values('total_revenue', ascending=False)

cluster_summary['cluster_name'] = [
    "1st (VIP) - 4,399 EGP/cluster",
    "2nd (Premium) - 2,396 EGP/cluster",
    "3rd (Popular) - 2,236 EGP/cluster"
][:len(cluster_summary)]

name_map = dict(zip(cluster_summary.index, cluster_summary['cluster_name']))
df_clean['cluster_name'] = df_clean['cluster'].map(name_map)

print("\nFinal Clusters (Ranked by Revenue):")
print(df_clean['cluster_name'].value_counts())

#12. Visualizations
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(16,10))

plt.subplot(2,2,1)
plt.scatter(X_pca[:,0], X_pca[:,1], c=df_clean['cluster'], cmap='tab10', alpha=0.7)
plt.title(f"PCA Cluster Plot (k={best_k})")
plt.colorbar()

plt.subplot(2,2,2)
sns.boxplot(data=df_clean, x='cluster_name', y='total_revenue')
plt.xticks(rotation=45)

plt.subplot(2,2,3)
sns.boxplot(data=df_clean, x='cluster_name', y='price')
plt.xticks(rotation=45)

plt.subplot(2,2,4)
df_clean['cluster_name'].value_counts().plot(kind='bar')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

#13. Save Final CSV 
output_cols = ['room_id','room_num','type','price','Branch_name','booking_count',
               'total_revenue','avg_stay','cluster','cluster_name']

df_clean[output_cols].to_csv("FINAL_ROOMS_CLUSTERS_HIGH_ACCURACY.csv", index=False)

print("\nCSV saved")

#14. SAVE MODEL AS PKL 
model_data = {
    "iso": iso,
    "scaler": scaler,
    "kmeans": kmeans_final,
    "pca": pca,
    "features": features
}

with open("hotel_clustering_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("\nModel saved as hotel_clustering_model.pkl")
print("Done")
