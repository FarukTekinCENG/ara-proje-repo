import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans


# ============================================================
# 1. Diversity Sampling: KMeans Tabancı Örnekleme
# ============================================================
def select_diverse_batch_kmeans(X_pool, n_samples=100, random_state=42):
    n_pool = X_pool.shape[0]
    if n_pool <= n_samples:
        return np.arange(n_pool)

    kmeans = KMeans(
        n_clusters=n_samples,
        random_state=random_state,
        n_init="auto"
    )
    kmeans.fit(X_pool)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    selected = []

    for cluster_id in range(n_samples):
        idxs = np.where(labels == cluster_id)[0]
        if len(idxs) == 0:
            continue

        pts = X_pool[idxs]
        center = centers[cluster_id]

        dist = np.linalg.norm(pts - center, axis=1)
        closest_idx = idxs[np.argmin(dist)]
        selected.append(closest_idx)

    return np.array(selected)



# ============================================================
# 2. Aktif Öğrenme Döngüsü
# ============================================================
def active_learning_diversity(
    X_labeled,
    y_labeled,
    X_unlabeled,
    y_unlabeled_true,
    X_val,
    y_val,
    batch_size=100,
    min_improvement=0.001,  # yani 0.1%
    max_iterations=100
):
    model = LogisticRegression(max_iter=1000)
    val_acc_list = []

    for it in range(1, max_iterations + 1):
        model.fit(X_labeled, y_labeled)

        pred = model.predict(X_val)
        acc = accuracy_score(y_val, pred)
        val_acc_list.append(acc)

        print(f"Iterasyon {it} - Validation Accuracy: {acc:.4f}")

        # İyileşme kontrolü
        if len(val_acc_list) >= 2:
            improvement = val_acc_list[-1] - val_acc_list[-2]
            print(f"  İyileşme: {improvement:.4f}")
            if improvement < min_improvement:
                print("  İyileşme yeterince büyük değil. Döngü durduruluyor.")
                break

        # Havuz bittiyse dur
        if X_unlabeled.shape[0] == 0:
            print("Etiketsiz havuz bitti.")
            break

        n_select = min(batch_size, X_unlabeled.shape[0])
        idx = select_diverse_batch_kmeans(X_unlabeled, n_samples=n_select)

        X_new = X_unlabeled[idx]
        y_new = y_unlabeled_true[idx]

        X_labeled = np.vstack([X_labeled, X_new])
        y_labeled = np.concatenate([y_labeled, y_new])

        mask = np.ones(X_unlabeled.shape[0], dtype=bool)
        mask[idx] = False
        X_unlabeled = X_unlabeled[mask]
        y_unlabeled_true = y_unlabeled_true[mask]

        print(f"  Seçilen yeni örnek sayısı: {n_select}")
        print(f"  Kalan unlabeled: {X_unlabeled.shape[0]}")

    return model, val_acc_list



# ============================================================
# 3. DATABASE BAĞLANTISI + VERİ ÇEKME + EMBEDDING + TRAINING
# ============================================================
if __name__ == "__main__":

    # -------------------------------
    # 3.1 PostgreSQL bağlantısı
    # -------------------------------
    engine = create_engine(
        "postgresql://postgres:123@localhost:5432/Active_Learning"
    )

    # -------------------------------
    # 3.2 Veriyi çek (sadece gerekli kolonlar)
    # -------------------------------
    query = """
    SELECT description, work_type
    FROM public.postings
    WHERE description IS NOT NULL
      AND work_type IS NOT NULL
    """

    print("Veritabanından veri çekiliyor...")
    df = pd.read_sql(query, engine)

    print("Toplam satır:", len(df))
    
    # -------------------------------
    # 3.3 Label encode work_type
    # -------------------------------
    le = LabelEncoder()
    y_all = le.fit_transform(df["work_type"])

    # -------------------------------
    # 3.4 Embedding çıkar
    # -------------------------------
    print("SentenceTransformer modeli yükleniyor...")
    emb = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding hesaplanıyor...")
    X_all = emb.encode(df["description"].tolist(), batch_size=64, show_progress_bar=True)


    # -------------------------------
    # 3.5 Train / Validation böl
    # -------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    # Train içinden küçük labeled subset oluştur
    X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
        X_train, y_train, test_size=0.98, random_state=42, stratify=y_train
    )

    print("Başlangıç labeled:", len(X_labeled))
    print("Başlangıç unlabeled:", len(X_unlabeled))
    print("Validation:", len(X_val))

    # -------------------------------
    # 3.6 Aktif Öğrenme Başlat
    # -------------------------------
    model, history = active_learning_diversity(
        X_labeled, y_labeled, X_unlabeled, y_unlabeled, X_val, y_val,
        batch_size=100,
        min_improvement=0.001
    )

    print("Accuracy gelişimi:", history)
