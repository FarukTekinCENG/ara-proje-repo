import pandas as pd
import json
import matplotlib.pyplot as plt
from itertools import cycle
import os

# Excel dosyasını oku
df = pd.read_excel("results/results.xlsx")

# metrics sütununu dict olarak al
df['metrics_dict'] = df['metrics'].apply(json.loads)

# Kayıt dizini
output_dir = "graphs"
os.makedirs(output_dir, exist_ok=True)

# Grafikleri saklamak için liste
graphs = []
current_graph = []

# Veriyi gruplama: base_classifier ile yeni grafiğe başla
for _, row in df.iterrows():
    if row['method'] == 'base_classifier':
        if current_graph:
            graphs.append(current_graph)
        current_graph = []
    current_graph.append({
        'method': row['method'],
        'model_name': row['model_name'],
        'train_data_size': row['train_data_size'],
        'data_size': row['data_size'],
        'accuracy': row['metrics_dict'].get('accuracy') * 100 if row['metrics_dict'].get('accuracy') else None,
        'avg_uncertainty': row['metrics_dict'].get('avg_uncertainty')
    })

if current_graph:
    graphs.append(current_graph)

# Grafikleri çiz ve kaydet
for idx, graph_data in enumerate(graphs, 1):
    x_values = list(range(len(graph_data)))
    
    # Renk döngüsü diğer yöntemler için
    colors = cycle(['green', 'blue'])
    
    # Accuracy grafiği
    plt.figure(figsize=(12,5))
    used_labels = set()
    for i, entry in enumerate(graph_data):
        if entry['method'] == 'base_classifier':
            color = 'red'
        else:
            color = next(colors)
        plt.scatter(x_values[i], entry['accuracy'], color=color, s=80, edgecolor='black', zorder=3)
        
        # Legend etiketi oluştur
        label = f"{entry['model_name']} | method={entry['method']} | train_data_size={entry['train_data_size']} | data_size={entry['data_size']}"
        if label not in used_labels:
            plt.scatter([], [], color=color, label=label)  # boş scatter ile legend ekle
            used_labels.add(label)
    
    # Noktaları birleştiren siyah çizgi
    y_values = [entry['accuracy'] for entry in graph_data]
    plt.plot(x_values, y_values, color='black', linewidth=1, zorder=1)
    
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Accuracy Plot - Graph {idx}')
    plt.legend(loc='best', fontsize=8)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'graph-{idx}-accuracy.jpeg'))
    plt.close()
    
    # Avg Uncertainty grafiği
    plt.figure(figsize=(12,5))
    colors = cycle(['green', 'blue'])
    used_labels = set()
    for i, entry in enumerate(graph_data):
        if entry['method'] == 'base_classifier':
            color = 'red'
        else:
            color = next(colors)
        plt.scatter(x_values[i], entry['avg_uncertainty'], color=color, s=80, edgecolor='black', zorder=3)
        
        # Legend etiketi
        label = f"{entry['model_name']} | method={entry['method']} | train_data_size={entry['train_data_size']} | data_size={entry['data_size']}"
        if label not in used_labels:
            plt.scatter([], [], color=color, label=label)
            used_labels.add(label)
    
    y_values = [entry['avg_uncertainty'] for entry in graph_data]
    plt.plot(x_values, y_values, color='black', linewidth=1, zorder=1)
    
    plt.xlabel('Iteration')
    plt.ylabel('Average Uncertainty')
    plt.title(f'Avg Uncertainty Plot - Graph {idx}')
    plt.legend(loc='best', fontsize=8)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'graph-{idx}-uncertainty.jpeg'))
    plt.close()
