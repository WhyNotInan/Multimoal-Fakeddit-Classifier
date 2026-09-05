import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set a professional visual style for presentation slides
sns.set_theme(style="whitegrid", context="talk")

print("Loading prototype data...")
df = pd.read_csv("prototype_5k.csv")

# Ensure the output directory for plots exists
os.makedirs("plots", exist_ok=True)

# Calculate word count (since we only did it in memory last time)
df['word_count'] = df['clean_title'].apply(lambda x: len(str(x).split()))

# ---------------------------------------------------------
# Plot 1: 2-Way Label Distribution (Bar Plot)
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
ax1 = sns.countplot(data=df, x='2_way_label', palette=['#2ecc71', '#e74c3c'])
plt.title('2-Way Classification Balance (0=Real, 1=Fake)', weight='bold')
plt.xlabel('Label')
plt.ylabel('Number of Posts')
# Add count labels on top of the bars
for p in ax1.patches:
    ax1.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/2_way_distribution.png', dpi=300)
plt.close()

# ---------------------------------------------------------
# Plot 2: 6-Way Label Distribution (Bar Plot)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
# 0: True, 1: Satire, 2: Misleading, 3: Imposter, 4: False Connection, 5: Manipulated
labels_map = {0: 'True', 1: 'Satire', 2: 'Misleading', 3: 'Imposter', 4: 'False Conn.', 5: 'Manipulated'}
df['6_way_name'] = df['6_way_label'].map(labels_map)

ax2 = sns.countplot(data=df, x='6_way_name', palette='viridis', 
                    order=['True', 'False Conn.', 'Misleading', 'Satire', 'Manipulated', 'Imposter'])
plt.title('Fine-Grained (6-Way) Fake News Categories', weight='bold')
plt.xlabel('Category')
plt.ylabel('Number of Posts')
plt.xticks(rotation=30)
for p in ax2.patches:
    ax2.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/6_way_distribution.png', dpi=300)
plt.close()

# ---------------------------------------------------------
# Plot 3: Word Count Frequency (Histogram)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.histplot(df['word_count'], bins=30, color='#3498db', kde=True)
plt.title('Distribution of Headline Word Counts', weight='bold')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('plots/word_count_histogram.png', dpi=300)
plt.close()

print("Success! Check the 'plots' folder for your presentation images.")
