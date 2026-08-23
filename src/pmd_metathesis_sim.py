import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from pykakasi import kakasi

plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Meiryo', 'Yu Gothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

kks = kakasi()

def to_hiragana(text):
    result = kks.convert(text)
    return "".join([item['hira'] for item in result])

KANA_MAP = {
    'あ': ('', 'a'), 'い': ('', 'i'), 'う': ('', 'u'), 'え': ('', 'e'), 'お': ('', 'o'),
    'か': ('k', 'a'), 'き': ('k', 'i'), 'く': ('k', 'u'), 'け': ('k', 'e'), 'こ': ('k', 'o'),
    'さ': ('s', 'a'), 'し': ('s', 'i'), 'す': ('s', 'u'), 'せ': ('s', 'e'), 'そ': ('s', 'o'),
    'た': ('t', 'a'), 'ち': ('t', 'i'), 'つ': ('t', 'u'), 'て': ('t', 'e'), 'と': ('t', 'o'),
    'な': ('n', 'a'), 'に': ('n', 'i'), 'ぬ': ('n', 'u'), 'ね': ('n', 'e'), 'の': ('n', 'o'),
    'は': ('h', 'a'), 'ひ': ('h', 'i'), 'ふ': ('h', 'u'), 'へ': ('h', 'e'), 'ほ': ('h', 'o'),
    'ま': ('m', 'a'), 'み': ('m', 'i'), 'む': ('m', 'u'), 'め': ('m', 'e'), 'も': ('m', 'o'),
    'ら': ('r', 'a'), 'り': ('r', 'i'), 'る': ('r', 'u'), 'れ': ('r', 'e'), 'ろ': ('r', 'o'),
    'ば': ('b', 'a'), 'び': ('b', 'i'), 'ぶ': ('b', 'u'), 'べ': ('b', 'e'), 'ぼ': ('b', 'o'),
    'ん': ('N', 'N'), 'ー': ('-', '-'), 'っ': ('Q', 'Q')
}

CONSONANT_GROUP = {
    'm': 1, 'b': 1, 'p': 1,
    't': 2, 'd': 2, 's': 2, 'z': 2, 'r': 2, 'n': 2,
    'k': 3, 'g': 3,
    '': 0, 'h': 0, 'N': 0, '-': 0, 'Q': 0
}

def calculate_pmd(orig_word, meta_word):
    orig_p = [KANA_MAP.get(c, ('?', '?')) for c in orig_word]
    meta_p = [KANA_MAP.get(c, ('?', '?')) for c in meta_word]
    
    min_l = min(len(orig_p), len(meta_p))
    diff_l = abs(len(orig_p) - len(meta_p))
    
    cost = 0.0
    for i in range(min_l):
        oc, ov = orig_p[i]
        mc, mv = meta_p[i]
        if ov != mv: cost += 1.0
        if oc != mc:
            g1, g2 = CONSONANT_GROUP.get(oc, 0), CONSONANT_GROUP.get(mc, 0)
            dist = 0.5 if (g1 == 0 or g2 == 0) else abs(g1 - g2) * 0.4
            cost += (0.4 + dist)
            
    cost += diff_l * 0.5
    return round(cost / max(len(orig_p), len(meta_p)), 3)

def run_pmd_analysis():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "..", "data", "words.txt")
    output_dir = os.path.join(base_dir, "..", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} が見つかりません。")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    all_results = []
    
    for raw in words:
        hira = to_hiragana(raw)
        chars = list(hira)
        permutations = set(itertools.permutations(chars))
        
        for p in permutations:
            meta = "".join(p)
            score = calculate_pmd(hira, meta)
            
            category = "起こりやすい音位転換" if score <= 0.8 else "不自然なランダム"
            
            all_results.append({
                "Original": raw,
                "Hiragana": hira,
                "Modified": meta,
                "Category": category,
                "PMD": score
            })

    df = pd.DataFrame(all_results)
    
    # 1. CSV
    df.to_csv(os.path.join(output_dir, "pmd_results.csv"), index=False, encoding="utf-8-sig")
    
    # 2. MD
    md_path = os.path.join(output_dir, "pmd_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Phonetic Metathesis Distance (PMD) 全パターン解析レポート\n\n")
        for raw in words:
            hira = to_hiragana(raw)
            sub_df = df[df["Hiragana"] == hira].sort_values("PMD")
            f.write(f"### 単語: {raw} ({hira}) - 全 {len(sub_df)} パターン\n")
            f.write("| シャッフル後 | PMD スコア | 分類条件 |\n")
            f.write("| :--- | :---: | :--- |\n")
            for _, row in sub_df.iterrows():
                f.write(f"| {row['Modified']} | {row['PMD']:.3f} | {row['Category']} |\n")
            f.write("\n---\n\n")

    print(f"=== PMD 全パターン解析完了 ({len(df)}件) ===")
    print(f"- ドキュメント: output/pmd_report.md")
    print(f"- 詳細CSV: output/pmd_results.csv")

    # 3. 構造比較グラフ（箱ひげ図）
    plt.figure(figsize=(9, 6))
    plausible_scores = df[df["Category"] == "起こりやすい音位転換"]["PMD"]
    random_scores = df[df["Category"] == "不自然なランダム"]["PMD"]
    
    plt.boxplot([plausible_scores, random_scores], tick_labels=["起こりやすい音位転換\n(自然に誤認・定着しうる領域)", "不自然なランダム"])
    plt.axhline(y=0.8, color='red', linestyle='--', label='PMD 許容限界線 (0.8)')
    plt.ylabel("PMD スコア")
    plt.title("音素シャッフルの構造条件によるPMDスコア分布の比較")
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "pmd_analysis_single.png"), dpi=300)
    plt.show()

if __name__ == "__main__":
    run_pmd_analysis()