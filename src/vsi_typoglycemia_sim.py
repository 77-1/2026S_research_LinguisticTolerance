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

def calculate_vsi(orig, shuff, alpha=0.5):
    n = len(orig)
    if n <= 2: return 0.0
    head_p = (orig[0] == shuff[0])
    tail_p = (orig[-1] == shuff[-1])
    p_b = 0.0 if (head_p and tail_p) else alpha
    
    total_dist = 0
    used = set()
    for i, c in enumerate(list(shuff)):
        m = [j for j, orig_c in enumerate(list(orig)) if orig_c == c and j not in used]
        if m:
            closest = min(m, key=lambda j: abs(j - i))
            used.add(closest)
            total_dist += abs(closest - i)
        else:
            total_dist += n
            
    vsi = (total_dist / (n - 1)) + p_b
    return round(vsi, 3)

def run_vsi_analysis():
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
            shuff = "".join(p)
            score = calculate_vsi(hira, shuff)
            is_ht_fixed = (hira[0] == shuff[0] and hira[-1] == shuff[-1])
            category = "語頭語末固定" if is_ht_fixed else "完全ランダム"
            
            all_results.append({
                "Original": raw,
                "Hiragana": hira,
                "Shuffled": shuff,
                "Category": category,
                "VSI": score
            })

    df = pd.DataFrame(all_results)
    
    # 1. 詳細CSV出力
    df.to_csv(os.path.join(output_dir, "vsi_results.csv"), index=False, encoding="utf-8-sig")
    
    # 2. Markdownドキュメント出力
    md_path = os.path.join(output_dir, "vsi_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Visual Shuffle Index (VSI) 全パターン解析レポート\n\n")
        for raw in words:
            hira = to_hiragana(raw)
            sub_df = df[df["Hiragana"] == hira].sort_values("VSI")
            f.write(f"### 単語: {raw} ({hira}) - 全 {len(sub_df)} パターン\n")
            f.write("| シャッフル後 | VSI スコア | 分類条件 |\n")
            f.write("| :--- | :---: | :--- |\n")
            for _, row in sub_df.iterrows():
                f.write(f"| {row['Shuffled']} | {row['VSI']:.3f} | {row['Category']} |\n")
            f.write("\n---\n\n")

    print(f"=== VSI 全パターン解析完了 ({len(df)}件) ===")
    print(f"- ドキュメント: output/vsi_report.md")
    print(f"- 詳細CSV: output/vsi_results.csv")

    # 3. 構造比較グラフ（箱ひげ図）
    plt.figure(figsize=(9, 6))
    fixed_scores = df[df["Category"] == "語頭語末固定"]["VSI"]
    random_scores = df[df["Category"] == "完全ランダム"]["VSI"]
    
    plt.boxplot([fixed_scores, random_scores], tick_labels=["語頭語末固定\n(視覚錯覚が起きやすい構造)", "完全ランダム"])
    plt.axhline(y=0.4, color='red', linestyle='--', label='VSI 許容限界線 (0.4)')
    plt.ylabel("VSI スコア")
    plt.title("文字順列の構造条件によるVSIスコア分布の明確差")
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "vsi_analysis_single.png"), dpi=300)
    plt.show()

if __name__ == "__main__":
    run_vsi_analysis()