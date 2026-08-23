import streamlit as st
import itertools
import json
import os
import re
import pandas as pd

DATA_FILE = "app_data/search_history.json"

# --- ROMAJI / PHONEME MAPPING ---
KATAKANA_TO_ROMAJI = {
    'ア': 'a', 'イ': 'i', 'ウ': 'u', 'エ': 'e', 'オ': 'o',
    'カ': 'ka', 'キ': 'ki', 'ク': 'ku', 'ケ': 'ke', 'コ': 'ko',
    'サ': 'sa', 'シ': 'si', 'ス': 'su', 'セ': 'se', 'ソ': 'so',
    'タ': 'ta', 'チ': 'ti', 'ツ': 'tu', 'テ': 'te', 'ト': 'to',
    'ナ': 'na', 'ニ': 'ni', 'ヌ': 'nu', 'ネ': 'ne', 'ノ': 'no',
    'ハ': 'ha', 'ヒ': 'hi', 'フ': 'hu', 'ヘ': 'he', 'ホ': 'ho',
    'マ': 'ma', 'ミ': 'mi', 'ム': 'mu', 'メ': 'me', 'モ': 'mo',
    'ヤ': 'ya', 'ユ': 'yu', 'ヨ': 'yo',
    'ラ': 'ra', 'リ': 'ri', 'ル': 'ru', 'レ': 're', 'ロ': 'ro',
    'ワ': 'wa', 'ヲ': 'wo', 'ン': 'n',
    'ガ': 'ga', 'ギ': 'gi', 'グ': 'gu', 'ゲ': 'ge', 'ゴ': 'go',
    'ザ': 'za', 'ジ': 'zi', 'ズ': 'zu', 'ゼ': 'ze', 'ゾ': 'zo',
    'ダ': 'da', 'ヂ': 'di', 'ヅ': 'du', 'デ': 'de', 'ド': 'do',
    'バ': 'ba', 'ビ': 'bi', 'ブ': 'bu', 'ベ': 'be', 'ボ': 'bo',
    'パ': 'pa', 'ピ': 'pi', 'プ': 'pu', 'ペ': 'pe', 'ポ': 'po',
    'キャ': 'kya', 'キュ': 'kyu', 'キョ': 'kyo',
    'シャ': 'sya', 'シュ': 'syu', 'ショ': 'syo',
    'チャ': 'tya', 'チュ': 'tyu', 'チョ': 'tyo',
    'ニャ': 'nya', 'ニュ': 'nyu', 'ニョ': 'nyo',
    'ヒャ': 'hya', 'ヒュ': 'hyu', 'ヒョ': 'hyo',
    'ミャ': 'mya', 'ミュ': 'myu', 'ミョ': 'myo',
    'リャ': 'rya', 'リュ': 'ryu', 'リョ': 'ryo',
    'ギャ': 'gya', 'ギュ': 'gyu', 'ギョ': 'gyo',
    'ジャ': 'zya', 'ジュ': 'zyu', 'ジョ': 'zyo',
    'ビャ': 'bya', 'ビュ': 'byu', 'ビョ': 'byo',
    'ピャ': 'pya', 'ピュ': 'pyu', 'ピョ': 'pyo',
    'ッ': 'q', 'ー': '-'
}

CONSONANT_GROUPS = {
    'k': 'velar', 'g': 'velar',
    's': 'fricative', 'z': 'fricative',
    't': 'dental', 'd': 'dental',
    'n': 'nasal', 'm': 'nasal',
    'h': 'glottal', 'b': 'labial', 'p': 'labial',
    'r': 'liquid', 'w': 'glide', 'y': 'glide'
}

def katakana_to_romaji_str(text: str) -> str:
    i = 0
    res = []
    while i < len(text):
        if i + 1 < len(text) and text[i:i+2] in KATAKANA_TO_ROMAJI:
            res.append(KATAKANA_TO_ROMAJI[text[i:i+2]])
            i += 2
        elif text[i] in KATAKANA_TO_ROMAJI:
            res.append(KATAKANA_TO_ROMAJI[text[i]])
            i += 1
        else:
            res.append(text[i])
            i += 1
    return "".join(res)

def katakana_to_romaji_seq(text: str) -> list[str]:
    i = 0
    res = []
    while i < len(text):
        if i + 1 < len(text) and text[i:i+2] in KATAKANA_TO_ROMAJI:
            res.append(KATAKANA_TO_ROMAJI[text[i:i+2]])
            i += 2
        elif text[i] in KATAKANA_TO_ROMAJI:
            res.append(KATAKANA_TO_ROMAJI[text[i]])
            i += 1
        else:
            res.append(text[i])
            i += 1
    return res

def calculate_vsi(orig: str, shuffled: str, alpha: float) -> float:
    n = len(orig)
    if n <= 1:
        return 0.0
    
    boundary_penalty = 0.0
    if orig[0] != shuffled[0]:
        boundary_penalty += alpha
    if orig[-1] != shuffled[-1]:
        boundary_penalty += alpha
        
    pos_diff = sum(abs(i - shuffled.index(c)) for i, c in enumerate(orig)) / (n * n)
    return round(boundary_penalty + pos_diff, 3)

def calculate_pmd(orig: str, shuffled: str, c_same_cost: float, c_diff_cost: float, v_cost_param: float) -> float:
    orig_rom = katakana_to_romaji_seq(orig)
    shuffled_rom = katakana_to_romaji_seq(shuffled)
    
    if len(orig_rom) != len(shuffled_rom):
        return 9.99
    
    n = len(orig_rom)
    total_cost = 0.0
    
    for i in range(n):
        o_syl = orig_rom[i]
        s_syl = shuffled_rom[i]
        
        if o_syl == s_syl:
            continue
            
        o_c = o_syl[0] if o_syl[0] not in 'aeiou-' else ''
        s_c = s_syl[0] if s_syl[0] not in 'aeiou-' else ''
        
        o_v = o_syl[-1] if o_syl[-1] in 'aeiou-' else ''
        s_v = s_syl[-1] if s_syl[-1] in 'aeiou-' else ''
        
        c_cost = 0.0
        if o_c != s_c:
            g1 = CONSONANT_GROUPS.get(o_c, 'other')
            g2 = CONSONANT_GROUPS.get(s_c, 'other')
            c_cost = c_same_cost if g1 == g2 else c_diff_cost
            
        v_cost = 0.0 if o_v == s_v else v_cost_param
        total_cost += (c_cost + v_cost)
        
    return round(total_cost / n, 3)

def load_history():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_history(history):
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def process_word_analysis(target_word: str, alpha: float, c_same: float, c_diff: float, v_cost: float):
    chars = list(target_word)
    raw_perms = set("".join(p) for p in itertools.permutations(chars))
    valid_perms = [p for p in raw_perms if not p.startswith("ー") and p != target_word]
    
    results = []
    for p in valid_perms:
        v_score = calculate_vsi(target_word, p, alpha)
        p_score = calculate_pmd(target_word, p, c_same, c_diff, v_cost)
        romaji_p = katakana_to_romaji_str(p)
        
        is_fixed = (p[0] == target_word[0] and p[-1] == target_word[-1])
        disp_word = f"★ {p}" if is_fixed else p
        
        results.append({
            "word_vsi": disp_word,
            "vsi": v_score,
            "word_pmd": p,
            "alphabet": romaji_p,
            "pmd": p_score,
            "is_fixed": is_fixed
        })
    
    vsi_sorted = sorted(results, key=lambda x: x["vsi"])
    pmd_sorted = sorted(results, key=lambda x: x["pmd"])
    
    st.session_state["analyzed_word"] = target_word
    st.session_state["vsi_sorted"] = vsi_sorted
    st.session_state["pmd_sorted"] = pmd_sorted
    st.session_state["total_patterns"] = len(valid_perms)
    
    history = load_history()
    history[target_word] = {
        "length": len(target_word),
        "total_patterns": len(valid_perms),
        "vsi_sorted": vsi_sorted,
        "pmd_sorted": pmd_sorted
    }
    save_history(history)

# --- UI DESIGN (モバイル最適化) ---
st.set_page_config(page_title="ニンゲンの許容範囲シュミレーター", layout="wide")

st.title('ニンゲンの許容範囲"シュミレーター"')

# スマホ画面でも操作しやすいよう、設定パラメータをExpanderでメイン画面に配置
with st.expander("⚙️ 評価基準（重み）の変更・調整"):
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("**1. VSI（タイポグリセミア）**")
        alpha_param = st.slider(
            "語頭・語尾位置ずれペナルティ (α)",
            min_value=0.0, max_value=1.0, value=0.35, step=0.05
        )
    with col_p2:
        st.markdown("**2. PMD（メタセシス）**")
        c_same_param = st.slider(
            "同子音グループ間コスト",
            min_value=0.0, max_value=1.0, value=0.2, step=0.05
        )
        c_diff_param = st.slider(
            "異子音グループ間コスト",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05
        )
        v_cost_param = st.slider(
            "母音不一致コスト",
            min_value=0.0, max_value=1.0, value=0.3, step=0.05
        )

# 履歴データの読み込み
history_data = load_history()
history_keys = list(history_data.keys())

# サジェスト入力用 Form（スマホ表示時はボタンと入力欄が自然にフィット）
with st.form("input_form", clear_on_submit=False):
    col_input, col_btn = st.columns([3, 1])

    with col_input:
        word_input = st.selectbox(
            label="入力欄",
            options=history_keys,
            index=None,
            placeholder="全角カタカナ, 3文字以上の単語を入力してください。",
            label_visibility="collapsed",
            accept_new_options=True
        )

    with col_btn:
        run_btn = st.form_submit_button("実行", type="primary", use_container_width=True)

# 実行トリガー判定
if run_btn:
    if word_input:
        target_word = word_input.strip()
        is_katakana = bool(re.match(r'^[ァ-タチ-ヶー]+$', target_word))
        
        if not is_katakana:
            st.error("全角カタカナで入力してください。")
        elif len(target_word) < 3:
            st.warning("3文字以上の単語を入力してください。")
        else:
            st.session_state["show_detail"] = False
            process_word_analysis(target_word, alpha_param, c_same_param, c_diff_param, v_cost_param)
    else:
        st.warning("単語を入力してください。")
elif "analyzed_word" in st.session_state:
    process_word_analysis(
        st.session_state["analyzed_word"],
        alpha_param, c_same_param, c_diff_param, v_cost_param
    )

# 結果の描画エリア
if "vsi_sorted" in st.session_state and st.session_state["vsi_sorted"]:
    st.divider()
    
    analyzed_word = st.session_state.get("analyzed_word", "")
    st.markdown(f"### 解析結果: **{analyzed_word}**")
    
    vsi_list = st.session_state["vsi_sorted"]
    pmd_list = st.session_state["pmd_sorted"]
    
    col_vsi, col_pmd = st.columns(2)
    
    show_all = st.session_state.get("show_detail", False)
    table_height = 800 if show_all else 400
    display_vsi = vsi_list if show_all else vsi_list[:10]
    display_pmd = pmd_list if show_all else pmd_list[:10]
    
    # --- 1. タイポグリセミア表示 (VSI) ---
    with col_vsi:
        st.subheader("許容されやすいタイポグリセミア")
        
        df_vsi = pd.DataFrame([
            {"単語": item["word_vsi"], "VSI": item["vsi"]}
            for item in display_vsi
        ])
        st.dataframe(df_vsi, use_container_width=True, hide_index=True, height=table_height)
        st.caption("初めと終わりの文字が固定されている単語の前に★がついています。")

    # --- 2. メタセシス表示 (PMD) ---
    with col_pmd:
        st.subheader("おこりやすいメタセシス")
        
        df_pmd = pd.DataFrame([
            {"単語": item["word_pmd"], "Alphabet": item["alphabet"], "PMD": item["pmd"]}
            for item in display_pmd
        ])
        st.dataframe(df_pmd, use_container_width=True, hide_index=True, height=table_height)

    # --- 3. 詳細はコチラ ボタン ---
    st.write("")
    if st.button("閉じる" if show_all else "詳細はコチラ", use_container_width=True):
        st.session_state["show_detail"] = not show_all
        st.rerun()