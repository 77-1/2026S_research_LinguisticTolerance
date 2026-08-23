import streamlit as st
import itertools
import os
import re
import math
import statistics
import pandas as pd
from janome.tokenizer import Tokenizer
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

# --- GOOGLE SHEETS SETTINGS ---
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

SPREADSHEET_NAME = "2026S_research_LinguisticTolerance"
SECRET_FILE_PATH = "s-r-linguistictolerance-d71c5893c907.json"

@st.cache_resource
def get_gspread_client():
    if os.path.exists(SECRET_FILE_PATH):
        try:
            creds = Credentials.from_service_account_file(SECRET_FILE_PATH, scopes=SCOPES)
            return gspread.authorize(creds)
        except Exception as e:
            st.error(f"認証ファイルの読み込みエラー: {e}")
            return None
            
    try:
        if "gcp_service_account" in st.secrets:
            creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=SCOPES)
            return gspread.authorize(creds)
    except Exception:
        pass

    st.error(f"認証情報が見つかりません。プロジェクト直下に '{SECRET_FILE_PATH}' を配置してください。")
    return None

def get_worksheet(sheet_name: str):
    client = get_gspread_client()
    if client:
        try:
            sh = client.open(SPREADSHEET_NAME)
            return sh.worksheet(sheet_name)
        except Exception as e:
            st.error(f"スプレッドシートの取得に失敗しました ({sheet_name}): {e}")
            return None
    return None

# --- キャッシュ機能付き：全データの取得（Read API削減用） ---
@st.cache_data(ttl=60)
def fetch_sheet_records(sheet_name: str) -> list[list[str]]:
    """シートの全データを60秒間キャッシュして取得"""
    ws = get_worksheet(sheet_name)
    if ws:
        try:
            return ws.get_all_values()
        except Exception as e:
            st.warning(f"シートの読み込みエラー ({sheet_name}): {e}")
    return []

# --- A列・B列 および E列・F列 を含む辞書データの読み込み ---
def load_dictionary_from_sheets() -> list[dict]:
    records = fetch_sheet_records("dictionary_data")
    dictionary = []
    if len(records) > 1:
        for row in records[1:]:
            # A列 (0) & B列 (1)
            if len(row) >= 1 and row[0].strip():
                word_a = row[0].strip()
                romaji_b = row[1].strip() if len(row) >= 2 and row[1].strip() else katakana_to_romaji_str(word_a)
                dictionary.append({"word": word_a, "romaji": romaji_b})
            
            # E列 (4) & F列 (5)
            if len(row) >= 5 and row[4].strip():
                word_e = row[4].strip()
                romaji_f = row[5].strip() if len(row) >= 6 and row[5].strip() else katakana_to_romaji_str(word_e)
                dictionary.append({"word": word_e, "romaji": romaji_f})

        # 重複排除
        unique_dict = {}
        for item in dictionary:
            if item["word"] not in unique_dict:
                unique_dict[item["word"]] = item["romaji"]
        
        return [{"word": w, "romaji": r} for w, r in unique_dict.items()]
    return dictionary

# --- モード別に分離された履歴取得機能 ---
def load_search_history_words(mode_filter: str) -> list[str]:
    records = fetch_sheet_records("search_history")
    if len(records) > 1:
        words = []
        for row in records[1:]:
            if len(row) >= 3 and row[1].strip() and row[2].strip() == mode_filter:
                words.append(row[1].strip())
        
        seen = set()
        unique_words = []
        for w in reversed(words):
            if w not in seen:
                seen.add(w)
                unique_words.append(w)
        return unique_words
    return []

# --- 重複チェック付き履歴保存機能 (search_history) ---
def save_history_to_sheets(input_word: str, mode: str, top_result: str):
    records = fetch_sheet_records("search_history")
    for row in records[1:]:
        if len(row) >= 3 and row[1] == input_word and row[2] == mode:
            return  # 既に存在する場合は追加しない
    
    ws = get_worksheet("search_history")
    if ws:
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ws.append_row([now, input_word, mode, top_result])
            st.cache_data.clear()  # 追加後にキャッシュをクリアして最新化
        except Exception as e:
            st.warning(f"履歴の保存に失敗しました: {e}")

# --- dictionary_data シートの E列・F列への単語・ローマ字追加機能 ---
def save_to_dictionary_data(word: str):
    records = fetch_sheet_records("dictionary_data")
    
    # 既存のE列データを確認
    e_values = [row[4] for row in records if len(row) >= 5]
    if word in e_values:
        return  # 重複登録回避

    ws = get_worksheet("dictionary_data")
    if not ws:
        return

    try:
        next_row = len(e_values) + 1
        if next_row < 2:
            next_row = 2

        romaji = katakana_to_romaji_str(word)
        ws.update(f"E{next_row}:F{next_row}", [[word, romaji]])
        st.cache_data.clear()  # 追加後にキャッシュをクリア
    except Exception as e:
        st.warning(f"dictionary_data への保存に失敗しました: {e}")

# --- 全アナグラム統計データ保存機能 ---
def save_full_patterns_to_sheets(target_word: str, all_results: list[dict]):
    records = fetch_sheet_records("pattern_data")
    for row in records:
        if row and row[0] == target_word:
            return  # 既に保存済みならスキップ

    ws = get_worksheet("pattern_data")
    if not ws:
        return

    try:
        n = len(target_word)
        factorial_count = math.factorial(n) - 1
        actual_patterns = len(all_results)
        
        vsi_scores = [r["vsi"] for r in all_results]
        pmd_scores = [r["pmd"] for r in all_results]

        avg_vsi = round(statistics.mean(vsi_scores), 3) if vsi_scores else 0
        min_vsi = min(vsi_scores) if vsi_scores else 0
        max_vsi = max(vsi_scores) if vsi_scores else 0

        avg_pmd = round(statistics.mean(pmd_scores), 3) if pmd_scores else 0
        min_pmd = min(pmd_scores) if pmd_scores else 0
        max_pmd = max(pmd_scores) if pmd_scores else 0

        ws.append_row([
            target_word, 
            n, 
            factorial_count, 
            actual_patterns, 
            avg_vsi, min_vsi, max_vsi, 
            avg_pmd, min_pmd, max_pmd
        ])
        st.cache_data.clear()

    except Exception as e:
        st.warning(f"パターン統計データの保存に失敗しました: {e}")

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
        
    pos_diff = sum(abs(i - shuffled.index(c)) for i, c in enumerate(orig) if c in shuffled) / (n * n)
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
    
    top_res = vsi_sorted[0]["word_vsi"].replace("★ ", "") if vsi_sorted else ""
    
    save_history_to_sheets(target_word, "許容度解析", f"Top VSI: {top_res}")
    save_full_patterns_to_sheets(target_word, results)
    save_to_dictionary_data(target_word)

@st.cache_resource
def get_tokenizer():
    return Tokenizer()

# --- A列・B列およびE列・F列を参照したハイブリッド逆引き ---
def reverse_lookup_hybrid(typo_word: str, alpha: float, c_same: float, c_diff: float, v_cost: float):
    t = get_tokenizer()
    typo_len = len(typo_word)
    
    sheet_dict_items = load_dictionary_from_sheets()
    candidates_dict = {}
    
    for item in sheet_dict_items:
        word = item["word"]
        if abs(len(word) - typo_len) <= 2:
            candidates_dict[word] = item["romaji"]
            
    try:
        sys_dic = t.sys_dic
        for entry in sys_dic.entries:
            morph = entry[4]
            if len(morph) >= 3 and abs(len(morph) - typo_len) <= 1:
                if bool(re.match(r'^[ァ-タチ-ヶー]+$', morph)) and morph not in candidates_dict:
                    candidates_dict[morph] = katakana_to_romaji_str(morph)
    except Exception:
        pass

    results = []
    typo_chars = set(typo_word)
    
    for cand_word, cand_romaji in candidates_dict.items():
        if cand_word == typo_word:
            continue
            
        cand_chars = set(cand_word)
        common_chars = typo_chars.intersection(cand_chars)
        
        if len(common_chars) >= max(2, len(typo_word) // 2):
            min_len = min(len(cand_word), len(typo_word))
            cand_sub = cand_word[:min_len]
            typo_sub = typo_word[:min_len]
            
            v_score = calculate_vsi(cand_sub, typo_sub, alpha)
            p_score = calculate_pmd(cand_sub, typo_sub, c_same, c_diff, v_cost)
            
            len_penalty = abs(len(cand_word) - len(typo_word)) * 0.2
            combined_score = round(((v_score + p_score) / 2) + len_penalty, 3)
            
            results.append({
                "元単語候補": cand_word,
                "推定アルファベット": cand_romaji,
                "VSI (視覚コスト)": v_score,
                "PMD (音律コスト)": p_score,
                "総合誤認スコア": combined_score
            })
            
    sorted_res = sorted(results, key=lambda x: x["総合誤認スコア"])
    return sorted_res[:10]

# --- UI DESIGN ---
st.set_page_config(page_title="言語的許容度シュミレーター", layout="wide")

st.title('ニンゲンの許容範囲"シュミレーター"')

tab1, tab2 = st.tabs(["許容度解析", "逆引き"])

with st.expander("評価基準"):
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("**1. VSI（タイポグリセミア）**")
        alpha_param = st.slider("語頭・語尾位置ずれペナルティ (α)", 0.0, 1.0, 0.35, 0.05)
    with col_p2:
        st.markdown("**2. PMD（メタセシス）**")
        c_same_param = st.slider("同子音グループ間コスト", 0.0, 1.0, 0.2, 0.05)
        c_diff_param = st.slider("異子音グループ間コスト", 0.0, 1.0, 0.5, 0.05)
        v_cost_param = st.slider("母音不一致コスト", 0.0, 1.0, 0.3, 0.05)

# --- TAB 1: 許容度解析 ---
with tab1:
    history_words_tab1 = load_search_history_words("許容度解析")
    selected_from_history_tab1 = st.session_state.get("selected_history_word_tab1", None)
    
    with st.form("input_form_tab1", clear_on_submit=False):
        col_input, col_btn = st.columns([3, 1])
        with col_input:
            word_input = st.selectbox(
                label="入力欄",
                options=history_words_tab1,
                index=history_words_tab1.index(selected_from_history_tab1) if selected_from_history_tab1 in history_words_tab1 else None,
                placeholder="クリックして入力するか、過去の解析履歴から選択してください",
                label_visibility="collapsed",
                accept_new_options=True
            )
        with col_btn:
            run_btn = st.form_submit_button("実行", type="primary", use_container_width=True)

    target_word_to_process = None
    if run_btn and word_input:
        target_word_to_process = word_input.strip()
    elif selected_from_history_tab1:
        target_word_to_process = selected_from_history_tab1
        st.session_state["selected_history_word_tab1"] = None

    if target_word_to_process:
        if not bool(re.match(r'^[ァ-タチ-ヶー]+$', target_word_to_process)):
            st.error("全角カタカナで入力してください。")
        elif len(target_word_to_process) < 3:
            st.warning("3文字以上の単語を入力してください。")
        else:
            st.session_state["show_detail"] = False
            process_word_analysis(target_word_to_process, alpha_param, c_same_param, c_diff_param, v_cost_param)

    if "vsi_sorted" in st.session_state and st.session_state["vsi_sorted"]:
        st.divider()
        analyzed_word = st.session_state.get("analyzed_word", "")
        n_len = len(analyzed_word)
        theory_max = math.factorial(n_len) - 1
        actual_pats = st.session_state.get('total_patterns', 0)
        
        st.markdown(f"### 解析結果: **{analyzed_word}** (文字数: {n_len} / 理論上パターン数 $n!-1$: {theory_max} / 生成パターン数: {actual_pats})")
        
        vsi_list = st.session_state["vsi_sorted"]
        pmd_list = st.session_state["pmd_sorted"]
        
        col_vsi, col_pmd = st.columns(2)
        show_all = st.session_state.get("show_detail", False)
        table_height = 800 if show_all else 400
        
        with col_vsi:
            st.subheader("許容されやすいタイポグリセミア")
            df_vsi = pd.DataFrame([{"単語": item["word_vsi"], "VSI": item["vsi"]} for item in (vsi_list if show_all else vsi_list[:10])])
            st.dataframe(df_vsi, use_container_width=True, hide_index=True, height=table_height)
            st.caption("初めと終わりの文字が固定されている単語の前に★がついています。")

        with col_pmd:
            st.subheader("おこりやすいメタセシス")
            df_pmd = pd.DataFrame([{"単語": item["word_pmd"], "Alphabet": item["alphabet"], "PMD": item["pmd"]} for item in (pmd_list if show_all else pmd_list[:10])])
            st.dataframe(df_pmd, use_container_width=True, hide_index=True, height=table_height)

        if st.button("閉じる" if show_all else "詳細はコチラ", use_container_width=True):
            st.session_state["show_detail"] = not show_all
            st.rerun()

    st.divider()
    st.markdown("### 許容度解析の履歴")
    st.caption("ボタンをクリックすると、過去の解析結果を呼び出します。")

    if history_words_tab1:
        cols = st.columns(5)
        for idx, h_word in enumerate(history_words_tab1):
            col = cols[idx % 5]
            if col.button(h_word, key=f"hist_btn_tab1_{idx}", use_container_width=True):
                st.session_state["selected_history_word_tab1"] = h_word
                st.rerun()
    else:
        st.info("解析履歴はまだありません。")

# --- TAB 2: 逆引き ---
with tab2:
    st.markdown("### 誤字・言い間違いから「元の正しい単語」を推測")
    st.caption("スプレッドシート辞書 (dictionary_data の A/B列 および E/F列) と形態素解析エンジンに基づき、元単語候補を復元します。")
    
    history_words_tab2 = load_search_history_words("逆引き推定")
    selected_from_history_tab2 = st.session_state.get("selected_history_word_tab2", None)

    with st.form("input_form_tab2", clear_on_submit=False):
        col_input2, col_btn2 = st.columns([3, 1])
        with col_input2:
            typo_input = st.selectbox(
                label="誤字入力欄",
                options=history_words_tab2,
                index=history_words_tab2.index(selected_from_history_tab2) if selected_from_history_tab2 in history_words_tab2 else None,
                placeholder="誤字・崩れた単語を入力、または過去の逆引き履歴から選択",
                label_visibility="collapsed",
                accept_new_options=True
            )
        with col_btn2:
            run_btn_tab2 = st.form_submit_button("元単語を推定する", type="primary", use_container_width=True)

    target_typo_to_process = None
    if run_btn_tab2 and typo_input:
        target_typo_to_process = typo_input.strip()
    elif selected_from_history_tab2:
        target_typo_to_process = selected_from_history_tab2
        st.session_state["selected_history_word_tab2"] = None

    if target_typo_to_process:
        clean_typo = target_typo_to_process
        with st.spinner("スプレッドシート辞書 (A/B列 & E/F列) から推定中..."):
            candidates = reverse_lookup_hybrid(clean_typo, alpha_param, c_same_param, c_diff_param, v_cost_param)
        
        if candidates:
            st.markdown(f"**「{clean_typo}」 の元単語 推定ランキング**")
            df_cand = pd.DataFrame(candidates)
            st.dataframe(df_cand, use_container_width=True, hide_index=True)
            
            top_cand = candidates[0]['元単語候補']
            st.success(f"最も可能性が高い元単語は **「{top_cand}」** です！（総合誤認スコア: {candidates[0]['総合誤認スコア']}）")
            
            # 検索履歴にのみ保存（dictionary_data に誤字は保存しない）
            save_history_to_sheets(clean_typo, "逆引き推定", f"推定結果: {top_cand}")
        else:
            st.warning("適合する元単語候補が見つかりませんでした。スプレッドシートの `dictionary_data` に正解単語が登録されているかご確認ください。")

    st.divider()
    st.markdown("### 逆引きの履歴")
    st.caption("ボタンをクリックすると、過去の逆引きを再実行します。")

    if history_words_tab2:
        cols2 = st.columns(5)
        for idx, h_word in enumerate(history_words_tab2):
            col = cols2[idx % 5]
            if col.button(h_word, key=f"hist_btn_tab2_{idx}", use_container_width=True):
                st.session_state["selected_history_word_tab2"] = h_word
                st.rerun()
    else:
        st.info("逆引きの履歴はまだありません。")