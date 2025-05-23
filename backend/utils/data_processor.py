"""
Data processing module for WeChat chat records
"""
import os
import pandas as pd
import jieba
import re
from collections import Counter
from datetime import datetime
import matplotlib
# Use Agg backend to avoid thread issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
import numpy as np
from pyecharts import options as opts
from pyecharts.charts import WordCloud as PyechartsWordCloud
from pyecharts.globals import ThemeType
from googletrans import Translator

# 添加配置变量 - 是否启用emoji符号
USE_EMOJI_SYMBOLS = True

# Create translator instance
translator = Translator()

# 新增emoji符号映射字典
EMOJI_TO_SYMBOL = {
    # 基本表情
    '微笑': '😊', '开心': '😄', '大笑': '😂', '可爱': '🥰',
    '愤怒': '😡', '悲伤': '😢', '惊讶': '😮', '害怕': '😨',
    '无语': '😑', '尴尬': '😅', '思考': '🤔', '点赞': '👍',
    '爱心': '❤️', '玫瑰': '🌹', '蛋糕': '🎂', '礼物': '🎁',
    '笑哭': '😂', '害羞': '😊', '委屈': '🥺', '呆': '😶',
    '疑问': '❓', '晕': '😵', '衰': '😩', '叹气': '😮‍💨',
    
    # 微信常用表情
    '强': '💪', '弱': '👎', 'ok': '👌', '拜托': '🙏',
    '呲牙': '😁', '偷笑': '😏', '再见': '👋', '抓狂': '😫',
    '鼓掌': '👏', '流汗': '😓', '憨笑': '😄', '悠闲': '😌',
    '奋斗': '💪', '咒骂': '🤬', '疑问': '❓', '嘘': '🤫',
    '晕': '😵', '折磨': '😖', '衰': '😩', '骷髅': '💀',
    '敲打': '👊', '再见': '👋', '闭嘴': '🤐', '鄙视': '😤',
    '闪电': '⚡', '发抖': '😨', '难过': '😞', '酷': '😎',
    '抠鼻': '👃', '坏笑': '😏', '左哼哼': '😤', '右哼哼': '😤',
    '哈欠': '🥱', '鄙视': '😤', '委屈': '🥺', '快哭了': '😢',
    '阴险': '😈', '亲亲': '😘', '吓': '😱', '可怜': '🥺',
    '菜刀': '🔪', '西瓜': '🍉', '啤酒': '🍺', '篮球': '🏀',
    '乒乓': '🏓', '拥抱': '🤗', '握手': '🤝'
}

# 新增emoji的英文翻译
EMOJI_TO_ENGLISH = {
    # 基本表情
    '微笑': 'Smile', '开心': 'Happy', '大笑': 'Laugh', '可爱': 'Cute',
    '愤怒': 'Angry', '悲伤': 'Sad', '惊讶': 'Surprised', '害怕': 'Scared',
    '无语': 'Speechless', '尴尬': 'Embarrassed', '思考': 'Thinking', '点赞': 'Thumbs up',
    '爱心': 'Heart', '玫瑰': 'Rose', '蛋糕': 'Cake', '礼物': 'Gift',
    '笑哭': 'Laugh Cry', '害羞': 'Shy', '委屈': 'Wronged', '呆': 'Blank',
    '疑问': 'Question', '晕': 'Dizzy', '衰': 'Depressed', '叹气': 'Sigh',
    
    # 微信常用表情
    '强': 'Strong', '弱': 'Weak', 'ok': 'OK', '拜托': 'Please',
    '呲牙': 'Grin', '偷笑': 'Smirk', '再见': 'Goodbye', '抓狂': 'Crazy',
    '鼓掌': 'Clap', '流汗': 'Sweat', '憨笑': 'Silly Smile', '悠闲': 'Relaxed',
    '奋斗': 'Struggle', '咒骂': 'Curse', '疑问': 'Question', '嘘': 'Shh',
    '晕': 'Dizzy', '折磨': 'Torture', '衰': 'Depressed', '骷髅': 'Skull',
    '敲打': 'Punch', '再见': 'Goodbye', '闭嘴': 'Zip it', '鄙视': 'Despise',
    '闪电': 'Lightning', '发抖': 'Shiver', '难过': 'Sad', '酷': 'Cool',
    '抠鼻': 'Nose', '坏笑': 'Evil Smile', '左哼哼': 'Hmph Left', '右哼哼': 'Hmph Right',
    '哈欠': 'Yawn', '鄙视': 'Despise', '委屈': 'Wronged', '快哭了': 'About to Cry',
    '阴险': 'Sinister', '亲亲': 'Kiss', '吓': 'Scared', '可怜': 'Pitiful',
    '菜刀': 'Knife', '西瓜': 'Watermelon', '啤酒': 'Beer', '篮球': 'Basketball',
    '乒乓': 'Ping Pong', '拥抱': 'Hug', '握手': 'Handshake'
}

def contains(text, checklist):
    """
    Check if text contains any pattern in checklist
    
    Args:
        text: Text to check
        checklist: List of regex patterns
        
    Returns:
        bool: True if text contains any pattern
    """
    res = [bool(re.findall(each, text)) for each in checklist]
    return any(res)

def process_chat_data(file_path, analysis_id):
    """
    Process WeChat chat data from CSV file
    
    Args:
        file_path: Path to CSV file
        analysis_id: Unique ID for this analysis
        
    Returns:
        dict: Analysis results
    """
    # Create output directories
    output_dir = os.path.join('static', 'images', analysis_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    records = pd.read_csv(file_path, usecols=['IsSender', 'StrContent', 'StrTime']).dropna(how='any')
    
    # Filter out unwanted messages
    not_want_msg = [
        '<.+',  # e.g., <msg, <?xml
        r'^\d{1,}$'  # Pure numbers like verification codes
    ]
    records_not_want = records['StrContent'].apply(lambda x: contains(x, not_want_msg))
    records = records[~records_not_want]
    records.index = range(records.shape[0])
    
    # 修复：改进停用词文件路径的获取方式
    # 绝对路径
    current_file_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    
    emoji_file = os.path.join(base_dir, 'input_data', 'emoji.txt')
    stopword_file = os.path.join(base_dir, 'input_data', 'stopwords_hit_modified.txt')
    transform_file = os.path.join(base_dir, 'input_data', 'transformDict.txt')
    user_dict_file = os.path.join(base_dir, 'input_data', 'userDict.txt')
    
    print(f"Using stopword file: {stopword_file}")
    
    # Load emoji dictionary
    try:
        emoji_eng2cn = pd.read_table(emoji_file).set_index('eng').to_dict()['cn']
        print(f"Successfully loaded emoji dictionary with {len(emoji_eng2cn)} entries")
    except Exception as e:
        print(f"Error loading emoji file: {e}")
        emoji_eng2cn = {}
    
    # 修复：改进停用词加载方式并添加更多调试信息
    try:
        with open(stopword_file, 'r', encoding='utf-8') as f1:
            stop_words = set(f1.read().splitlines())
        print(f"Successfully loaded {len(stop_words)} stopwords")
        print(f"Sample stopwords: {list(stop_words)[:20]}")
    except Exception as e:
        print(f"Error loading stopwords: {e}")
        # 添加一个基本的停用词列表作为备用
        stop_words = {'一个','吧','吗','我', '你', '的', '了', '在', '是', '有', '和', '不', '这', '我们', '你们', '他们', '那', '就', '也', '都', '很', '到', '说'}
        print(f"Using fallback stopwords: {stop_words}")
    
    # Load transform dictionary
    try:
        transformDict = pd.read_table(transform_file).set_index('original').to_dict()['transformed']
    except Exception as e:
        print(f"Error loading transform dictionary: {e}")
        transformDict = {}
    
    # Load user dictionary
    try:
        jieba.load_userdict(user_dict_file)
    except Exception as e:
        print(f"Error loading user dictionary: {e}")
    
    # Process records
    records['keywords'] = [''] * records.shape[0]
    records['emoji'] = [''] * records.shape[0]
    
    # 新增：创建emoji符号列
    if USE_EMOJI_SYMBOLS:
        records['emoji_symbols'] = [''] * records.shape[0]
    
    # Parse dates
    records['StrTime'] = pd.to_datetime(records['StrTime'])
    
    # 修复：改进关键词提取和停用词过滤逻辑
    for i in range(records.shape[0]):
        result = []
        emoji_res = []
        emoji_symbols = []  # 新增：存储emoji符号
        
        # 打印前几条消息内容用于调试
        if i < 3:
            print(f"Processing message {i}: {records.loc[i, 'StrContent'][:30]}...")
        
        # 分词
        content = records.loc[i, 'StrContent']
        words = list(jieba.cut(content))
        
        if i < 3:
            print(f"Segmented words for message {i} (first 10): {words[:10]}")
        
        # 提取关键词，强化停用词过滤
        for word in words:
            word = word.strip()
            
            # 检查是否是停用词
            if word in stop_words:
                if i < 3:
                    print(f"Filtered out stopword: '{word}'")
                continue  # 跳过停用词
            
            # 其他过滤条件
            if word and re.findall(r'[\[\]一-龟a-zA-Z0-9]+', word) and \
               not re.findall(r'^\d{1,}$|^\d{1,}\.\d{1,}$', word) and \
               not re.findall('^[a-zA-Z]$', word):
                if word in transformDict:
                    word = transformDict[word]
                result.append(word)
        
        # 处理表情
        for _ in range(result.count(']')):
            ind2 = result.index(']')
            if len(result) < 3:
                break
            if result[ind2 - 2] != '[':
                continue
            emoji_text = result[ind2 - 1]
            
            # 保持原有的表情处理逻辑
            if emoji_text in emoji_eng2cn.keys():
                # If English and in dictionary, convert to Chinese
                cur_emoji = '[' + emoji_eng2cn[emoji_text] + ']'
                emoji_text_clean = emoji_eng2cn[emoji_text]  # 转换为中文名称供符号查找
            else:
                if re.findall('^[0-9a-zA-Z]+$', emoji_text):
                    # If emoji_text is all alphanumeric and not in emoji dictionary, delete and skip
                    del result[ind2 - 2:ind2 + 1]
                    continue
                cur_emoji = '[' + emoji_text + ']'  # 纯汉字
                emoji_text_clean = emoji_text  # 直接使用中文名称
            
            # 新增：获取对应的emoji符号
            if USE_EMOJI_SYMBOLS:
                # 处理表情文本，去掉可能的括号
                if emoji_text_clean.startswith('[') and emoji_text_clean.endswith(']'):
                    emoji_text_clean = emoji_text_clean[1:-1]
                
                # 查找对应的emoji符号
                emoji_symbol = EMOJI_TO_SYMBOL.get(emoji_text_clean, '')
                
                # 如果在字典中找不到完整匹配，尝试部分匹配
                if not emoji_symbol:
                    for key, symbol in EMOJI_TO_SYMBOL.items():
                        if key in emoji_text_clean or emoji_text_clean in key:
                            emoji_symbol = symbol
                            break
                
                if emoji_symbol:
                    emoji_symbols.append(emoji_symbol)
                    # 调试信息
                    if i < 3:
                        print(f"Found emoji: {emoji_text_clean} -> {emoji_symbol}")
                else:
                    if i < 3:
                        print(f"Could not find emoji symbol for: {emoji_text_clean}")
            
            del result[ind2 - 2:ind2 + 1]
            emoji_res.append(cur_emoji)
        
        # 保存关键词和表情
        records.at[i, 'keywords'] = ', '.join(result)
        records.at[i, 'emoji'] = ', '.join(emoji_res)
        
        # 新增：保存emoji符号
        if USE_EMOJI_SYMBOLS:
            records.at[i, 'emoji_symbols'] = ', '.join(emoji_symbols)
    
    # Replace empty strings with NaN for easier filtering
    records.replace('', float('nan'), inplace=True)
    
    # Remove records with no keywords or emojis
    records.dropna(how='all', subset=['keywords', 'emoji'], inplace=True)
    
    # Generate analysis results
    analysis_results = {}
    
    # Word frequency analysis
    analysis_results['word_cloud'] = generate_word_cloud(records, output_dir)
    
    # Time analysis
    analysis_results['time_analysis'] = analyze_time(records, output_dir)
    
    # Calendar heatmap
    analysis_results['calendar_heatmap'] = generate_calendar_heatmap(records, output_dir)
    
    # 修复：替换translate_keywords函数，避免异步问题
    analysis_results['keywords'] = extract_keywords(records)
    
    # 新增：如果启用了emoji符号，生成emoji词云
    if USE_EMOJI_SYMBOLS and 'emoji_symbols' in records.columns:
        emoji_cloud_data = generate_emoji_word_cloud(records, output_dir)
        if emoji_cloud_data:
            analysis_results['emoji_symbol_cloud'] = emoji_cloud_data
    
    return analysis_results

def generate_word_cloud(records, output_dir):
    """
    Generate word cloud images
    
    Args:
        records: DataFrame with chat records
        output_dir: Directory to save images
        
    Returns:
        dict: Paths to word cloud images
    """
    # Extract keywords
    all_keywords = ', '.join(records['keywords'].dropna().tolist()).split(', ')
    sender_keywords = ', '.join(records.loc[records['IsSender'] == 1, 'keywords'].dropna().tolist()).split(', ')
    receiver_keywords = ', '.join(records.loc[records['IsSender'] == 0, 'keywords'].dropna().tolist()).split(', ')
    
    # Count word frequencies
    all_counter = Counter(all_keywords)
    sender_counter = Counter(sender_keywords)
    receiver_counter = Counter(receiver_keywords)
    
    # Remove brackets if they exist
    if '[' in all_counter:
        all_counter.pop('[')
    if ']' in all_counter:
        all_counter.pop(']')
    
    # Generate word clouds
    word_cloud_paths = {}
    
    # Both participants
    both_wc_path = os.path.join(output_dir, 'both_wordcloud.html')
    generate_pyecharts_wordcloud(all_counter, 'Both Participants', both_wc_path)
    word_cloud_paths['both'] = both_wc_path
    
    # Sender
    sender_wc_path = os.path.join(output_dir, 'sender_wordcloud.html')
    generate_pyecharts_wordcloud(sender_counter, 'You', sender_wc_path)
    word_cloud_paths['sender'] = sender_wc_path
    
    # Receiver
    receiver_wc_path = os.path.join(output_dir, 'receiver_wordcloud.html')
    generate_pyecharts_wordcloud(receiver_counter, 'Other Person', receiver_wc_path)
    word_cloud_paths['receiver'] = receiver_wc_path
    
    # Generate French version
    # Translate top 100 words
    top_words = [word for word, _ in all_counter.most_common(100)]
    french_translations = {}
    
    try:
        for word in top_words:
            french_translations[word] = translator.translate(word, src='zh-cn', dest='fr').text
        
        # Create French word cloud
        french_counter = Counter({french_translations.get(word, word): count 
                                 for word, count in all_counter.most_common(100)})
        
        french_wc_path = os.path.join(output_dir, 'french_wordcloud.html')
        generate_pyecharts_wordcloud(french_counter, 'Nuage de mots (Français)', french_wc_path)
        word_cloud_paths['french'] = french_wc_path
    except:
        # If translation fails, skip French word cloud
        pass
    
    return word_cloud_paths

def generate_pyecharts_wordcloud(word_counter, title, output_path):
    """
    Generate word cloud using pyecharts
    
    Args:
        word_counter: Counter object with word frequencies
        title: Title for the word cloud
        output_path: Path to save the HTML file
    """
    word_pairs = [(word, count) for word, count in word_counter.most_common(150)]
    
    wc = (
        PyechartsWordCloud(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
        .add("", word_pairs, word_size_range=[15, 80], shape="circle")
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=title,
                title_textstyle_opts=opts.TextStyleOpts(font_size=25, color="midnightblue")
            )
        )
    )
    
    wc.render(output_path)

def analyze_time(records, output_dir):
    """
    Analyze message frequency by time
    """
    # Calculate time span
    date0 = records['StrTime'].min().date()
    date1 = records['StrTime'].max().date()
    n_date = (date1 - date0).days + 1
    
    # Add year-month and hour columns with format that ensures proper chronological sorting
    records['year-month'] = records['StrTime'].apply(lambda x: f"{x.year}-{str(x.month).zfill(2)}")
    records['hour'] = records['StrTime'].apply(lambda x: str(x.hour).zfill(2))
    
    # Monthly data
    monthly_data = records.groupby(by='year-month')['year-month'].count()
    
    # Hourly data
    hourly_data = records.groupby(by='hour')['hour'].count() / n_date
    
    # Fill missing hours with zeros
    missing_hours = set(str(each).zfill(2) for each in range(24)) - set(hourly_data.index)
    zero_hours = pd.Series([0 for _ in range(len(missing_hours))], index=missing_hours)
    hourly_data = pd.concat((hourly_data, zero_hours))
    hourly_data.sort_index(inplace=True)
    
    # Generate monthly chart
    plt.figure(figsize=(12, 8))
    plt.rc('font', family='SimSun', size=15)
    plt.rcParams['axes.unicode_minus'] = False
    
    # Use consistent color for all bars
    plt.bar(monthly_data.index, monthly_data.values, color='#1f77b4')  # Standard blue color
    
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Message Count', fontsize=15)
    plt.title('Monthly Message Count', fontsize=20)
    
    monthly_path = os.path.join(output_dir, 'monthly_messages.png')
    plt.tight_layout()  # Improve layout
    plt.savefig(monthly_path)
    plt.close()
    
    # Generate hourly chart
    plt.figure(figsize=(12, 8))
    plt.rc('font', family='SimSun', size=15)
    
    x2 = list(range(24))
    plt.bar(x2, hourly_data, color='#1f77b4')  # Standard blue color
    plt.xticks(x2, hourly_data.index)
    plt.xlabel('Hour', fontsize=20)
    plt.ylabel('Average Message Count', fontsize=20)
    plt.title('Hourly Message Distribution', fontsize=20)
    
    hourly_path = os.path.join(output_dir, 'hourly_messages.png')
    plt.tight_layout()  # Improve layout
    plt.savefig(hourly_path)
    plt.close()
    
    return {
        'first_date': date0.strftime('%Y-%m-%d'),
        'last_date': date1.strftime('%Y-%m-%d'),
        'total_days': n_date,
        'total_messages': len(records),
        'monthly_chart': monthly_path,
        'hourly_chart': hourly_path
    }

def generate_calendar_heatmap(records, output_dir):
    """
    Generate calendar heatmap showing message frequency by date
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame if it's a list
    if isinstance(records, list):
        df = pd.DataFrame(records)
    else:
        df = records
    
    # Count messages by date
    df['date'] = pd.to_datetime(df['StrTime']).dt.date
    date_counts = df.groupby('date').size().reset_index(name='count')
    
    # Get date range
    min_date = date_counts['date'].min()
    max_date = date_counts['date'].max()
    
    # Only use actual data range
    date_range = pd.date_range(start=min_date, end=max_date)
    
    # Create date index
    date_index = pd.DataFrame({'date': date_range})
    date_index['date'] = date_index['date'].dt.date
    
    # Merge with counts
    merged_df = date_index.merge(date_counts, on='date', how='left').fillna(0)
    
    # Format for heatmap with year-month for proper chronological sorting
    merged_df['year_month'] = pd.to_datetime(merged_df['date']).dt.strftime('%Y-%m')
    merged_df['day'] = pd.to_datetime(merged_df['date']).dt.day
    
    # Create pivot table
    pivot_df = merged_df.pivot(index='day', columns='year_month', values='count')
    
    # Reorder columns to ensure chronological order
    pivot_df = pivot_df[sorted(pivot_df.columns)]
    
    # Set up plot with better aspect ratio
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set font for characters
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create heatmap with improved color scheme
    cmap = plt.cm.Blues
    heatmap = ax.pcolor(pivot_df, cmap=cmap)
    
    # Format axes
    ax.set_yticks(np.arange(pivot_df.shape[0]) + 0.5)
    ax.set_yticklabels(pivot_df.index)
    ax.set_xticks(np.arange(pivot_df.shape[1]) + 0.5)
    ax.set_xticklabels(pivot_df.columns)
    
    # Add colorbar with better positioning
    cbar = plt.colorbar(heatmap, fraction=0.046, pad=0.04)
    cbar.set_label('Message Count', rotation=270, labelpad=15)
    
    # Title with year range
    min_year = min_date.year
    max_year = max_date.year
    year_range = f"{min_year}" if min_year == max_year else f"{min_year}-{max_year}"
    plt.title(f"Chat Frequency Heatmap {year_range}")
    
    # Save image
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    output_path = os.path.join(output_dir, f'calendar_heatmap_{timestamp}.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def extract_keywords(records):
    """
    Extract keywords without translation for sentiment analysis
    """
    # Extract all keywords
    all_keywords = ', '.join(records['keywords'].dropna().tolist()).split(', ')
    
    # Count word frequencies
    word_counter = Counter(all_keywords)
    
    # Get top 200 words
    top_words = [word for word, _ in word_counter.most_common(200)]
    
    return {
        'original': top_words,
        'translated': top_words,  # Same as original, no translation needed
        'frequencies': [word_counter[word] for word in top_words]
    }

def generate_emoji_word_cloud(records, output_dir):
    """
    生成支持emoji的词云
    
    Args:
        records: DataFrame with chat records
        output_dir: Directory to save images
        
    Returns:
        dict: Paths to emoji word cloud images
    """
    # 检查是否有emoji符号数据
    if 'emoji_symbols' not in records.columns or records['emoji_symbols'].dropna().empty:
        print("No emoji symbols found, skipping emoji word cloud generation")
        return None
    
    # 提取所有emoji符号
    all_emojis = []
    sender_emojis = []
    receiver_emojis = []
    
    # 逐行处理emoji符号，避免丢失数据
    for _, row in records.iterrows():
        if pd.notna(row.get('emoji_symbols')) and row['emoji_symbols']:
            emojis = row['emoji_symbols'].split(', ')
            all_emojis.extend([e for e in emojis if e])
            
            if row['IsSender'] == 1:
                sender_emojis.extend([e for e in emojis if e])
            else:
                receiver_emojis.extend([e for e in emojis if e])
    
    if not all_emojis:
        print("No valid emoji symbols found after filtering")
        return None
    
    # 计数
    all_counter = Counter(all_emojis)
    sender_counter = Counter(sender_emojis)
    receiver_counter = Counter(receiver_emojis)
    
    # 移除空值
    if '' in all_counter:
        all_counter.pop('')
    
    print(f"Found {len(all_counter)} unique emoji symbols")
    print(f"Most common emojis: {all_counter.most_common(5)}")
    
    # 生成词云
    emoji_cloud_paths = {}
    
    # 发送者
    if sender_counter:
        print(f"Sender emojis: {sender_counter.most_common(5)}")
        sender_emoji_path = os.path.join(output_dir, 'sender_emoji_cloud.html')
        if generate_emoji_pyecharts_wordcloud(sender_counter, 'Your Emoji Usage', sender_emoji_path):
            emoji_cloud_paths['sender'] = sender_emoji_path
    
    # 接收者
    if receiver_counter:
        print(f"Receiver emojis: {receiver_counter.most_common(5)}")
        receiver_emoji_path = os.path.join(output_dir, 'receiver_emoji_cloud.html')
        if generate_emoji_pyecharts_wordcloud(receiver_counter, 'Other Person Emoji Usage', receiver_emoji_path):
            emoji_cloud_paths['receiver'] = receiver_emoji_path
    
    return emoji_cloud_paths

def generate_emoji_pyecharts_wordcloud(emoji_counter, title, output_path):
    """
    生成支持emoji的词云
    
    Args:
        emoji_counter: Counter object with emoji frequencies
        title: Title for the word cloud
        output_path: Path to save the HTML file
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not emoji_counter:
        return False
    
    # 准备带有tooltip的词云数据
    word_pairs = []
    tooltip_data = []
    
    # 不设置上限，显示所有emoji
    for emoji, count in emoji_counter.most_common():
        if emoji and count > 0:
            # 查找对应的中文表情名称和英文翻译
            found = False
            for ch_name, symbol in EMOJI_TO_SYMBOL.items():
                if symbol == emoji:
                    english_name = EMOJI_TO_ENGLISH.get(ch_name, ch_name)
                    word_pairs.append((emoji, count))
                    tooltip_data.append(f"{english_name} ({count})")
                    found = True
                    break
            
            if not found:
                # 如果找不到对应的中文名称，直接使用emoji
                word_pairs.append((emoji, count))
                tooltip_data.append(f"Emoji ({count})")
    
    if not word_pairs:
        print("No valid emoji word pairs found")
        return False
    
    print(f"Generating emoji word cloud with {len(word_pairs)} emojis")
    
    try:
        # 创建词云，不使用内置tooltip
        wc = (
            PyechartsWordCloud(init_opts=opts.InitOpts(
                theme=ThemeType.LIGHT,
                width="100%",
                height="400px"  # 增加高度以容纳更多emoji
            ))
            .add("", 
                 word_pairs,  # 只传递emoji和频率
                 word_size_range=[20, 80],  # 调整词云字体大小
                 shape="circle", 
                 textstyle_opts=opts.TextStyleOpts(
                     font_family="'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', sans-serif")
                )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=title,
                    title_textstyle_opts=opts.TextStyleOpts(font_size=22, color="midnightblue")
                )
            )
        )
        
        # 渲染词云
        wc.render(output_path)
        
        # 读取渲染后的HTML
        with open(output_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # 添加自定义CSS和JavaScript
        custom_style = """
        <style>
        .chart-container {
            height: 400px !important;
            overflow: visible !important;
        }
        .chart-container text {
            font-family: 'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', sans-serif;
            font-size: 1.5em !important;
            cursor: pointer;
        }
        .chart-container svg {
            overflow: visible !important;
        }
        .tooltip {
            position: absolute;
            display: none;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-family: Arial, sans-serif;
            font-size: 14px;
            z-index: 100;
            pointer-events: none;
        }
        </style>
        <div class="tooltip" id="custom_tooltip"></div>
        <script>
        // 定义tooltip数据
        var tooltipData = """ + str(tooltip_data).replace("'", '"') + """;
        
        window.addEventListener('load', function() {
            var chart = document.querySelector('.chart-container');
            var tooltip = document.getElementById('custom_tooltip');
            
            // 获取所有emoji文本元素
            var textElements = chart.querySelectorAll('text');
            
            // 为每个文本元素添加事件
            for (var i = 0; i < textElements.length; i++) {
                if (i < tooltipData.length) {
                    let tooltipText = tooltipData[i];
                    textElements[i].addEventListener('mousemove', function(e) {
                        tooltip.textContent = tooltipText;
                        tooltip.style.display = 'block';
                        tooltip.style.left = (e.pageX + 10) + 'px';
                        tooltip.style.top = (e.pageY + 10) + 'px';
                    });
                    
                    textElements[i].addEventListener('mouseout', function() {
                        tooltip.style.display = 'none';
                    });
                }
            }
        });
        </script>
        """
        
        # 替换HTML内容
        html_content = html_content.replace('</body>', custom_style + '</body>')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Successfully generated emoji word cloud with tooltips: {output_path}")
        return True
    except Exception as e:
        print(f"Error generating emoji word cloud: {e}")
        return False 