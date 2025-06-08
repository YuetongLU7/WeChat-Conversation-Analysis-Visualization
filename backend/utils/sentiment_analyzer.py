"""
Enhanced sentiment analysis module for Chinese text.
Combines lexicon-based and machine learning approaches.
"""
import os
import json
import re
import jieba
import jieba.analyse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
from snownlp import SnowNLP
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from datetime import datetime

# Import stopwords
try:
    # Use the correct stopwords path (stop_words.txt instead of stopwords.txt)
    stopwords_path = os.path.join('backend', 'data', 'stop_words.txt')
    if not os.path.exists(stopwords_path):
        # Try relative path from current directory
        stopwords_path = os.path.join('data', 'stop_words.txt')
        if not os.path.exists(stopwords_path):
            # Try absolute path
            stopwords_path = os.path.abspath(os.path.join('backend', 'data', 'stop_words.txt'))
    
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        STOPWORDS = set([line.strip() for line in f.readlines()])
    print(f"Successfully loaded stopwords, total: {len(STOPWORDS)}")
except Exception as e:
    print(f"Failed to load stopwords: {e}")
    STOPWORDS = set()

# Load emotion lexicon
EMOTION_LEXICON = {
    'emotions': {
        'joy': {'words': set(['高兴', '快乐', '喜悦', '开心', '欢喜', '欣喜', '欢乐']), 'weight': 1.2},
        'love': {'words': set(['爱', '喜欢', '爱慕', '宝贝', '亲爱', '亲密', '宝宝']), 'weight': 1.3},
        'surprise': {'words': set(['惊讶', '震惊', '吃惊', '意外', '惊喜', '惊奇']), 'weight': 0.9},
        'sadness': {'words': set(['伤心', '难过', '悲伤', '悲痛', '忧伤', '哀伤']), 'weight': 0.8},
        'anger': {'words': set(['生气', '愤怒', '恼火', '发火', '气愤', '恼怒']), 'weight': 0.8},
        'fear': {'words': set(['害怕', '恐惧', '担心', '忧虑', '焦虑', '惶恐']), 'weight': 0.7}
    }
}

# Try to load custom emotion lexicon
try:
    custom_lexicon_path = os.path.join('backend', 'data', 'sentiment_dict.json')
    if not os.path.exists(custom_lexicon_path):
        custom_lexicon_path = os.path.join('data', 'sentiment_dict.json')
    
    if os.path.exists(custom_lexicon_path):
        with open(custom_lexicon_path, 'r', encoding='utf-8') as f:
            custom_lexicon = json.load(f)
            
            # Merge custom lexicon into existing lexicon
            for emotion, data in custom_lexicon.get('emotions', {}).items():
                if emotion in EMOTION_LEXICON['emotions']:
                    EMOTION_LEXICON['emotions'][emotion]['words'].update(set(data.get('words', [])))
                    # Optionally update weight
                    if 'weight' in data:
                        EMOTION_LEXICON['emotions'][emotion]['weight'] = data['weight']
                else:
                    EMOTION_LEXICON['emotions'][emotion] = {
                        'words': set(data.get('words', [])),
                        'weight': data.get('weight', 1.0)
                    }
            
            print(f"Successfully loaded custom emotion lexicon")
except Exception as e:
    print(f"Failed to load custom emotion lexicon: {e}")

# Load jieba dictionary
try:
    # Try different paths for user dictionary
    user_dict_path = os.path.join('backend', 'data', 'userDict.txt')
    if not os.path.exists(user_dict_path):
        user_dict_path = os.path.join('data', 'userDict.txt')
    
    if os.path.exists(user_dict_path):
        jieba.load_userdict(user_dict_path)
        print(f"Successfully loaded user dictionary")
    else:
        print(f"User dictionary not found, continuing without it")
except Exception as e:
    print(f"Failed to load user dictionary: {e}")

# Update stopwords for jieba
try:
    if os.path.exists(stopwords_path):
        jieba.analyse.set_stop_words(stopwords_path)
        print(f"Set stopwords for jieba.analyse")
except Exception as e:
    print(f"Failed to set stopwords for jieba: {e}")

def load_emotion_lexicon(lexicon_path='backend/data/emotion_lexicon.json'):
    """Load emotion lexicon from JSON file."""
    try:
        if not os.path.exists(lexicon_path):
            dir_path = os.path.dirname(os.path.realpath(__file__))
            alt_path = os.path.join(dir_path, '..', 'data', 'emotion_lexicon.json')
            if os.path.exists(alt_path):
                lexicon_path = alt_path
            else:
                raise FileNotFoundError(f"Emotion lexicon not found at '{lexicon_path}' or '{alt_path}'")
        
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading emotion lexicon: {e}")
        return {"emotions": {}, "degree_words": {}}

# Load lexicon globally
EMOTION_LEXICON = load_emotion_lexicon()

def init_jieba():
    """Initialize jieba with emotion words."""
    for emotion, data in EMOTION_LEXICON['emotions'].items():
        for word in data['words']:
            jieba.add_word(word, freq=10000)
        for word in data['intensity_words']:
            jieba.add_word(word, freq=10000)

# Initialize jieba
init_jieba()

def split_sentences(text):
    """Split text into sentences considering Chinese punctuation."""
    text = text.replace('。', '。|').replace('！', '！|').replace('？', '？|')
    return [s.strip() for s in text.split('|') if s.strip()]

def analyze_sentence(sentence):
    """
    Analyze the emotions in a sentence using lexicon-based approach
    """
    # Word segmentation
    words = jieba.lcut(sentence)
    
    # Filter stopwords
    words = [word for word in words if word not in STOPWORDS and len(word.strip()) > 0]
    
    # Initialize emotion scores
    emotion_scores = defaultdict(float)
    
    # Check if each word is in the emotion lexicon
    for word in words:
        for emotion, data in EMOTION_LEXICON['emotions'].items():
            if word in data['words']:
                emotion_scores[emotion] += data['weight']
    
    # Check special emoji symbols
    emoji_patterns = {
        'joy': [r'😄', r'😊', r'😀', r'😁', r'😆', r'😂', r'🤣', r'😃', r'😅'],
        'love': [r'❤️', r'😍', r'😘', r'🥰', r'💕', r'💓', r'💗', r'💖', r'💘', r'💝'],
        'surprise': [r'😲', r'😮', r'😯', r'😱', r'🙀', r'😨', r'😧'],
        'sadness': [r'😢', r'😭', r'😥', r'😓', r'😔', r'😞', r'😟'],
        'anger': [r'😠', r'😡', r'🤬', r'💢', r'👿'],
        'fear': [r'😰', r'😱', r'😨', r'😧', r'😦']
    }
    
    # Check for emojis
    for emotion, patterns in emoji_patterns.items():
        for pattern in patterns:
            matches = re.findall(pattern, sentence)
            if matches:
                # Emoji emotions have higher weight
                emotion_scores[emotion] += len(matches) * 1.5
    
    # Check special text emoticons
    text_emoticons = {
        'joy': [r'\(\s*\^\s*\^\s*\)', r'\(\s*\:\s*\)', r'\^\s*\_\s*\^', r'\:\s*\)', r'\:\s*D', r'\=\s*\)'],
        'love': [r'\<\s*3', r'\(\s*\>\s*\<\s*\)', r'\(\s*\^\s*3\s*\^\s*\)'],
        'surprise': [r'\:\s*o', r'\:\s*O', r'\=\s*o', r'\=\s*O'],
        'sadness': [r'\:\s*\(', r'\:\s*\-\s*\(', r'\:\s*\<', r'\=\s*\('],
        'anger': [r'\>\s*\:\s*\(', r'\>\s*\<'],
        'fear': [r'\:\s*S', r'\:\s*s']
    }
    
    # Check for text emoticons
    for emotion, patterns in text_emoticons.items():
        for pattern in patterns:
            matches = re.findall(pattern, sentence)
            if matches:
                emotion_scores[emotion] += len(matches) * 0.8
    
    # Check for repeated punctuation, indicating emotional intensity
    punctuation_patterns = {
        'joy': [r'!{2,}', r'\?!+'],
        'surprise': [r'\?{2,}', r'!+\?+'],
        'anger': [r'!{3,}']
    }
    
    for emotion, patterns in punctuation_patterns.items():
        for pattern in patterns:
            matches = re.findall(pattern, sentence)
            if matches:
                for match in matches:
                    # Increase emotional intensity based on punctuation length
                    emotion_scores[emotion] += min(len(match) * 0.3, 1.5)
    
    # Detect expressions of intimacy
    intimacy_patterns = [
        r'宝贝', r'亲爱的', r'老婆', r'老公', r'媳妇', r'爱人',
        r'想你', r'思念', r'想念', r'爱你', r'亲亲', r'么么哒'
    ]
    
    for pattern in intimacy_patterns:
        if re.search(pattern, sentence):
            emotion_scores['love'] += 1.2
    
    # Detect expressions of teasing
    teasing_patterns = [
        r'哈哈+', r'呵呵+', r'嘿嘿+', r'嘻嘻+', 
        r'逗你', r'开玩笑', r'玩笑', r'调侃', r'搞笑'
    ]
    
    for pattern in teasing_patterns:
        matches = re.findall(pattern, sentence)
        if matches:
            # Teasing usually increases joy and surprise
            emotion_scores['joy'] += 0.8
            emotion_scores['surprise'] += 0.4
            # And may reduce anger and sadness
            if 'anger' in emotion_scores and emotion_scores['anger'] > 0:
                emotion_scores['anger'] *= 0.5
            if 'sadness' in emotion_scores and emotion_scores['sadness'] > 0:
                emotion_scores['sadness'] *= 0.5
    
    return dict(emotion_scores)

def analyze_sentiment(keywords_data):
    """
    Main sentiment analysis function combining multiple approaches.
    """
    original_words = keywords_data.get('original', [])
    frequencies = keywords_data.get('frequencies', [])
    
    if not original_words:
        return fallback_sentiment_analysis()
    
    # Debug info: print keywords
    print(f"Analyzing keywords: {original_words[:10]}...")
    
    # Create word frequency dictionary
    word_freq_dict = {}
    for i, word in enumerate(original_words):
        if i < len(frequencies):
            word_freq_dict[word] = frequencies[i]
        else:
            word_freq_dict[word] = 1
    
    # Group emotions into positive and negative
    positive_emotions = ['joy', 'love', 'surprise']  # Positive emotions
    negative_emotions = ['sadness', 'anger', 'fear']  # Negative emotions
    
    # Join words into text for analysis
    text = ' '.join(original_words)
    
    # Split into sentences
    sentences = split_sentences(text)
    if not sentences:
        sentences = [text]  # Use whole text if no sentence boundaries found
    
    # Debug info: print sentences
    print(f"Sentences: {sentences[:3]}...")
    
    # Analyze each sentence
    all_emotions = defaultdict(float)
    sentence_count = len(sentences)
    
    # Debug info: SnowNLP sentiment analysis results
    print("\n===== SnowNLP Sentiment Analysis Results =====")
    
    # Analyze keywords directly from lexicon
    for word, freq in word_freq_dict.items():
        for emotion, data in EMOTION_LEXICON['emotions'].items():
            if word in data['words']:
                # Weight by frequency
                weight = min(freq, 5) / 5.0  # Limit maximum weight
                all_emotions[emotion] += data['weight'] * weight
    
    # Adjust base weights for emotions
    base_adjustment = {
        'anger': 0.7,   # Reduce anger weight
        'sadness': 0.7, # Reduce sadness weight
        'fear': 0.6,    # Reduce fear weight
        'joy': 1.2,     # Increase joy weight
        'love': 1.3,    # Increase love weight
        'surprise': 1.0 # Keep surprise weight unchanged
    }
    
    for emotion in all_emotions:
        all_emotions[emotion] *= base_adjustment.get(emotion, 1.0)
    
    # Sentence level analysis
    for sentence in sentences:
        # Get SnowNLP sentiment score
        try:
            snlp = SnowNLP(sentence)
            sentiment_score = snlp.sentiments
            # Debug info: print SnowNLP sentiment score
            print(f"Sentence: '{sentence[:30]}...' SnowNLP score: {sentiment_score:.4f} (>0.5 means positive)")
        except Exception as e:
            sentiment_score = 0.5  # Default to neutral
            print(f"SnowNLP analysis error: {e}, Sentence: '{sentence[:30]}...'")
        
        # Get detailed emotion analysis
        emotion_scores = analyze_sentence(sentence)
        
        # Use non-linear function to adjust sentiment scores
        # Use a sigmoid variant to amplify differences
        def sigmoid_amplify(x, center=0.5, steepness=5):
            return 1.0 / (1.0 + np.exp(-steepness * (x - center)))
        
        # Calculate adjustment factors for positive and negative emotions
        positive_factor = sigmoid_amplify(sentiment_score)  # Positive emotions adjustment factor
        negative_factor = sigmoid_amplify(1 - sentiment_score)  # Negative emotions adjustment factor
        
        # Adjust each emotion score
        for emotion in positive_emotions:
            if emotion in emotion_scores:
                # Positive emotions increase with SnowNLP score
                emotion_scores[emotion] *= (1.0 + positive_factor)
        
        for emotion in negative_emotions:
            if emotion in emotion_scores:
                # Negative emotions increase as SnowNLP score decreases
                emotion_scores[emotion] *= (1.0 + negative_factor)
        
        # Debug info: print adjusted emotion scores
        print(f"Adjusted emotion scores: {emotion_scores}")
        
        # Accumulate scores
        for emotion, score in emotion_scores.items():
            all_emotions[emotion] += score
    
    # Process special keywords
    # Detect intimacy keywords
    intimacy_keywords = ["宝", "宝贝", "宝宝", "亲亲", "爱", "想你", "想念", "思念", "老婆", "老公", "媳妇", "爱人"]
    intimacy_count = sum(1 for word in original_words if any(ik in word for ik in intimacy_keywords))
    
    # Detect teasing keywords
    teasing_keywords = ["哈哈", "笑死", "逗", "调侃", "开玩笑", "玩笑", "搞笑", "好笑", "笑话"]
    teasing_count = sum(1 for word in original_words if any(tk in word for tk in teasing_keywords))
    
    # If intimacy keywords are present, increase love emotion
    if intimacy_count > 0:
        all_emotions['love'] += intimacy_count * 0.5
    
    # If teasing keywords are present, increase joy and surprise, decrease anger and sadness
    if teasing_count > 0:
        all_emotions['joy'] += teasing_count * 0.3
        all_emotions['surprise'] += teasing_count * 0.2
        # Reduce negative emotions, but don't eliminate completely
        if all_emotions['anger'] > 0:
            all_emotions['anger'] *= max(0.1, 1.0 - teasing_count * 0.1)
        if all_emotions['sadness'] > 0:
            all_emotions['sadness'] *= max(0.1, 1.0 - teasing_count * 0.1)
    
    # Detect emoji usage for all emotion types
    emoji_mapping = {
        'joy': ["😄", "😊", "😀", "😁", "😆", "😂", "🤣", "😃", "😅"],
        'love': ["😘", "❤️", "💕", "😊", "🥰", "💓", "💗", "💖"],
        'surprise': ["😲", "😮", "😯", "😱", "🙀", "😨", "😧", "😳"],
        'sadness': ["😢", "😭", "😥", "😓", "😔", "😞", "😟", "💔"],
        'anger': ["😠", "😡", "🤬", "💢", "👿", "😤"],
        'fear': ["😰", "😱", "😨", "😧", "😦", "🥶"]
    }
    
    # Count emojis for each emotion type
    for emotion, emojis in emoji_mapping.items():
        emoji_count = sum(1 for word in original_words if any(emoji in word for emoji in emojis))
        if emoji_count > 0:
            all_emotions[emotion] += emoji_count * 0.4
    
    # Average the scores
    emotions = {k: v/max(1, sentence_count/2) for k, v in all_emotions.items()}
    
    # Debug info: print averaged emotion scores
    print(f"\nAveraged emotion scores: {emotions}")
    
    # Ensure all emotions have values
    for emotion in positive_emotions + negative_emotions:
        if emotion not in emotions:
            emotions[emotion] = 0.2  # Set baseline value
    
    # Ensure no negative values
    emotions = {k: max(0, v) for k, v in emotions.items()}
    
    # Use more reasonable normalization method
    max_value = max(emotions.values()) if emotions.values() else 1.0
    if max_value > 0:
        # Maintain relative proportions, but set max value to 3.0
        scale_factor = 3.0 / max_value
        emotions = {k: v * scale_factor for k, v in emotions.items()}
    
    # Debug info: print normalized emotion scores
    print(f"Normalized emotion scores: {emotions}")
    
    # Generate outputs using existing visualization functions
    radar_chart_path = generate_emotion_radar_chart(emotions)
    summary = generate_sentiment_summary(emotions, keywords_data)
    
    return {
        'emotions': emotions,
        'radar_chart': radar_chart_path,
        'summary': summary
    }

def fallback_sentiment_analysis():
    """
    Provide a fallback sentiment analysis when no data is available
    """
    # Default balanced emotions
    emotions = {
        'joy': 1.0,
        'love': 1.0,
        'surprise': 1.0,
        'sadness': 1.0,
        'anger': 1.0,
        'fear': 1.0
    }
    
    # Generate radar chart with default emotions
    radar_chart_path = generate_emotion_radar_chart(emotions)
    
    # Generate a neutral summary
    summary = "无足够数据进行情感分析。(Insufficient data for sentiment analysis.)"
    
    return {
        'emotions': emotions,
        'radar_chart': radar_chart_path,
        'summary': summary
    }

def generate_emotion_radar_chart(emotions):
    """
    Generate a radar chart for emotions
    """
    # Define emotion labels with English translation
    emotion_labels = {
        'joy': '喜悦 (Joy)',
        'love': '亲密 (Love)',
        'surprise': '惊讶 (Surprise)',
        'sadness': '悲伤 (Sadness)',
        'anger': '愤怒 (Anger)',
        'fear': '恐惧 (Fear)'
    }
    
    # Ensure all emotions have values
    for emotion in emotion_labels.keys():
        if emotion not in emotions:
            emotions[emotion] = 0.1
    
    # Get emotion values in the correct order
    emotion_values = [emotions.get(emotion, 0) for emotion in emotion_labels.keys()]
    
    # Get labels in the correct order
    labels = [emotion_labels[emotion] for emotion in emotion_labels.keys()]
    
    # Create radar chart
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Number of variables
    N = len(labels)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Add values for each emotion, and close the loop
    values = emotion_values
    values += values[:1]
    
    # Draw the plot
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.4)
    
    # Set labels and their positions
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    
    # Set y-axis limits
    ax.set_ylim(0, 3)
    
    # Add title
    plt.title('情感雷达图 (Emotion Radar Chart)', size=15, y=1.1)
    
    # Save the chart - use static/images directory instead of frontend-v3/src/assets/charts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join('static', 'images')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'emotion_radar_{timestamp}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    # Return the path relative to static directory for frontend use
    return output_path

def generate_sentiment_summary(emotions, keywords_data):
    """
    Generate sentiment analysis summary
    """
    # Get top 5 keywords
    top_keywords = keywords_data.get('original', [])[:5]
    keywords_str = ', '.join(top_keywords)
    
    # Group emotions into positive and negative
    positive_emotions = ['joy', 'love', 'surprise']  # Positive emotions
    negative_emotions = ['sadness', 'anger', 'fear']  # Negative emotions
    
    # Calculate total scores for positive and negative emotions
    positive_score = sum(emotions.get(emotion, 0) for emotion in positive_emotions)
    negative_score = sum(emotions.get(emotion, 0) for emotion in negative_emotions)
    
    # Calculate total intensity and average
    total_intensity = sum(emotions.values())
    avg_intensity = total_intensity / len(emotions) if emotions else 0
    
    # Get main emotions (top two highest scoring emotions)
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    main_emotions = sorted_emotions[:2]
    
    # Check for intimacy and teasing keywords
    original_words = keywords_data.get('original', [])
    teasing_keywords = ["哈哈", "笑死", "逗", "调侃", "开玩笑", "玩笑", "搞笑", "好笑", "笑话", "逗乐"]
    intimacy_keywords = ["宝", "亲", "爱", "想你", "想念", "思念", "宝贝", "老婆", "老公", "媳妇", "爱人"]
    
    has_teasing = any(any(tk in word for tk in teasing_keywords) for word in original_words)
    has_intimacy = any(any(ik in word for ik in intimacy_keywords) for word in original_words)
    
    # Determine emotional tone
    if positive_score > negative_score * 2:
        # Clearly positive
        mood = "积极愉快 (positive and cheerful)"
    elif positive_score > negative_score * 1.2:
        # Moderately positive
        mood = "较为积极 (moderately positive)"
    elif negative_score > positive_score * 2:
        # Clearly negative
        mood = "消极低落 (negative and downbeat)"
    elif negative_score > positive_score * 1.2:
        # Moderately negative
        mood = "略显消极 (slightly negative)"
    else:
        # Balanced
        mood = "情感平衡 (emotionally balanced)"
    
    # Generate descriptions based on main emotion types
    emotion_descriptions = {
        'joy': '喜悦 (joy)',
        'love': '亲密友好 (intimate and friendly)',
        'surprise': '惊讶好奇 (surprised and curious)',
        'sadness': '低落忧伤 (sad and melancholic)',
        'anger': '烦躁不满 (irritated and dissatisfied)',
        'fear': '担忧焦虑 (worried and anxious)'
    }
    
    # Get description for main emotion
    main_emotion_desc = emotion_descriptions.get(main_emotions[0][0], '中性 (neutral)')
    
    # If second emotion is also strong, add its description
    if len(main_emotions) > 1 and main_emotions[1][1] > main_emotions[0][1] * 0.7:
        second_emotion_desc = emotion_descriptions.get(main_emotions[1][0], '')
        if second_emotion_desc and second_emotion_desc != main_emotion_desc:
            main_emotion_desc += f"，带有{second_emotion_desc}色彩 (with {main_emotions[1][0]} undertones)"
    
    # Generate summary
    if has_teasing and has_intimacy:
        # Intimate relationship with teasing
        summary = f"对话氛围亲密活跃，互相调侃中透露着感情。主要话题包括{keywords_str}。(The conversation atmosphere is intimate and lively, showing affection through playful teasing. Main topics include {keywords_str}.)"
    elif has_teasing:
        # Teasing without necessarily being intimate
        summary = f"对话氛围轻松幽默，有互相调侃的成分。主要话题包括{keywords_str}。(The conversation atmosphere is relaxed and humorous, with elements of teasing. Main topics include {keywords_str}.)"
    elif has_intimacy:
        # Intimate without necessarily teasing
        if main_emotions[0][0] in negative_emotions:
            # Complaints or venting in an intimate relationship
            summary = f"对话氛围亲密但略有情绪波动，可能是亲近关系中的小摩擦。主要话题包括{keywords_str}。(The conversation atmosphere is intimate but with some emotional fluctuations, possibly minor friction in a close relationship. Main topics include {keywords_str}.)"
        else:
            summary = f"对话氛围亲密温馨，表达了彼此的关心。主要话题包括{keywords_str}。(The conversation atmosphere is intimate and warm, expressing mutual care. Main topics include {keywords_str}.)"
    else:
        # Generate summary based on emotional tone
        summary = f"对话氛围{mood}，主要体现为{main_emotion_desc}。主要话题包括{keywords_str}。(The conversation atmosphere is {mood.split('(')[1][:-1]}, mainly characterized by {main_emotion_desc.split('(')[1][:-1]}. Main topics include {keywords_str}.)"
    
    # Add emotional intensity description
    if avg_intensity > 2.0:
        summary += "情感表达十分强烈。(Emotional expression is very intense.)"
    elif avg_intensity > 1.5:
        summary += "情感表达较为强烈。(Emotional expression is relatively intense.)"
    elif avg_intensity > 1.0:
        summary += "情感表达适中。(Emotional expression is moderate.)"
    else:
        summary += "情感表达较为平淡。(Emotional expression is relatively mild.)"
    
    return summary