"""
PDF report generator for WeChat sentiment analysis
"""
import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime

# Register Chinese font
chinese_font_registered = False

# Windows font paths
font_paths = [
    'C:/Windows/Fonts/simsun.ttc',     # SimSun
    'C:/Windows/Fonts/simhei.ttf',     # SimHei
    'C:/Windows/Fonts/msyh.ttc',       # Microsoft YaHei
    'C:/Windows/Fonts/simkai.ttf',     # SimKai
    # Add relative path support
    os.path.join(os.path.dirname(__file__), '..', '..', 'static', 'fonts', 'simsun.ttc'),
    os.path.join(os.path.dirname(__file__), '..', '..', 'static', 'fonts', 'simhei.ttf'),
]

# Try to register a Chinese font
for font_path in font_paths:
    try:
        if os.path.exists(font_path):
            if 'simsun' in font_path.lower():
                pdfmetrics.registerFont(TTFont('SimSun', font_path))
                chinese_font_registered = True
                print(f"Registered Chinese font: SimSun from {font_path}")
                break
            elif 'simhei' in font_path.lower():
                pdfmetrics.registerFont(TTFont('SimHei', font_path))
                chinese_font_registered = True
                print(f"Registered Chinese font: SimHei from {font_path}")
                break
            elif 'msyh' in font_path.lower():
                pdfmetrics.registerFont(TTFont('MicrosoftYaHei', font_path))
                chinese_font_registered = True
                print(f"Registered Chinese font: MicrosoftYaHei from {font_path}")
                break
            elif 'simkai' in font_path.lower():
                pdfmetrics.registerFont(TTFont('SimKai', font_path))
                chinese_font_registered = True
                print(f"Registered Chinese font: SimKai from {font_path}")
                break
    except Exception as e:
        print(f"Error loading font {font_path}: {e}")

if not chinese_font_registered:
    print("WARNING: No Chinese font registered. Chinese characters may not display correctly in the PDF report.")
    # Create a simple fallback font directory
    try:
        os.makedirs(os.path.join(os.path.dirname(__file__), '..', '..', 'static', 'fonts'), exist_ok=True)
        print("Created fonts directory. Please place Chinese font files (simsun.ttc or simhei.ttf) in the static/fonts directory.")
    except Exception as e:
        print(f"Error creating fonts directory: {e}")

# Define the font to use for Chinese text
def get_chinese_font():
    """Get registered Chinese font name or fallback to Helvetica"""
    if 'SimSun' in pdfmetrics.getRegisteredFontNames():
        return 'SimSun'
    elif 'SimHei' in pdfmetrics.getRegisteredFontNames():
        return 'SimHei'
    elif 'MicrosoftYaHei' in pdfmetrics.getRegisteredFontNames():
        return 'MicrosoftYaHei'
    elif 'SimKai' in pdfmetrics.getRegisteredFontNames():
        return 'SimKai'
    else:
        return 'Helvetica'

def generate_pdf_report(analysis_results, sentiment_results, output_path):
    """
    Generate PDF report with analysis results
    
    Args:
        analysis_results: Dictionary with analysis results
        sentiment_results: Dictionary with sentiment analysis results
        output_path: Path to save the PDF report
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create PDF document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Get Chinese font
    chinese_font = get_chinese_font()
    
    # Create custom styles
    custom_styles = {
        'CustomTitle': ParagraphStyle(
            name='CustomTitle',
            fontName=chinese_font,
            fontSize=24,
            leading=30,
            alignment=1,  # Center
            spaceAfter=12
        ),
        'CustomHeading1': ParagraphStyle(
            name='CustomHeading1',
            fontName=chinese_font,
            fontSize=18,
            leading=22,
            spaceAfter=6
        ),
        'CustomHeading2': ParagraphStyle(
            name='CustomHeading2',
            fontName=chinese_font,
            fontSize=14,
            leading=18,
            spaceAfter=6
        ),
        'CustomNormal': ParagraphStyle(
            name='CustomNormal',
            fontName=chinese_font,
            fontSize=12,
            leading=14,
            spaceAfter=6
        )
    }
    
    # Add custom styles if they don't already exist
    for style_name, style in custom_styles.items():
        if style.name not in styles:
            styles.add(style)
    
    # Create content
    content = []
    
    # Add title
    content.append(Paragraph("微信聊天记录分析报告", styles['CustomTitle']))
    content.append(Paragraph("WeChat Chat Analysis Report", styles['CustomHeading1']))
    content.append(Spacer(1, 20))
    
    # Add generation date
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content.append(Paragraph(f"生成日期 (Generation Date): {now}", styles['CustomNormal']))
    content.append(Spacer(1, 20))
    
    # Add time analysis section
    content.append(Paragraph("1. 时间分析 (Time Analysis)", styles['CustomHeading1']))
    content.append(Spacer(1, 10))
    
    time_analysis = analysis_results.get('time_analysis', {})
    
    # Add time span information
    first_date = time_analysis.get('first_date', 'N/A')
    last_date = time_analysis.get('last_date', 'N/A')
    total_days = time_analysis.get('total_days', 0)
    total_messages = time_analysis.get('total_messages', 0)
    
    content.append(Paragraph(f"聊天时间跨度 (Chat Time Span): {first_date} 至 (to) {last_date}", styles['CustomNormal']))
    content.append(Paragraph(f"总天数 (Total Days): {total_days}", styles['CustomNormal']))
    content.append(Paragraph(f"总消息数 (Total Messages): {total_messages}", styles['CustomNormal']))
    
    if total_days > 0:
        avg_messages = total_messages / total_days
        content.append(Paragraph(f"日均消息数 (Average Messages per Day): {avg_messages:.2f}", styles['CustomNormal']))
    
    content.append(Spacer(1, 10))
    
    # Add monthly chart
    monthly_chart = time_analysis.get('monthly_chart')
    if monthly_chart and os.path.exists(monthly_chart):
        content.append(Paragraph("月度消息数量变化 (Monthly Message Count)", styles['CustomHeading2']))
        content.append(Spacer(1, 5))
        content.append(Image(monthly_chart, width=450, height=300))
        content.append(Spacer(1, 10))
    
    # Add hourly chart
    hourly_chart = time_analysis.get('hourly_chart')
    if hourly_chart and os.path.exists(hourly_chart):
        content.append(Paragraph("时均消息数量变化 (Hourly Message Distribution)", styles['CustomHeading2']))
        content.append(Spacer(1, 5))
        content.append(Image(hourly_chart, width=450, height=300))
        content.append(Spacer(1, 10))
    
    # Add calendar heatmap
    calendar_heatmap = analysis_results.get('calendar_heatmap')
    if calendar_heatmap and os.path.exists(calendar_heatmap):
        content.append(Paragraph("聊天记录热力图 (Chat Frequency Heatmap)", styles['CustomHeading2']))
        content.append(Spacer(1, 5))
        content.append(Image(calendar_heatmap, width=450, height=300))
        content.append(Spacer(1, 20))
    
    # Add word cloud section
    content.append(Paragraph("2. 词云分析 (Word Cloud Analysis)", styles['CustomHeading1']))
    content.append(Spacer(1, 10))
    
    # Add note about word clouds
    content.append(Paragraph("注：词云图以HTML格式保存，可在浏览器中查看。", styles['CustomNormal']))
    content.append(Paragraph("Note: Word clouds are saved in HTML format and can be viewed in a browser.", styles['CustomNormal']))
    content.append(Spacer(1, 20))
    
    # Add sentiment analysis section
    content.append(Paragraph("3. 情感分析 (Sentiment Analysis)", styles['CustomHeading1']))
    content.append(Spacer(1, 10))
    
    # Add emotion radar chart
    radar_chart = sentiment_results.get('radar_chart')
    if radar_chart and os.path.exists(radar_chart):
        content.append(Paragraph("情感雷达图 (Emotion Radar Chart)", styles['CustomHeading2']))
        content.append(Spacer(1, 5))
        content.append(Image(radar_chart, width=400, height=400))
        content.append(Spacer(1, 10))
    
    # Add sentiment summary
    summary = sentiment_results.get('summary', '')
    content.append(Paragraph("情感分析总结 (Sentiment Analysis Summary)", styles['CustomHeading2']))
    content.append(Spacer(1, 5))
    content.append(Paragraph(summary, styles['CustomNormal']))
    content.append(Spacer(1, 20))
    
    # Add top keywords section
    content.append(Paragraph("4. 高频词汇 (Top Keywords)", styles['CustomHeading1']))
    content.append(Spacer(1, 10))
    
    # Get top keywords
    keywords_data = analysis_results.get('keywords', {})
    original_keywords = keywords_data.get('original', [])[:20]
    translated_keywords = keywords_data.get('translated', [])[:20]
    frequencies = keywords_data.get('frequencies', [])[:20]
    
    # Create table data
    table_data = [["排名 (Rank)", "中文词汇 (Chinese)", "英文翻译 (English)", "频次 (Frequency)"]]
    
    for i, (original, translated, freq) in enumerate(zip(original_keywords, translated_keywords, frequencies)):
        table_data.append([str(i+1), original, translated, str(freq)])
    
    # Create table
    if len(table_data) > 1:
        table = Table(table_data, colWidths=[50, 150, 150, 80])
        
        # Add table style with Chinese font
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), chinese_font),  # Use Chinese font for header
            ('FONTNAME', (0, 1), (-1, -1), chinese_font),  # Use Chinese font for content
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        content.append(table)
    
    # Build PDF
    doc.build(content)
    
    return output_path 