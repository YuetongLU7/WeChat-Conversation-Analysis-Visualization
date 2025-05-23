"""
Main Flask application for WeChat Sentiment Analysis
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
import pandas as pd
from werkzeug.utils import secure_filename

from utils.data_processor import process_chat_data
from utils.report_generator import generate_pdf_report
from utils.sentiment_analyzer import analyze_sentiment

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'static/reports'
ALLOWED_EXTENSIONS = {'csv'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and process chat data"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique ID for this analysis
        analysis_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{analysis_id}_{filename}")
        file.save(file_path)
        
        try:
            # Process the chat data
            result = process_chat_data(file_path, analysis_id)
            
            # Analyze sentiment
            sentiment_result = analyze_sentiment(result['keywords'])
            
            # Generate report
            report_filename = f"{analysis_id}_report.pdf"
            report_path = os.path.join(REPORT_FOLDER, report_filename)
            generate_pdf_report(result, sentiment_result, report_path)
            
            # 处理emoji词云数据
            emoji_cloud_data = {}
            # 首先检查新版本的键名
            if 'emoji_symbol_cloud' in result:
                emoji_cloud_data = result.pop('emoji_symbol_cloud')
                print(f"Found emoji_symbol_cloud with keys: {emoji_cloud_data.keys()}")
            # 再检查旧版本的键名，保持向后兼容
            elif 'emoji_clouds' in result:
                emoji_cloud_data = result.pop('emoji_clouds')
                print(f"Found emoji_clouds with keys: {emoji_cloud_data.keys()}")
            
            print(f"Emoji cloud paths: {emoji_cloud_data}")
            
            # 返回API响应
            return jsonify({
                'success': True,
                'analysis_id': analysis_id,
                'data': result,
                'sentiment': sentiment_result,
                'emoji_clouds': emoji_cloud_data,  # 统一使用emoji_clouds键名返回emoji词云数据
                'report_url': f'/api/reports/{report_filename}'
            })
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error processing file: {str(e)}")
            print(f"Error details: {error_details}")
            return jsonify({'error': str(e), 'details': error_details}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/reports/<filename>')
def get_report(filename):
    """Serve generated PDF reports"""
    return send_from_directory(REPORT_FOLDER, filename)

@app.route('/api/test', methods=['GET'])
def test():
    """Test endpoint"""
    return jsonify({'message': 'Backend is working!'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8082) 