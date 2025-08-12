from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
import io
import numpy as np
from PIL import Image
import requests
from werkzeug.utils import secure_filename
import logging
import cv2
import json
from datetime import datetime
import base64

# Try to import YOLO, but handle if ultralytics is not installed
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Running in demo mode.")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# OpenRouter API Configuration
OPENROUTER_API_KEY = "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"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
KNOT_MODEL = None
LEAF_MODEL = None
DISEASE_MODEL = None

# Visualization settings
KNOT_COLOR = (0, 0, 255)     # Red for knots
LEAF_COLOR = (0, 255, 0)     # Green for leaves  
TEXT_COLOR = (255, 255, 255) # White text
BOX_THICKNESS = 2
TEXT_THICKNESS = 1
TEXT_SCALE = 0.7

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    """Load the YOLO models"""
    global KNOT_MODEL, LEAF_MODEL, DISEASE_MODEL
    
    if not YOLO_AVAILABLE:
        logger.warning("YOLO not available - running in demo mode")
        return
    
    try:
        # Update these paths to your actual model locations
        KNOT_MODEL_PATH = 'models/KNOTS.pt'      
        LEAF_MODEL_PATH = 'models/leaves2.pt'    
        DISEASE_MODEL_PATH = 'models/classify1.pt' 
        
        # Check if model files exist
        models_to_load = [
            (KNOT_MODEL_PATH, 'KNOT_MODEL'),
            (LEAF_MODEL_PATH, 'LEAF_MODEL'),
            (DISEASE_MODEL_PATH, 'DISEASE_MODEL')
        ]
        
        for model_path, model_name in models_to_load:
            if os.path.exists(model_path):
                try:
                    model = YOLO(model_path)
                    globals()[model_name] = model
                    logger.info(f"{model_name} loaded successfully from {model_path}")
                except Exception as e:
                    logger.error(f"Error loading {model_name}: {str(e)}")
            else:
                logger.warning(f"Model file not found: {model_path}")
        
        if any([KNOT_MODEL, LEAF_MODEL, DISEASE_MODEL]):
            logger.info("At least one model loaded successfully")
        else:
            logger.warning("No models loaded - running in demo mode")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        logger.info("Running in demo mode - models not loaded")

def generate_ai_report(analysis_results):
    """Generate AI-powered report using OpenRouter API"""
    try:
        # Prepare analysis data for the AI
        analysis_summary = {
            'knot_count': analysis_results.get('knot_count', 0),
            'leaf_count': analysis_results.get('leaf_count', 0),
            'health_status': analysis_results.get('health_status', 'Unknown'),
            'confidence_avg': analysis_results.get('confidence_avg', 0),
            'disease_summary': analysis_results.get('disease_summary', {}),
            'diseases_detected': analysis_results.get('diseases_detected', [])
        }
        
        # Create prompt for AI report generation
        prompt = f"""
You are an expert agricultural consultant specializing in olive tree health. Based on the following analysis results from an AI detection system, generate a comprehensive report with recommendations.

Analysis Results:
- Olive knots detected: {analysis_summary['knot_count']}
- Leaves analyzed: {analysis_summary['leaf_count']}
- Overall health status: {analysis_summary['health_status']}
- Average confidence: {analysis_summary['confidence_avg']:.2%}
- Disease distribution: {json.dumps(analysis_summary['disease_summary'], indent=2)}

Individual leaf analysis:
{json.dumps(analysis_summary['diseases_detected'][:5], indent=2)}

Please provide:

1. **Executive Summary** (2-3 sentences about overall tree health)

2. **Detailed Findings** (specific diseases detected, severity, distribution)

3. **Risk Assessment** (immediate vs long-term risks)

4. **Treatment Recommendations** (specific actions to take)
   - Immediate actions (0-7 days)
   - Short-term actions (1-4 weeks)
   - Long-term management (1-6 months)

5. **Prevention Strategies** (how to avoid future issues)

6. **Monitoring Schedule** (when to check again)

7. **Economic Impact** (potential costs of treatment vs no action)

Format the response as a professional agricultural report that a farmer could act upon immediately.
"""
        
        # API request to OpenRouter
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "anthropic/claude-3.5-sonnet",  # You can change this model
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.3
        }
        
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        ai_report = result['choices'][0]['message']['content']
        
        # Create structured report data
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'analysis_summary': analysis_summary,
            'ai_generated_report': ai_report,
            'report_id': f"olive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'model_used': data['model']
        }
        
        return report_data
        
    except Exception as e:
        logger.error(f"Error generating AI report: {str(e)}")
        # Return fallback report
        return generate_fallback_report(analysis_results)

def generate_fallback_report(analysis_results):
    """Generate a basic report when AI service is unavailable"""
    knot_count = analysis_results.get('knot_count', 0)
    leaf_count = analysis_results.get('leaf_count', 0)
    health_status = analysis_results.get('health_status', 'Unknown')
    disease_summary = analysis_results.get('disease_summary', {})
    
    # Basic report template
    report = f"""
**OLIVE TREE HEALTH REPORT**
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**EXECUTIVE SUMMARY**
Analysis of {leaf_count} leaves detected {knot_count} olive knots. Overall health status: {health_status}.

**FINDINGS**
- Leaves Analyzed: {leaf_count}
- Olive Knots Detected: {knot_count}
- Disease Distribution:
"""
    
    for disease, count in disease_summary.items():
        percentage = (count / leaf_count * 100) if leaf_count > 0 else 0
        report += f"  • {disease}: {count} leaves ({percentage:.1f}%)\n"
    
    report += f"""

**BASIC RECOMMENDATIONS**
"""
    
    if knot_count > 0:
        report += "• Olive knots detected - consider pruning affected branches\n"
    
    if 'Healthy' not in disease_summary or disease_summary.get('Healthy', 0) < leaf_count * 0.8:
        report += "• Disease symptoms present - consult local agricultural extension office\n"
        report += "• Consider fungicide treatment if appropriate for detected diseases\n"
    
    if health_status == 'Healthy':
        report += "• Tree appears healthy - continue regular monitoring\n"
    
    report += """
• Monitor tree health regularly
• Ensure proper irrigation and drainage
• Maintain good air circulation around the tree

**NOTE: This is a basic automated report. For detailed recommendations, consult with a certified arborist or agricultural specialist.**
"""
    
    return {
        'timestamp': datetime.now().isoformat(),
        'analysis_summary': analysis_results,
        'ai_generated_report': report,
        'report_id': f"olive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'model_used': 'fallback_template',
        'is_fallback': True
    }

def image_to_base64(image_array):
    """Convert OpenCV image to base64 string for frontend display"""
    try:
        _, buffer = cv2.imencode('.jpg', image_array)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64}"
    except Exception as e:
        logger.error(f"Error converting image to base64: {str(e)}")
        return None

def demo_classification(image_path):
    """Demo classification when models are not available"""
    import random
    
    # Demo disease classes
    demo_diseases = [
        'Healthy', 'Anthracnose', 'Cercospora Leaf Spot', 
        'Peacock Spot', 'Sooty Mold', 'Olive Knot'
    ]
    
    # Generate random demo results
    predicted_disease = random.choice(demo_diseases)
    confidence = random.uniform(0.7, 0.95)
    
    # Create demo analysis
    analysis = {
        'knot_count': random.randint(0, 3),
        'leaf_count': random.randint(2, 8),
        'diseases_detected': [],
        'disease_summary': {},
        'health_status': 'Healthy' if predicted_disease == 'Healthy' else 'Disease Detected',
        'confidence_avg': confidence
    }
    
    # Generate demo leaf results
    for i in range(analysis['leaf_count']):
        leaf_disease = random.choice(demo_diseases)
        leaf_conf = random.uniform(0.6, 0.9)
        analysis['diseases_detected'].append({
            'leaf_id': i+1,
            'disease': leaf_disease,
            'confidence': leaf_conf,
            'bbox': [random.randint(50, 200), random.randint(50, 200), 
                    random.randint(250, 400), random.randint(250, 400)]
        })
        
        # Count diseases
        if leaf_disease in analysis['disease_summary']:
            analysis['disease_summary'][leaf_disease] += 1
        else:
            analysis['disease_summary'][leaf_disease] = 1
    
    return {
        'success': True,
        'predicted_class': predicted_disease,
        'confidence': confidence,
        'analysis': analysis,
        'demo_mode': True
    }

def process_olive_image(image_path):
    """
    Complete olive analysis pipeline:
    1. Detect olive knots
    2. Detect leaves  
    3. Classify leaf diseases
    4. Generate annotated results
    """
    try:
        # If models not available, return demo results
        if not all([KNOT_MODEL, LEAF_MODEL, DISEASE_MODEL]):
            logger.info("Models not available - returning demo results")
            return demo_classification(image_path)
        
        # Read input image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to RGB for model processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = {
            'knot_count': 0,
            'leaf_count': 0,
            'diseases_detected': [],
            'disease_summary': {},
            'health_status': 'Unknown',
            'confidence_avg': 0.0,
            'processed_images': {}
        }
        
        # Step 1: Detect Olive Knots
        knot_results = KNOT_MODEL(image_rgb)
        knot_detections = len(knot_results[0].boxes) if knot_results[0].boxes is not None else 0
        results['knot_count'] = knot_detections
        
        logger.info(f"Detected {knot_detections} knots")
        
        # Create image with knot detections
        knot_image = image.copy()
        if knot_results[0].boxes is not None:
            for box in knot_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf.item()
                
                cv2.rectangle(knot_image, (x1, y1), (x2, y2), KNOT_COLOR, BOX_THICKNESS)
                
                label = f"Knot: {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_THICKNESS)
                
                cv2.rectangle(knot_image, (x1, y1 - h - 10), (x1 + w, y1), KNOT_COLOR, -1)
                cv2.putText(knot_image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
        
        results['processed_images']['knots'] = image_to_base64(knot_image)
        
        # Step 2: Detect Leaves
        leaf_results = LEAF_MODEL(image_rgb)
        leaf_detections = len(leaf_results[0].boxes) if leaf_results[0].boxes is not None else 0
        results['leaf_count'] = leaf_detections
        
        logger.info(f"Detected {leaf_detections} leaves")
        
        # Create image with both knots and leaves
        combined_image = knot_image.copy()
        if leaf_results[0].boxes is not None:
            for box in leaf_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(combined_image, (x1, y1), (x2, y2), LEAF_COLOR, BOX_THICKNESS)
        
        results['processed_images']['combined'] = image_to_base64(combined_image)
        
        # Step 3: Classify Leaves and Create Final Image
        final_image = combined_image.copy()
        disease_counts = {}
        total_confidence = 0.0
        classified_leaves = 0
        
        if leaf_results[0].boxes is not None:
            for i, box in enumerate(leaf_results[0].boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Crop leaf region
                leaf_crop = image[y1:y2, x1:x2]
                
                if leaf_crop.size == 0:
                    continue
                
                try:
                    # Classify disease
                    disease_results = DISEASE_MODEL(leaf_crop)
                    
                    # Get classification results
                    disease_name = DISEASE_MODEL.names[disease_results[0].probs.top1]
                    disease_conf = disease_results[0].probs.top1conf.item()
                    
                    logger.info(f"Leaf {i+1}: {disease_name} ({disease_conf:.2f})")
                    
                    # Store disease information
                    results['diseases_detected'].append({
                        'leaf_id': i+1,
                        'disease': disease_name,
                        'confidence': disease_conf,
                        'bbox': [x1, y1, x2, y2]
                    })
                    
                    # Count disease occurrences
                    if disease_name in disease_counts:
                        disease_counts[disease_name] += 1
                    else:
                        disease_counts[disease_name] = 1
                    
                    total_confidence += disease_conf
                    classified_leaves += 1
                    
                    # Add disease label to final image
                    label = f"{disease_name}: {disease_conf:.2f}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_THICKNESS)
                    
                    cv2.rectangle(final_image, (x1, y1 - h - 10), (x1 + w, y1), LEAF_COLOR, -1)
                    cv2.putText(final_image, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
                    
                except Exception as e:
                    logger.error(f"Error classifying leaf {i+1}: {str(e)}")
                    continue
        
        results['processed_images']['final'] = image_to_base64(final_image)
        results['disease_summary'] = disease_counts
        
        # Calculate overall health status
        if classified_leaves > 0:
            results['confidence_avg'] = total_confidence / classified_leaves
            
            healthy_count = disease_counts.get('Healthy', 0)
            if healthy_count == classified_leaves:
                results['health_status'] = 'Healthy'
            elif healthy_count > classified_leaves / 2:
                results['health_status'] = 'Mostly Healthy'
            else:
                results['health_status'] = 'Disease Detected'
        else:
            results['health_status'] = 'No leaves analyzed'
        
        # Return in the expected format for the frontend
        return {
            'success': True,
            'predicted_class': results['health_status'],  # Main classification for compatibility
            'confidence': results['confidence_avg'],
            'analysis': results,  # Detailed analysis data
            'processed_images': results['processed_images'],
            'demo_mode': False
        }
        
    except Exception as e:
        logger.error(f"Error in olive image processing: {str(e)}")
        # Return demo results on error
        return demo_classification(image_path)

def download_image_from_url(url):
    """Download image from URL"""
    try:
        response = requests.get(url, timeout=10, stream=True)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            raise ValueError("URL does not point to an image")
        
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > 16 * 1024 * 1024:
            raise ValueError("Image file too large")
        
        temp_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_url_image.jpg')
        with open(temp_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return temp_filename
        
    except Exception as e:
        logger.error(f"Error downloading image from URL: {str(e)}")
        return None

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/classify', methods=['POST'])
def classify_image():
    """Main classification endpoint using YOLO pipeline"""
    try:
        image_path = None
        
        # Handle file upload
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename != '' and allowed_file(file.filename):
                try:
                    filename = secure_filename(file.filename)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{timestamp}_{filename}"
                    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(image_path)
                except Exception as e:
                    return jsonify({
                        'error': 'File save failed',
                        'message': f'Could not save uploaded file: {str(e)}'
                    }), 500
        
        # Handle URL input
        elif 'image_url' in request.form:
            image_url = request.form['image_url']
            image_path = download_image_from_url(image_url)
            if image_path is None:
                return jsonify({
                    'error': 'Invalid image URL',
                    'message': 'Could not download image from the provided URL'
                }), 400
        
        else:
            return jsonify({
                'error': 'No image provided',
                'message': 'Please provide an image file or URL'
            }), 400
        
        # Process the image through the pipeline
        try:
            analysis_results = process_olive_image(image_path)
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            return jsonify({
                'error': 'Image processing failed',
                'message': f'Could not process the image: {str(e)}'
            }), 500
        
        # Clean up temporary files
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
        except:
            pass
        
        # Log successful classification for debugging
        if analysis_results.get('analysis'):
            analysis = analysis_results['analysis']
            logger.info(f"Classification completed - Knots: {analysis.get('knot_count', 0)}, Leaves: {analysis.get('leaf_count', 0)}, Health: {analysis.get('health_status', 'Unknown')}")
        
        return jsonify(analysis_results)
        
    except Exception as e:
        logger.error(f"Unexpected error in classification: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred during classification'
        }), 500
        
        # Determine primary disease and overall confidence
        if 'analysis' in analysis_results:
            diseases_detected = analysis_results['analysis']['diseases_detected']
            if diseases_detected:
                best_prediction = max(diseases_detected, key=lambda x: x['confidence'])
                primary_disease = best_prediction['disease']
                primary_confidence = best_prediction['confidence']
            else:
                primary_disease = 'No leaves detected'
                primary_confidence = 0.0
        else:
            # Demo mode
            primary_disease = analysis_results.get('predicted_class', 'Unknown')
            primary_confidence = analysis_results.get('confidence', 0.0)
        
        # Format response for frontend compatibility
        response_data = {
            'success': True,
            'predicted_class': primary_disease,
            'confidence': primary_confidence,
            'analysis': analysis_results.get('analysis', {}),
            'processed_images': analysis_results.get('processed_images', {}),
            'timestamp': datetime.now().isoformat(),
            'demo_mode': analysis_results.get('demo_mode', False)
        }
        
        # Clean up temporary files
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
        except:
            pass
        
        logger.info(f"Classification completed: {primary_disease} ({primary_confidence:.2%})")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Unexpected error in classification: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred during classification'
        }), 500

@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    """Generate AI-powered report based on analysis results"""
    try:
        # Get analysis results from request
        analysis_data = request.json
        if not analysis_data:
            return jsonify({
                'error': 'No analysis data provided',
                'message': 'Please provide analysis results to generate a report'
            }), 400
        
        # Generate AI report
        report_data = generate_ai_report(analysis_data.get('analysis', {}))
        
        # Save report to file
        report_filename = f"{report_data['report_id']}.json"
        report_path = os.path.join(app.config['RESULTS_FOLDER'], report_filename)
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Report generated: {report_data['report_id']}")
        
        return jsonify({
            'success': True,
            'report': report_data,
            'report_id': report_data['report_id']
        })
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return jsonify({
            'error': 'Report generation failed',
            'message': f'Could not generate report: {str(e)}'
        }), 500

@app.route('/api/download-report/<report_id>')
def download_report(report_id):
    """Download report as JSON file"""
    try:
        report_filename = f"{report_id}.json"
        report_path = os.path.join(app.config['RESULTS_FOLDER'], report_filename)
        
        if not os.path.exists(report_path):
            return jsonify({
                'error': 'Report not found',
                'message': f'Report {report_id} does not exist'
            }), 404
        
        return send_file(
            report_path,
            as_attachment=True,
            download_name=f"olive_tree_report_{report_id}.json",
            mimetype='application/json'
        )
        
    except Exception as e:
        logger.error(f"Error downloading report: {str(e)}")
        return jsonify({
            'error': 'Download failed',
            'message': 'Could not download report'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'knot_model': KNOT_MODEL is not None,
            'leaf_model': LEAF_MODEL is not None,
            'disease_model': DISEASE_MODEL is not None
        },
        'yolo_available': YOLO_AVAILABLE,
        'openrouter_configured': bool(OPENROUTER_API_KEY),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get available disease classes from the loaded model"""
    if DISEASE_MODEL is not None:
        classes = list(DISEASE_MODEL.names.values())
        return jsonify({
            'classes': classes,
            'total_classes': len(classes)
        })
    else:
        # Demo classes
        demo_classes = [
            'Healthy', 'Anthracnose', 'Cercospora Leaf Spot', 
            'Peacock Spot', 'Sooty Mold', 'Olive Knot'
        ]
        return jsonify({
            'classes': demo_classes,
            'total_classes': len(demo_classes),
            'demo_mode': True
        })

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'error': 'File too large',
        'message': 'Maximum file size is 16MB'
    }), 413

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'error': 'Not found',
        'message': 'The requested resource was not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

if __name__ == '__main__':
    # Load the models on startup
    load_models()
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,  # Set to False in production
        threaded=True
    )