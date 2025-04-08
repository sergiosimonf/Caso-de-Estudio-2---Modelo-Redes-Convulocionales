from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import base64
import os
import io
from PIL import Image

app = Flask(__name__)

# Almacenamiento para múltiples modelos
models = []
MODEL_PATHS = []

def load_keras_model(model_path, model_index=None):
    """Carga un modelo Keras desde un archivo .h5"""
    try:
        # Use the tf.keras.models.load_model function with compile=True
        loaded_model = tf.keras.models.load_model(model_path, compile=True)
        print(f"Model loaded successfully from: {model_path}")
        
        # Si se especifica un índice, actualiza ese modelo específico
        if model_index is not None and model_index < len(models):
            models[model_index] = loaded_model
            MODEL_PATHS[model_index] = model_path
        else:
            # Si no hay índice o el índice está fuera de rango, añadir como nuevo modelo
            models.append(loaded_model)
            MODEL_PATHS.append(model_path)
            
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

# Function to preprocess the image
def preprocess_image(image_data):
    # Convert image data to numpy format
    if isinstance(image_data, str) and image_data.startswith('data:image'):
        # Handle image data in base64 format
        image_data = image_data.split(',')[1]
        image_data = base64.b64decode(image_data)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    elif isinstance(image_data, np.ndarray):
        # Already a numpy array
        img = image_data
    else:
        # Handle uploaded file
        img = Image.open(io.BytesIO(image_data))
        img = np.array(img.convert('L'))  # Convert to grayscale

    # Ensure the image is 28x28
    img = cv2.resize(img, (28, 28))
    
    # Invert colors if the background is white (assuming digits are dark on light background)
    if np.mean(img) > 128:
        img = 255 - img
        
    # Normalize to [0,1]
    img = img.astype('float32') / 255.0
    
    # Reshape for the CNN model (keep as 28x28x1 for processing)
    img = img.reshape(28, 28, 1)
    
    return img

# Función para predicción con ensemble
def predict_with_ensemble(image_array):
    """
    Realiza una predicción ensemble sobre una imagen de dígito
    """
    if len(models) == 0:
        return {"error": "No models loaded"}
    
    # Asegurarse que la imagen tenga el formato correcto
    if image_array.shape != (28, 28, 1):
        image_array = image_array.reshape(28, 28, 1)
   
    # Normalizar si es necesario (ya debería estar normalizada)
    if image_array.max() > 1.0:
        image_array = image_array.astype('float32') / 255.0
   
    # Añadir dimensión de batch
    image_array = np.expand_dims(image_array, axis=0)
   
    # Predicciones de cada modelo
    predictions = []
    individual_results = []
    
    for i, model in enumerate(models):
        pred = model.predict(image_array, verbose=0)
        predictions.append(pred)
        
        # Obtener resultado individual de este modelo
        model_digit = int(np.argmax(pred))
        model_confidence = float(np.max(pred))
        
        # Guardar resultados individuales
        individual_results.append({
            "model_index": i,
            "model_path": MODEL_PATHS[i],
            "digit": model_digit,
            "confidence": model_confidence,
            "probabilities": pred[0].tolist()
        })
   
    # Promediar predicciones para el ensemble
    avg_pred = np.mean(predictions, axis=0)
    ensemble_digit = int(np.argmax(avg_pred))
    ensemble_confidence = float(np.max(avg_pred))
    
    # Crear resultados ordenados por confianza para el ensemble
    ensemble_results = []
    for i, prob in enumerate(avg_pred[0].tolist()):
        ensemble_results.append({"digit": i, "probability": float(prob)})
    
    # Ordenar resultados por probabilidad descendente
    ensemble_results.sort(key=lambda x: x["probability"], reverse=True)
    
    return {
        "ensemble": {
            "digit": ensemble_digit,
            "confidence": ensemble_confidence,
            "all_predictions": ensemble_results
        },
        "individual_models": individual_results
    }

@app.route('/')
def index():
    return render_template('index.html', models_loaded=len(models))

@app.route('/upload-model', methods=['POST'])
def upload_model():
    if 'model' not in request.files:
        return jsonify({"status": "error", "message": "No file was sent"})
    
    model_file = request.files['model']
    model_index = request.form.get('model_index')
    
    if model_file.filename == '':
        return jsonify({"status": "error", "message": "No file was selected"})
    
    if model_file:
        # Save the uploaded model
        temp_model_path = f"model_{len(models)}.h5"
        if model_index is not None:
            model_index = int(model_index)
            if model_index < len(models):
                temp_model_path = f"model_{model_index}.h5"
        
        model_file.save(temp_model_path)
        
        # Try to load the model
        if load_keras_model(temp_model_path, model_index if model_index is not None else None):
            return jsonify({
                "status": "success", 
                "message": f"Model loaded successfully (Total: {len(models)})",
                "models_loaded": len(models),
                "model_paths": MODEL_PATHS
            })
        else:
            return jsonify({"status": "error", "message": "Error loading the model. Verify it's a valid Keras/TensorFlow model."})

@app.route('/clear-models', methods=['POST'])
def clear_models():
    global models, MODEL_PATHS
    models = []
    MODEL_PATHS = []
    return jsonify({"status": "success", "message": "All models cleared", "models_loaded": 0})

@app.route('/predict', methods=['POST'])
def predict():
    if len(models) == 0:
        return jsonify({"status": "error", "message": "No models loaded. Please load at least one model first."})
    
    try:
        # Procesar la imagen
        if 'canvas_data' in request.form:
            # Predict from canvas data
            img = preprocess_image(request.form['canvas_data'])
        elif 'image' in request.files:
            # Predict from image file
            img = preprocess_image(request.files['image'].read())
        else:
            return jsonify({"status": "error", "message": "No image data provided"})
        
        # Convertir imagen a formato base64 para mostrar en frontend
        preprocessed_img_base64 = get_preprocessed_image_base64(img)
        
        # Hacer predicción con ensemble
        result = predict_with_ensemble(img)
        
        if "error" in result:
            return jsonify({"status": "error", "message": result["error"]})
        
        # Transformar formato para el frontend
        transformed_result = {
            "preprocessed_image": preprocessed_img_base64,
            "ensemble_prediction": str(result["ensemble"]["digit"]),
            "ensemble_confidence": result["ensemble"]["confidence"],
            "ensemble_probabilities": {str(pred["digit"]): pred["probability"] for pred in result["ensemble"]["all_predictions"]},
            "individual_results": []
        }
        
        # Transformar resultados individuales
        for idx, model_result in enumerate(result["individual_models"]):
            individual = {
                "prediction": str(model_result["digit"]),
                "confidence": model_result["confidence"],
                "probabilities": {str(i): prob for i, prob in enumerate(model_result["probabilities"])}
            }
            transformed_result["individual_results"].append(individual)
        
        return jsonify({
            "status": "success",
            "result": transformed_result
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": f"Error processing the image: {str(e)}"})

# Función auxiliar para convertir la imagen preprocesada a base64
def get_preprocessed_image_base64(img):
    # Convertir de [0,1] a [0,255]
    img_display = (img.reshape(28, 28) * 255).astype(np.uint8)
    
    # Crear imagen PIL
    pil_img = Image.fromarray(img_display)
    
    # Guardar en buffer
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    
    # Convertir a base64
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

@app.route('/model-status', methods=['GET'])
def model_status():
    return jsonify({
        "models_loaded": len(models),
        "model_paths": MODEL_PATHS
    })

if __name__ == '__main__':
    # Run the Flask application
    app.run(debug=True)