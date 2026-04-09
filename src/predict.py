import json, numpy as np, tensorflow as tf, sys
from tensorflow.keras.preprocessing import image as keras_image

def load_model_and_classes(model_path="models/plant_disease_model.h5",
                            classes_path="models/class_names.json"):
    model = tf.keras.models.load_model(model_path)
    with open(classes_path) as f:
        class_names = json.load(f)
    return model, class_names

def predict(img_path, model, class_names, img_size=(224,224), top_k=3):
    img = keras_image.load_img(img_path, target_size=img_size)
    arr = keras_image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    top = preds.argsort()[-top_k:][::-1]
    return [{"class": class_names[i].replace("___"," - ").replace("_"," "),
             "confidence": float(preds[i])} for i in top]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path>")
        sys.exit(1)
    model, class_names = load_model_and_classes()
    results = predict(sys.argv[1], model, class_names)
    for r in results:
        print(f"  {r['class']}: {r['confidence']*100:.2f}%")
