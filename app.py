from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os
import cv2
from collections import Counter

app = Flask(__name__)

# Load YOLO model
model = YOLO("best.pt")

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER



# Class Mapping

CLASS_STAGE_MAPPING = {
    "freshunripe": "Stage_1_Fresh_Unripe",
    "unripe": "Stage_2_Unripe",
    "ripe": "Stage_3_Ripe",
    "freshripe": "Stage_4_Fresh_Ripe",
    "overripe": "Stage_5_Overripe",
    "rotten": "Stage_6_Rotten"
}

# Days left + advice
STAGE_INFO = {
    "Stage_1_Fresh_Unripe": {
        "days": "7–10 days remaining ⏳",
        "summary": "🟢 Freshly harvested and completely unripe. Firm, green, and starch-rich.",
        "expectation": "🌤️ It will gradually turn yellow over the next few days as sugars develop.",
        "actions": [
            "📦 Store at room temperature away from direct sunlight.",
            "🚫 Do NOT refrigerate at this stage.",
            "🛒 Ideal for planned consumption later in the week.",
            "📊 Suitable for bulk buying and storage."
        ],
        "best_for": "🗓️ Future consumption and controlled ripening."
    },

    "Stage_2_Unripe": {
        "days": "5–7 days remaining ⏳",
        "summary": "🟢🟡 Transitioning from green to yellow. Still firm with mild sweetness.",
        "expectation": "🔄 Will become fully ripe within a few days.",
        "actions": [
            "🏠 Allow to ripen naturally at room temperature.",
            "🍎 Place in a paper bag with an apple to speed ripening.",
            "❄️ Avoid refrigeration until fully yellow."
        ],
        "best_for": "⏰ Timing ripeness or slightly firm preference."
    },

    "Stage_3_Ripe": {
        "days": "3–4 days remaining ⏳",
        "summary": "🟡 Perfectly ripe with balanced sweetness and soft texture.",
        "expectation": "😋 Ideal stage for fresh eating.",
        "actions": [
            "🍽️ Consume within the next few days.",
            "❄️ Refrigerate to slow further ripening if needed.",
            "🥣 Great for fruit bowls and snacks."
        ],
        "best_for": "🥗 Fresh daily consumption."
    },

    "Stage_4_Fresh_Ripe": {
        "days": "2–3 days remaining ⏳",
        "summary": "🟡✨ Very ripe with brown speckles and enhanced sweetness.",
        "expectation": "🍯 Sugars are fully developed, making it extra flavorful.",
        "actions": [
            "⏱️ Consume soon for best quality.",
            "🥤 Ideal for smoothies and desserts.",
            "❄️ Refrigerate if not consuming immediately."
        ],
        "best_for": "🍰 Baking and sweet recipes."
    },

    "Stage_5_Overripe": {
        "days": "1–2 days remaining ⚠️",
        "summary": "🟤 Heavily spotted or brown. Very soft and extremely sweet.",
        "expectation": "⚡ Will spoil quickly if not used soon.",
        "actions": [
            "🍞 Use immediately for baking.",
            "🧊 Freeze peeled bananas for smoothies.",
            "🚫 Avoid extended room temperature storage."
        ],
        "best_for": "🧁 Baking or freezing."
    },

    "Stage_6_Rotten": {
        "days": "No safe consumption time remaining 🚨",
        "summary": "⚫ Mushy, leaking, or producing fermented odor.",
        "expectation": "⚠️ Unsafe due to potential microbial growth.",
        "actions": [
            "🗑️ Discard immediately.",
            "🚫 Do not attempt to consume.",
            "🧼 Clean storage area if leakage occurred."
        ],
        "best_for": "❌ Disposal only."
    }
}


# Generate Smart Response

from collections import Counter

def generate_response(detections):
    """
    detections = list of tuples (stage_name, confidence)
    """

    if not detections:
        return "🤔 No bananas detected."

    stages = [d[0] for d in detections]
    stage_counts = Counter(stages)

    
    # SINGLE CLASS DETECTED
    
    if len(stage_counts) == 1:
        stage = stages[0]
        info = STAGE_INFO.get(stage)

        if not info:
            return "⚠️ Unknown ripeness stage detected."

        actions_formatted = "\n".join([f"• {a}" for a in info.get("actions", [])])

        response = f"""
🍌 Detection Result: {stage.replace("_", " ")}

⏳ Estimated Shelf Life:
{info.get('days', 'Not available')}

📝 Summary:
{info.get('summary', '')}

🔄 What to Expect:
{info.get('expectation', '')}

📦 Recommended Actions:
{actions_formatted}

🎯 Best For:
{info.get('best_for', '')}
"""

        # Trigger
        if stage in ["Stage_4_Fresh_Ripe", "Stage_5_Overripe", "Stage_6_Rotten"]:
            response += "\n\n💡 Best Practice: Once brown spots appear, check bananas daily to prevent spoilage and reduce food waste."

        return response.strip()

    
    # MULTIPLE CLASSES DETECTED
    
    response = "🍌 Multiple ripeness stages detected:\n"

    # Sort naturally by stage number
    sorted_stages = sorted(stage_counts.keys())

    for stage in sorted_stages:
        count = stage_counts[stage]
        info = STAGE_INFO.get(stage)

        if not info:
            continue

        response += f"""

🔹 {stage.replace("_", " ")} (Detected: {count})
   ⏳ Shelf Life: {info.get('days', '')}
   📝 {info.get('summary', '')}
"""

    response += """

📌 Smart Recommendation:
• Consume the ripest bananas first.
• Use overripe ones for baking or smoothies.
• Store unripe bananas at room temperature.
"""

    # Waste prevention trigger if any late-stage detected
    if any(stage in ["Stage_4_Fresh_Ripe", "Stage_5_Overripe", "Stage_6_Rotten"] for stage in stage_counts):
        response += "\n💡 Check bananas daily to reduce spoilage and food waste."

    return response.strip()


# Draw Bounding Boxes

def draw_boxes(image_path, results):
    image = cv2.imread(image_path)

    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            stage_name = CLASS_STAGE_MAPPING.get(class_name, class_name)

            label = f"{stage_name} ({conf*100:.1f}%)"

            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put label
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

            detections.append((stage_name, conf))

    output_path = os.path.join(
        app.config["OUTPUT_FOLDER"],
        "output_" + os.path.basename(image_path)
    )

    cv2.imwrite(output_path, image)

    return output_path, detections



# Routes

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    results = model(filepath)

    output_path, detections = draw_boxes(filepath, results)

    message = generate_response(detections)

    return jsonify({
        "image_url": output_path,
        "detections": detections,
        "message": message
    })


if __name__ == "__main__":
    app.run(debug=True)