import os
import gdown
import torch
from transformers import DistilBertTokenizerFast
from emotion_model import EmotionClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "emotion__bert_model.pt"

file_id = "1MvwyuZE7TgsbOcAO0Lbl5Vg6zWdmovO-"

model_url = f"https://drive.google.com/uc?id={file_id}"

def model_download():
    if not os.path.exists(model_path):
        print("Downloading model from Google Drive...")

        gdown.download(
            model_url,
            model_path,
            quiet=False
        )

model_download()        

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

model = EmotionClassifier(n_features=6)
model.load_state_dict(torch.load(model_path, map_location=device,weights_only=False))
model = model.to(device)
model.eval()

emotion_mapping = ["Sadness","Anxiety","Anger","Burnout","Positive","Neutral"]

def prediction_emotion(text):

    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        probability = torch.sigmoid(output)

    probability = probability.squeeze().cpu().numpy()

    prediction = {}

    for emotion, prob in zip(emotion_mapping, probability):
        prediction[emotion] = float(prob)

    sorted_result = sorted(prediction.items(),key=lambda x:x[1],reverse=True)

    return dict(sorted_result)


# def margin_rule(prediction, margin=0.15):

#     sorted_emotions = sorted(
#         prediction.items(),
#         key=lambda x: x[1],
#         reverse=True
#     )

#     top_emotion, top_score = sorted_emotions[0]
#     second_score = sorted_emotions[1][1]

#     if top_score - second_score < margin:
#         return "Neutral"

#     return top_emotion


# if __name__ == "__main__":

#     text ="i name is priyanshu"

#     prediction = prediction_emotion(text)
#     # final_emotion = margin_rule(prediction)

#     print("Text:", text)
#     print("Probabilities:", prediction)
#     # print("Final Emotion:", final_emotion)
