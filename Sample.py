from datetime import datetime

class Sample:
    def __init__(self, text, prediction_score=None, label=None, creation_datetime=None):
        self.text = text
        if prediction_score is not None:
            self.prediction_score = float(prediction_score)
        else:
            self.prediction_score = None
        if label is not None:
            self.labeled = True
            self.label = float(label)
        else:
            self.labeled = False
            self.label = None
        if creation_datetime is not None:
            self.creation_datetime = creation_datetime
        else:
            self.creation_datetime = datetime.now()

    def get_text_size_class(self):
        if len(self.text) < 20:
            return "sample-text-extra-large"
        elif len(self.text) < 40:
            return "sample-text-large"
        elif len(self.text) < 80:
            return "sample-text-medium"
        else:
            return "sample-text-small"

    def get_label_color(self):
        if self.prediction_score == None:
            return "rbga(0.6, 0.65, 0.8, 0.1)"
        red_val = 0.3 + (self.prediction_score * 0.2)
        green_val = 0.3 + (self.prediction_score * 0.2)
        if self.prediction_score <= 0.5:
            blue_val = 0.3 + (self.prediction_score * 0.3)
        else:
            blue_val = 0.3 + (self.prediction_score * 0.5)
        return "rgba({}, {}, {}, 1.0)".format(255 * red_val, 255 * green_val, 255 * blue_val)

    def get_human_readable_moral_sentiment_label(self):
        if self.prediction_score > 0.5:
            return "Morally Salient"
        else:
            return "Not Morally Salient"

    def get_human_readable_confidence_label(self):
        if self.prediction_score > 0.95:
            return "(High Confidence)"
        elif self.prediction_score > 0.7:
            return "(Medium Confidence)"
        elif self.prediction_score > 0.5:
            return "(Low Confidence)"
        elif self.prediction_score > 0.3:
            return "(Low Confidence)"
        elif self.prediction_score > 0.05:
            return "(Medium Confidence)"
        else:
            return "(Low Confidence)"

    def get_firebase_dict(self):
        firebase_dict = dict()
        firebase_dict["text"] = self.text
        firebase_dict["prediction_score"] = self.prediction_score
        firebase_dict["labeled"] = self.labeled
        firebase_dict["label"] = self.label
        firebase_dict["creation_timestamp"] = self.creation_datetime.timestamp()
        return firebase_dict

    @staticmethod
    def build_from_firebase_record(sample_firebase_record):
        text = sample_firebase_record["text"]
        if "prediction_score" in sample_firebase_record:
            prediction_score = sample_firebase_record["prediction_score"]
        else:
            prediction_score = None
        if "label" in sample_firebase_record:
            label = sample_firebase_record["label"]
        else:
            label = None
        creation_datetime = datetime.fromtimestamp(sample_firebase_record["creation_timestamp"])
        return Sample(text=text, prediction_score=prediction_score, label=label, creation_datetime=creation_datetime)
