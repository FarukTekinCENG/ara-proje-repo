import numpy as np
from sklearn.metrics import classification_report

def compute_minority_metrics(logits, labels, label_encoder):
    preds = np.argmax(logits, axis=-1)

    report = classification_report(
        labels,
        preds,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    macro_f1 = report["macro avg"]["f1-score"]

    minority = ["Internship", "Temporary", "Volunteer"]

    minority_recalls = {
        cls: report[cls]["recall"]
        for cls in minority
        if cls in report
    }

    return macro_f1, minority_recalls
