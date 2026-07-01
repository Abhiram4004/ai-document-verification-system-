from sklearn.metrics import confusion_matrix, classification_report

def generate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    return cm, report
