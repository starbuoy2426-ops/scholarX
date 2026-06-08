Document Classifier

Usage:

Train on the sample data:

```bash
py doc_classifier/train.py
```

Or specify custom paths:

```bash
py doc_classifier/train.py --data path/to/data.csv --out path/to/model.joblib
```

Predicting from Python:

```python
from doc_classifier.model import predict
print(predict(["A new phone was announced"], model_path='doc_classifier/model.joblib'))
```

The training script saves the classifier to `doc_classifier/model.joblib`.
You can retrain the model any time with:

```bash
py doc_classifier/train.py
```

If you want probability scores as well:

```python
from doc_classifier.model import predict_proba
print(predict_proba(["A new phone was announced"], model_path='doc_classifier/model.joblib'))
```
