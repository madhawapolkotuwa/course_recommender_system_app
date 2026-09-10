# Course Recommender — Streamlit App (RecommenderNet v2)

This app trains the neural network embedding model (`RecommenderNet v2`) live
from the IBM online-course enrollment dataset, then recommends unseen courses
to a chosen user based on predicted rating scores.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

This will open the app in your browser (usually at `http://localhost:8501`).

## How it works

1. On load, the app fetches the course enrollment ratings and course catalog
   directly from the IBM S3 URLs (same datasets used in the notebooks).
2. It builds and trains `RecommenderNetV2` — an embedding model with a small
   feed-forward network on top, trained to predict normalized ratings.
3. Training is cached (`st.cache_resource`) so it only re-runs when you change
   a model setting (embedding size, epochs, batch size) in the sidebar.
4. You can enter any user ID from the dataset (or click "Random user") to see
   their enrolled courses and their top-N recommended, unseen courses.

## Notes

- **First load will take a minute or two** since the model trains from scratch
  in-session — there's no pre-saved model file. Lowering "Training epochs" in
  the sidebar speeds this up for quick testing.
- Model settings changed in the sidebar (embedding size, epochs, batch size)
  will trigger a full retrain; the same settings are cached so revisiting them
  is instant.
- User IDs must exist in the training dataset — the app will tell you if an
  entered ID isn't found. Use the "Random user" button to get a valid one.
