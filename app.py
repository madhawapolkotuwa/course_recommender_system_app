import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

st.set_page_config(page_title="Course Recommender (Neural Net)", layout="wide")

RATING_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-ML0321EN-Coursera/labs/v2/module_3/ratings.csv"
COURSE_GENRE_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/course_genre.csv"

RANDOM_STATE = 123


# ---------------- Data loading ----------------

@st.cache_data(show_spinner="Loading course enrollment ratings...")
def load_ratings():
    return pd.read_csv(RATING_URL)


@st.cache_data(show_spinner="Loading course catalog...")
def load_courses():
    df = pd.read_csv(COURSE_GENRE_URL)
    return df[["COURSE_ID", "TITLE"]]


# ---------------- Data processing (same logic as the notebook) ----------------

def process_dataset(raw_data):
    encoded_data = raw_data.copy()

    user_list = encoded_data["user"].unique().tolist()
    user_id2idx_dict = {x: i for i, x in enumerate(user_list)}
    user_idx2id_dict = {i: x for i, x in enumerate(user_list)}

    course_list = encoded_data["item"].unique().tolist()
    course_id2idx_dict = {x: i for i, x in enumerate(course_list)}
    course_idx2id_dict = {i: x for i, x in enumerate(course_list)}

    encoded_data["user"] = encoded_data["user"].map(user_id2idx_dict)
    encoded_data["item"] = encoded_data["item"].map(course_id2idx_dict)
    encoded_data["rating"] = encoded_data["rating"].values.astype("int")

    return encoded_data, user_id2idx_dict, user_idx2id_dict, course_id2idx_dict, course_idx2id_dict


def generate_train_test_datasets(dataset, random_state=123):
    min_rating = min(dataset["rating"])
    max_rating = max(dataset["rating"])

    dataset = dataset.sample(frac=1, random_state=random_state)
    x = dataset[["user", "item"]].values
    y = dataset["rating"].apply(lambda r: (r - min_rating) / (max_rating - min_rating)).values

    train_indices = int(0.8 * dataset.shape[0])
    test_indices = int(0.9 * dataset.shape[0])

    x_train, x_val, x_test = x[:train_indices], x[train_indices:test_indices], x[test_indices:]
    y_train, y_val, y_test = y[:train_indices], y[train_indices:test_indices], y[test_indices:]

    return x_train, x_val, x_test, y_train, y_val, y_test


# ---------------- Model (RecommenderNet v2, from the notebook) ----------------

class RecommenderNetV2(keras.Model):
    def __init__(self, num_users, num_items, embedding_size=32, **kwargs):
        super(RecommenderNetV2, self).__init__(**kwargs)

        self.user_embedding_layer = layers.Embedding(
            input_dim=num_users, output_dim=embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
            name="user_embedding_layer")
        self.user_bias = layers.Embedding(input_dim=num_users, output_dim=1, name="user_bias")

        self.item_embedding_layer = layers.Embedding(
            input_dim=num_items, output_dim=embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
            name="item_embedding_layer")
        self.item_bias = layers.Embedding(input_dim=num_items, output_dim=1, name="item_bias")

        self.dense1 = layers.Dense(64, activation="relu")
        self.dense2 = layers.Dense(32, activation="relu")
        self.dense3 = layers.Dense(1, activation="relu")

    def call(self, inputs):
        user_vector = self.user_embedding_layer(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        item_vector = self.item_embedding_layer(inputs[:, 1])
        item_bias = self.item_bias(inputs[:, 1])

        concat = tf.concat([user_vector, item_vector], axis=1)
        x = self.dense1(concat)
        x = self.dense2(x)
        x = self.dense3(x)
        x = x + user_bias + item_bias
        return tf.nn.relu(x)


@st.cache_resource(show_spinner="Training RecommenderNet (v2)... this can take a minute or two")
def train_model(_ratings_df, embedding_size, epochs, batch_size):
    encoded_data, user_id2idx, user_idx2id, course_id2idx, course_idx2id = process_dataset(_ratings_df)

    num_users = len(user_id2idx)
    num_items = len(course_id2idx)

    x_train, x_val, x_test, y_train, y_val, y_test = generate_train_test_datasets(
        encoded_data, random_state=RANDOM_STATE
    )

    model = RecommenderNetV2(num_users, num_items, embedding_size)
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )

    history = model.fit(
        x=x_train, y=y_train,
        batch_size=batch_size, epochs=epochs,
        validation_data=(x_val, y_val),
        verbose=0,
    )

    test_loss, test_rmse = model.evaluate(x_test, y_test, verbose=0)

    return {
        "model": model,
        "history": history.history,
        "test_loss": test_loss,
        "test_rmse": test_rmse,
        "user_id2idx": user_id2idx,
        "course_id2idx": course_id2idx,
        "num_users": num_users,
        "num_items": num_items,
    }


# ---------------- Recommendation logic ----------------

def get_recommendations(artifacts, ratings_df, courses_df, raw_user_id, top_n):
    model = artifacts["model"]
    user_id2idx = artifacts["user_id2idx"]
    course_id2idx = artifacts["course_id2idx"]

    if raw_user_id not in user_id2idx:
        return None, None

    user_idx = user_id2idx[raw_user_id]

    enrolled_items = set(ratings_df[ratings_df["user"] == raw_user_id]["item"].tolist())
    all_items = set(course_id2idx.keys())
    unseen_items = list(all_items.difference(enrolled_items))

    if not unseen_items:
        return enrolled_items, pd.DataFrame(columns=["COURSE_ID", "TITLE", "PREDICTED_SCORE"])

    unseen_idx = np.array([course_id2idx[i] for i in unseen_items])
    user_idx_arr = np.full_like(unseen_idx, user_idx)

    inputs = np.stack([user_idx_arr, unseen_idx], axis=1)
    preds = model.predict(inputs, verbose=0).flatten()

    rec_df = pd.DataFrame({"COURSE_ID": unseen_items, "PREDICTED_SCORE": preds})
    rec_df = rec_df.sort_values(by="PREDICTED_SCORE", ascending=False).head(top_n).reset_index(drop=True)
    rec_df = pd.merge(rec_df, courses_df, how="left", on="COURSE_ID")
    rec_df = rec_df[["COURSE_ID", "TITLE", "PREDICTED_SCORE"]]

    return enrolled_items, rec_df


# ---------------- UI ----------------

st.title("🎓 Course Recommender — Neural Network Embeddings (RecommenderNet v2)")
st.caption(
    "Trains a neural embedding model live from the IBM course enrollment dataset "
    "and recommends unseen courses for a chosen user."
)

with st.sidebar:
    st.header("Model settings")
    embedding_size = st.slider("Embedding size", min_value=8, max_value=64, value=32, step=8)
    epochs = st.slider("Training epochs", min_value=1, max_value=20, value=5, step=1)
    batch_size = st.select_slider("Batch size", options=[32, 64, 128, 256], value=64)
    top_n = st.slider("Number of recommendations", min_value=3, max_value=20, value=10)
    st.markdown("---")
    st.caption("Training is cached — it only reruns if you change these settings.")

ratings_df = load_ratings()
courses_df = load_courses()

st.write(
    f"Loaded **{ratings_df['user'].nunique():,}** users, "
    f"**{ratings_df['item'].nunique():,}** courses, and "
    f"**{len(ratings_df):,}** enrollment ratings."
)

artifacts = train_model(ratings_df, embedding_size, epochs, batch_size)

col1, col2 = st.columns(2)
with col1:
    st.metric("Test RMSE", f"{artifacts['test_rmse']:.4f}")
with col2:
    st.metric("Test Loss (MSE)", f"{artifacts['test_loss']:.4f}")

with st.expander("Training / validation loss curve"):
    fig, ax = plt.subplots()
    ax.plot(artifacts["history"]["loss"], label="Train loss")
    ax.plot(artifacts["history"]["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.legend()
    st.pyplot(fig)

st.markdown("---")
st.subheader("Get recommendations for a user")

sample_users = ratings_df["user"].drop_duplicates().sample(n=5, random_state=RANDOM_STATE).tolist()

if "user_input" not in st.session_state:
    st.session_state["user_input"] = str(sample_users[0])

col_a, col_b = st.columns([2, 1])
with col_a:
    st.session_state["user_input"] = st.text_input("Enter a user ID", value=st.session_state["user_input"])
with col_b:
    st.write("")
    st.write("")
    if st.button("🎲 Random user"):
        st.session_state["user_input"] = str(ratings_df["user"].sample(1).iloc[0])
        st.rerun()

st.caption(f"Example user IDs to try: {', '.join(str(u) for u in sample_users)}")

user_input = st.session_state["user_input"]

try:
    raw_user_id = int(user_input)
except ValueError:
    raw_user_id = None
    st.error("Please enter a valid numeric user ID.")

if raw_user_id is not None:
    enrolled_items, rec_df = get_recommendations(artifacts, ratings_df, courses_df, raw_user_id, top_n)

    if enrolled_items is None:
        st.warning(f"User ID {raw_user_id} was not found in the training dataset.")
    else:
        st.write(f"**User {raw_user_id}** has enrolled in **{len(enrolled_items)}** course(s).")

        enrolled_df = courses_df[courses_df["COURSE_ID"].isin(enrolled_items)]
        with st.expander("Show enrolled courses"):
            st.dataframe(enrolled_df, use_container_width=True)

        st.markdown(f"### Top {top_n} recommended (unseen) courses")
        st.dataframe(rec_df, use_container_width=True)
