import os
import json
import logging
import random
import datetime
import numpy as np
import pandas as pd
from absl import app, flags
from tqdm.auto import tqdm
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from accelerate import Accelerator
from accelerate.logging import get_logger
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score
from tensorflow.keras.optimizers import AdamW
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import nltk
from nltk.corpus import wordnet
from imblearn.over_sampling import SMOTE

nltk.download('wordnet')

# Initialize logger
logger = get_logger('my_logger')

# Define flags for input arguments
FLAGS = flags.FLAGS

# Define flags for various input parameters
flags.DEFINE_string('data_dir', None, 'The directory of the data')
flags.DEFINE_string('feature_output', 'text_features.npy', 'Path to save extracted features')
flags.DEFINE_string('output_log_dir', '~/CCS/code/DistillBert/log', 'Output directory for logs')
flags.DEFINE_string('output_model_dir', '~/CCS/code/DistillBert/final_model', 'Output directory for models')
flags.DEFINE_string('task', None, 'Task to perform: train or predict')
flags.DEFINE_string('prediction_output', 'predictions.json', 'Path to save predictions')
flags.DEFINE_integer('epochs', 10, 'Number of training epochs')
flags.DEFINE_float('learning_rate', 5e-5, 'Learning rate for training')
flags.DEFINE_string('dev_data_dir', None, 'Directory for development data')
flags.DEFINE_string('train_split_file', None, 'Path to the training split file')
flags.DEFINE_string('dev_split_file', None, 'Path to the development split file')
flags.DEFINE_string('optimizer_choice', 'adamw', 'Choice of optimizer: adamw, or adam')
flags.DEFINE_string('sampling_strategy', 'over', 'Sampling strategy: over or under')
flags.DEFINE_string('new_data_dir', None, 'Path to the directory containing new data for prediction')
flags.DEFINE_string('model_path', None, 'Path to the trained model for prediction')

# Function to set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Function to load and clean data from a given file path
def load_and_clean_data(file_path):
    try:
        data = pd.read_csv(file_path, sep='\t')
        logger.info(f"Loaded data from {file_path} with shape {data.shape}")
        cleaned_data = data[data['speaker'] == 'Participant']
        logger.info(f"Cleaned data shape (only 'Participant' speaker): {cleaned_data.shape}")
        logger.info(f"First few rows of cleaned data: {cleaned_data.head()}")
        return cleaned_data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()

# Function to replace words with their synonyms
def synonym_replacement(text):
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = wordnet.synsets(random_word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            if synonym != random_word:
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
        if num_replaced >= 1:
            break
    return ' '.join(new_words)

# Function to preprocess text data using a tokenizer
def preprocess_text(texts, tokenizer, max_len=128, augment=False):
    texts = [str(text) for text in texts if isinstance(text, str) and text.strip()]
    if augment:
        texts = [synonym_replacement(text) for text in texts]
    tokens = tokenizer(texts, padding='max_length', truncation=True, max_length=max_len, return_tensors='tf')
    return tokens['input_ids'], tokens['attention_mask']

# Function to build a custom text model with multiple outputs
def build_custom_text_model():
    base_model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    input_ids = tf.keras.Input(shape=(128,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(128,), dtype=tf.int32, name="attention_mask")
    feature_count = tf.keras.Input(shape=(1,), dtype=tf.float32, name="feature_count")
    
    outputs = base_model.distilbert([input_ids, attention_mask])[0][:, 0, :]
    
    concat_features = tf.concat([outputs, feature_count], axis=-1)
    
    binary_output = Dense(2, activation='softmax', name='binary_output')(concat_features)
    multi_output = Dense(5, activation='softmax', name='multi_output')(concat_features)
    
    model = Model(inputs=[input_ids, attention_mask, feature_count], outputs=[binary_output, multi_output])
    
    return model

# Function to get PHQ-8 label
def get_phq8_label(score):
    if score <= 4:
        return 0
    elif score <= 9:
        return 1
    elif score <= 14:
        return 2
    elif score <= 19:
        return 3
    else:
        return 4

# Function to load and prepare data with sampling
def load_and_prepare_data(data_dir, tokenizer, labels_df, sampling_strategy='over'):
    all_input_ids = []
    all_attention_masks = []
    all_binary_labels = []
    all_multi_labels = []
    all_feature_counts = []

    for folder in tqdm(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            transcript_file = os.path.join(folder_path, f"{folder.replace('_P', '')}_TRANSCRIPT.csv")
            if os.path.exists(transcript_file):
                data = load_and_clean_data(transcript_file)
                if not data.empty:
                    texts = data['value']
                    input_ids, attention_masks = preprocess_text(texts, tokenizer, augment=(sampling_strategy=='over'))
                    all_input_ids.append(input_ids)
                    all_attention_masks.append(attention_masks)
                    participant_id = int(folder.split('_')[0])
                    score = labels_df[labels_df['Participant_ID'] == participant_id]['PHQ8_Score'].values[0]
                    binary_label = 1 if score >= 10 else 0
                    multi_label = get_phq8_label(score)
                    all_binary_labels.extend([binary_label] * len(input_ids))
                    all_multi_labels.extend([multi_label] * len(input_ids))
                    all_feature_counts.extend([len(input_ids)] * len(input_ids))

    if all_input_ids:
        input_ids = np.concatenate(all_input_ids, axis=0)
        attention_masks = np.concatenate(all_attention_masks, axis=0)
        binary_labels = np.array(all_binary_labels)
        multi_labels = np.array(all_multi_labels)
        feature_counts = np.array(all_feature_counts)

        if sampling_strategy == 'over':
            sampler = SMOTE(sampling_strategy='auto')
        elif sampling_strategy == 'under':
            sampler = RandomUnderSampler()
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

        input_ids_resampled, binary_labels_resampled = sampler.fit_resample(input_ids, binary_labels)
        attention_masks_resampled, _ = sampler.fit_resample(attention_masks, binary_labels)
        multi_labels_resampled, _ = sampler.fit_resample(multi_labels.reshape(-1, 1), binary_labels)
        feature_counts_resampled, _ = sampler.fit_resample(feature_counts.reshape(-1, 1), binary_labels)

        return input_ids_resampled, attention_masks_resampled, binary_labels_resampled, multi_labels_resampled.flatten(), feature_counts_resampled.flatten()
    else:
        raise ValueError("No valid data found for training.")


def build_optimizer(model, optimizer_choice, learning_rate):
    if optimizer_choice == 'adamw':
        optimizer = AdamW(learning_rate=learning_rate)
    elif optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer choice: {optimizer_choice}")
    return optimizer

# Function to train the model
def train_model(train_data_dir, dev_data_dir, model_output, epochs, learning_rate, train_split_file, dev_split_file, optimizer_choice, sampling_strategy):
    logger.info("Starting model training...")
    train_data_dir = os.path.expanduser(train_data_dir)
    dev_data_dir = os.path.expanduser(dev_data_dir)
    model_output = os.path.expanduser(model_output)
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_labels_df = pd.read_csv(train_split_file)
    dev_labels_df = pd.read_csv(dev_split_file)
    
    model = build_custom_text_model()
    
    optimizer = build_optimizer(model, optimizer_choice, learning_rate)
    
    model.compile(optimizer=optimizer, 
                  loss={'binary_output': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                        'multi_output': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)}, 
                  metrics={'binary_output': ['accuracy'], 'multi_output': ['accuracy'] } )

    train_input_ids, train_attention_masks, train_binary_labels, train_multi_labels, train_feature_counts = load_and_prepare_data(train_data_dir, tokenizer, train_labels_df, sampling_strategy=sampling_strategy)
    dev_input_ids, dev_attention_masks, dev_binary_labels, dev_multi_labels, dev_feature_counts = load_and_prepare_data(dev_data_dir, tokenizer, dev_labels_df, sampling_strategy='over')

    train_dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': train_input_ids, 'attention_mask': train_attention_masks, 'feature_count': train_feature_counts}, 
                                                        {'binary_output': train_binary_labels, 'multi_output': train_multi_labels}))
    dev_dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': dev_input_ids, 'attention_mask': dev_attention_masks, 'feature_count': dev_feature_counts}, 
                                                      {'binary_output': dev_binary_labels, 'multi_output': dev_multi_labels}))

    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(16)
    dev_dataset = dev_dataset.batch(16)

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch + 1}/{epochs}")
        model.fit(train_dataset, epochs=1, validation_data=dev_dataset, callbacks=[early_stopping])
        results = model.evaluate(dev_dataset)
        val_loss = results[0]
        val_binary_accuracy = results[3]
        val_multi_accuracy = results[4]

        y_true_binary = dev_binary_labels
        y_pred_binary = np.argmax(model.predict(dev_dataset)[0], axis=1)
        f1_binary = f1_score(y_true_binary, y_pred_binary, average='macro')
        
        y_true_multi = dev_multi_labels
        y_pred_multi = np.argmax(model.predict(dev_dataset)[1], axis=1)
        f1_multi = f1_score(y_true_multi, y_pred_multi, average='macro')
        
        logger.info(f"Epoch {epoch + 1}: Validation loss: {val_loss}, Validation binary accuracy: {val_binary_accuracy * 100:.5f}%, Validation multi accuracy: {val_multi_accuracy * 100:.5f}%, Binary F1 score: {f1_binary}, Multi F1 score: {f1_multi}")

    model.save(os.path.join(model_output, "distilbert_model_adamw"))
    logger.info(f"Model saved to {os.path.join(model_output, 'distilbert_model_adamw')}")

def predict(new_data_dir, model_path, prediction_output):
    logger.info("Starting prediction...")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.error(f"Error setting up GPU memory growth: {e}")

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = tf.keras.models.load_model(model_path)

    all_input_ids = []
    all_attention_masks = []
    all_feature_counts = []
    participant_ids = []

    new_data_dir = os.path.expanduser(new_data_dir)
    for folder in tqdm(os.listdir(new_data_dir)):
        folder_path = os.path.join(new_data_dir, folder)
        if os.path.isdir(folder_path):
            transcript_file = os.path.join(folder_path, f"{folder.replace('_P', '')}_TRANSCRIPT.csv")
            if os.path.exists(transcript_file):
                logger.info(f"Processing file: {transcript_file}")
                data = load_and_clean_data(transcript_file)
                if data.empty:
                    logger.info(f"No valid data found in {transcript_file}. Skipping...")
                    continue

                texts = data['value']
                input_ids, attention_masks = preprocess_text(texts, tokenizer)
                all_input_ids.append(input_ids)
                all_attention_masks.append(attention_masks)
                all_feature_counts.extend([len(texts)] * len(input_ids))
                participant_id = int(folder.split('_')[0])
                participant_ids.extend([participant_id] * len(input_ids))

    if all_input_ids:
        input_ids = np.concatenate(all_input_ids, axis=0)
        attention_masks = np.concatenate(all_attention_masks, axis=0)
        feature_counts = np.array(all_feature_counts)
        test_dataset = tf.data.Dataset.from_tensor_slices(
            ({'input_ids': input_ids, 'attention_mask': attention_masks, 'feature_count': feature_counts})
        ).batch(16)

        predictions = model.predict(test_dataset)
        binary_predictions = np.argmax(predictions[0], axis=1)
        multi_predictions = np.argmax(predictions[1], axis=1)

        # Prepare the dictionaries for JSON output
        binary_predictions_dict = {}
        multi_predictions_dict = {}
        for participant_id, binary_pred, multi_pred in zip(participant_ids, binary_predictions, multi_predictions):
            binary_predictions_dict[str(participant_id)] = int(binary_pred)
            multi_predictions_dict[str(participant_id)] = int(multi_pred)

        with open(prediction_output.replace(".json", "_binary.json"), "w") as f:
            json.dump(binary_predictions_dict, f)

        with open(prediction_output.replace(".json", "_multi.json"), "w") as f:
            json.dump(multi_predictions_dict, f)

        logger.info(f"Predictions saved to {prediction_output.replace('.json', '_binary.json')}")
        logger.info(f"Predictions saved to {prediction_output.replace('.json', '_multi.json')}")
    else:
        logger.info("No valid data found for prediction.")







def main(argv):
    set_seed(42)
    accelerator = Accelerator(log_with="wandb")

    if accelerator.is_main_process:
        now_time = datetime.datetime.now().strftime('%m-%d_%H-%M-%S')
        if not os.path.exists(os.path.expanduser(FLAGS.output_log_dir)):
            os.makedirs(os.path.expanduser(FLAGS.output_log_dir))
        if not os.path.exists(os.path.expanduser(FLAGS.output_model_dir)):
            os.makedirs(os.path.expanduser(FLAGS.output_model_dir))
        log_file = os.path.join(os.path.expanduser(FLAGS.output_log_dir), f'{now_time}.log')
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger.info(json.dumps(FLAGS.flag_values_dict(), indent=4))

    task = FLAGS.task

    if task == 'train':
        if FLAGS.data_dir and FLAGS.dev_data_dir and FLAGS.output_model_dir and FLAGS.train_split_file and FLAGS.dev_split_file:
            train_model(FLAGS.data_dir, FLAGS.dev_data_dir, FLAGS.output_model_dir, FLAGS.epochs, FLAGS.learning_rate, FLAGS.train_split_file, FLAGS.dev_split_file, FLAGS.optimizer_choice, FLAGS.sampling_strategy)
    elif task == 'predict':
        if FLAGS.new_data_dir and FLAGS.model_path:
            predict(FLAGS.new_data_dir, FLAGS.model_path, FLAGS.prediction_output)
    else:
        logger.info("Invalid task. Please specify 'train' or 'predict'.")

if __name__ == '__main__':
    flags.mark_flag_as_required('task')
    app.run(main)

