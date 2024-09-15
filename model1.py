import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Focal Loss function for addressing class imbalance
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        y_true = K.cast(y_true, K.floatx())
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (1 - y_pred)
        fl = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(fl)
    return focal_loss_fixed

# Import the data
column_names = ['ID', 'AGE', 'SEX', 'INF_ANAM', 'STENOK_AN', 'FK_STENOK', 'IBS_POST', 'IBS_NASL',
                'GB', 'SIM_GIPERT', 'DLIT_AG', 'ZSN_A', 'nr11', 'nr01', 'nr02', 'nr03', 'nr04',
                'nr07', 'nr08', 'np01', 'np04', 'np05', 'np07', 'np08', 'np09', 'np10', 'endocr_01',
                'endocr_02', 'endocr_03', 'zab_leg_01', 'zab_leg_02', 'zab_leg_03', 'zab_leg_04',
                'zab_leg_06', 'S_AD_KBRIG', 'D_AD_KBRIG', 'S_AD_ORIT', 'D_AD_ORIT', 'O_L_POST',
                'K_SH_POST', 'MP_TP_POST', 'SVT_POST', 'GT_POST', 'FIB_G_POST', 'ant_im', 'lat_im',
                'inf_im', 'post_im', 'IM_PG_P', 'ritm_ecg_p_01', 'ritm_ecg_p_02', 'ritm_ecg_p_04',
                'ritm_ecg_p_06', 'ritm_ecg_p_07', 'ritm_ecg_p_08', 'n_r_ecg_p_01', 'n_r_ecg_p_02',
                'n_r_ecg_p_03', 'n_r_ecg_p_04', 'n_r_ecg_p_05', 'n_r_ecg_p_06', 'n_r_ecg_p_08',
                'n_r_ecg_p_09', 'n_r_ecg_p_10', 'n_p_ecg_p_01', 'n_p_ecg_p_03', 'n_p_ecg_p_04',
                'n_p_ecg_p_05', 'n_p_ecg_p_06', 'n_p_ecg_p_07', 'n_p_ecg_p_08', 'n_p_ecg_p_09',
                'n_p_ecg_p_10', 'fibr_ter_01', 'fibr_ter_02', 'fibr_ter_03', 'fibr_ter_05',
                'fibr_ter_06', 'fibr_ter_07', 'fibr_ter_08', 'GIPO_K', 'K_BLOOD', 'GIPER_Na',
                'Na_BLOOD', 'ALT_BLOOD', 'AST_BLOOD', 'KFK_BLOOD', 'L_BLOOD', 'ROE']

data = pd.read_csv("/Users/boncui/Desktop/Projects/Personal Projects/Mydocardial infarction/MI.data", sep=',', na_values='?', header=None, names=column_names)

# Handle missing values for numerical columns (mean)
data.fillna(data.mean(), inplace=True)

# Encode the categorical 'SEX' column
label_encoder = LabelEncoder()
data['SEX'] = label_encoder.fit_transform(data['SEX'])

# Features (columns 2-112)
X = data.iloc[:, 1:77]  # Exclude ID column
y = data.loc[:, 'fibr_ter_01':'ROE']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert target values to binary (0 and 1)
y_train = y_train.apply(lambda x: (x > 0).astype(int))
y_test = y_test.apply(lambda x: (x > 0).astype(int))

# Apply SMOTE for each label separately and align the number of samples
X_train_resampled = []
y_train_resampled_list = []

smote = SMOTE(random_state=42)

# Resample each label individually using SMOTE and find max samples
max_samples = 0
for i in range(y_train.shape[1]):
    X_res, y_res = smote.fit_resample(X_train, y_train.iloc[:, i])
    max_samples = max(max_samples, len(X_res))  # Track the max number of samples generated

# Resample each label to match the maximum samples
for i in range(y_train.shape[1]):
    X_res, y_res = smote.fit_resample(X_train, y_train.iloc[:, i])
    if len(X_res) < max_samples:
        extra_samples_needed = max_samples - len(X_res)
        X_res = np.vstack([X_res, X_res[:extra_samples_needed]])
        y_res = np.concatenate([y_res, y_res[:extra_samples_needed]])
    X_train_resampled.append(X_res)
    y_train_resampled_list.append(y_res)

# Convert the resampled data into numpy arrays
X_train_resampled = np.array(X_train_resampled[0])  # Using the resampled feature set of the first target
y_train_resampled = np.column_stack(y_train_resampled_list)

# Build the model
num_targets = y.shape[1]
model = Sequential()
model.add(Input(shape=(X_train_resampled.shape[1],)))
model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(num_targets, activation='sigmoid'))  # Multi-label classification

# Compile the model with a lower learning rate and focal loss
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss=focal_loss(gamma=2., alpha=0.25), metrics=['accuracy', tf.keras.metrics.AUC()])

# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train_resampled, y_train_resampled, epochs=20, batch_size=32, validation_split=0.2, 
                    callbacks=[early_stopping])

# Predict on the test set
y_pred_prob = model.predict(X_test)

# Adjust the threshold dynamically for each complication
thresholds = [0.3] * num_targets  # You can fine-tune these thresholds based on validation data
y_pred_binary = np.zeros_like(y_pred_prob)
for i in range(num_targets):
    y_pred_binary[:, i] = (y_pred_prob[:, i] > thresholds[i]).astype(int)

# Evaluation for each complication
for i in range(num_targets):
    print(f"\nClassification report for complication {y.columns[i]}:")
    print(classification_report(y_test.iloc[:, i], y_pred_binary[:, i]))
    print(confusion_matrix(y_test.iloc[:, i], y_pred_binary[:, i]))

# Plot training & validation accuracy values
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.show()
