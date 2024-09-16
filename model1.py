import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Add
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt

# Load the data
def load_data(file_path):
    column_names = [
        'ID', 'AGE', 'SEX', 'INF_ANAM', 'STENOK_AN', 'FK_STENOK', 'IBS_POST', 'IBS_NASL',
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
        'Na_BLOOD', 'ALT_BLOOD', 'AST_BLOOD', 'KFK_BLOOD', 'L_BLOOD', 'ROE'
    ]
    data = pd.read_csv(file_path, sep=',', header=None, names=column_names, na_values='?')
    return data

# Preprocess the data
def preprocess_data(data):
    # Separate features and target
    X = data.iloc[:, 1:77]  # Exclude ID column
    y = data.loc[:, 'fibr_ter_01':'ROE']
    
    # Handle missing values in features
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Encode categorical variables
    le = LabelEncoder()
    X['SEX'] = le.fit_transform(X['SEX'])
    
    # Feature engineering (example: create age groups)
    X['AGE_GROUP'] = pd.cut(X['AGE'], bins=[0, 40, 60, 80, 100], labels=[0, 1, 2, 3])
    X['AGE_GROUP'] = X['AGE_GROUP'].cat.add_categories([-1]).fillna(-1).astype(int)
    
    # Convert target values to binary (0 and 1)
    y = y.apply(lambda x: (x > 0).astype(int))
    
    # Handle missing values in target variables
    y = y.fillna(0)  # Fill NaN with 0 for target variables
    
    return X, y

# Feature selection
def select_features(X, y, k=50):
    numeric_X = X.select_dtypes(include=[np.number])
    selector = SelectKBest(f_classif, k=k)
    X_new = selector.fit_transform(numeric_X, y.iloc[:, 0])
    selected_features = numeric_X.columns[selector.get_support()]
    return X[selected_features]

# Apply SMOTEENN for resampling
def apply_smoteenn(X_train, y_train):
    smote_enn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
    return X_resampled, pd.DataFrame(y_resampled, columns=y_train.columns)

# Build the model with residual connections
def build_model(input_shape, output_shape):
    inputs = Input(shape=(input_shape,))
    x = Dense(256, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    residual = x
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Add()([x, residual])
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    outputs = [Dense(1, activation='sigmoid', name=f'output_{i}')(x) for i in range(output_shape)]
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Custom weighted loss function
def weighted_binary_crossentropy(y_true, y_pred):
    epsilon = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    weights = tf.where(tf.equal(y_true, 1), 
                       tf.ones_like(y_true) * tf.reduce_sum(1 - y_true) / (tf.reduce_sum(y_true) + epsilon),
                       tf.ones_like(y_true))
    bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    weighted_bce = bce * weights
    return tf.reduce_mean(weighted_bce)

# Load and preprocess the data
data = load_data("/Users/boncui/Desktop/Projects/Personal Projects/Mydocardial infarction/MI.data")
X, y = preprocess_data(data)
X = select_features(X, y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# K-fold cross-validation with Stratified KFold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for fold, (train_index, val_index) in enumerate(skf.split(X, y['fibr_ter_01']), 1):
    print(f"Fold {fold}")
    
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
    
    # Apply SMOTEENN
    X_train_resampled, y_train_resampled = apply_smoteenn(X_train_fold, y_train_fold)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_val_scaled = scaler.transform(X_val_fold)
    
    # Build and compile the model
    model = build_model(X_train_scaled.shape[1], y_train_resampled.shape[1])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=[weighted_binary_crossentropy] * y_train_resampled.shape[1],
                  metrics=['accuracy'] * y_train_resampled.shape[1])
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    
    # Train the model
    history = model.fit(X_train_scaled, 
                        [y_train_resampled.iloc[:, i] for i in range(y_train_resampled.shape[1])], 
                        epochs=100, 
                        batch_size=32, 
                        validation_data=(X_val_scaled, [y_val_fold.iloc[:, i] for i in range(y_val_fold.shape[1])]),
                        callbacks=[early_stopping, reduce_lr])
    
    # Evaluate the model
    y_pred = model.predict(X_val_scaled)
    y_pred_classes = (np.array(y_pred) > 0.5).astype(int)
    
    for i, col in enumerate(y.columns):
        print(f"\nClassification Report for {col}:")
        print(classification_report(y_val_fold.iloc[:, i], y_pred_classes[i]))
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss - Fold {fold}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['output_0_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_output_0_accuracy'], label='Validation Accuracy')
    plt.title(f'Model Accuracy - Fold {fold}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()