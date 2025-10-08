import os
import json
import time
import gc
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
import joblib

# -----------------------------
# CẤU HÌNH
# -----------------------------
CSV_PATH = 'iot23_combined_new.csv'
OUTPUT_DIR = 'outputs'
CHUNK_SIZE = 100_000        # kích thước mỗi phần khi đọc 6M dòng
SAMPLE_PER_CLASS = 40_000   # số dòng tối đa mỗi lớp để train RF/SVM
TEST_SIZE = 0.2
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUTPUT_DIR, 'run_log.txt')

# -----------------------------
# TIỆN ÍCH GHI LOG
# -----------------------------

def log(msg: str) -> None:
	print(msg)
	with open(LOG_PATH, 'a', encoding='utf-8') as f:
		f.write(msg + '\n')

# -----------------------------
# TIỀN XỬ LÝ
# -----------------------------

def preprocess_raw_df(df_raw: pd.DataFrame, label_encoder: LabelEncoder, fit_encoder: bool = False) -> pd.DataFrame:
	"""
	- Loại cột không cần thiết
	- Drop NA và trùng lặp
	- Mã hóa nhãn thành label_encoded
	"""
	cols_drop = ['Unnamed: 0', 'uid']
	df = df_raw.drop(columns=cols_drop, errors='ignore')
	df = df.dropna()
	df = df.drop_duplicates()
	# Mã hóa nhãn
	if 'label' not in df.columns:
		raise ValueError("Thiếu cột 'label' trong dữ liệu")
	if fit_encoder:
		df['label_encoded'] = label_encoder.fit_transform(df['label'])
	else:
		df['label_encoded'] = label_encoder.transform(df['label'])
	return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
	"""One-hot cho cột phân loại, loại bỏ địa chỉ IP ở dạng chuỗi."""
	ip_cols = ['id.orig_h', 'id.resp_h']
	categorical_cols = ['service', 'conn_state', 'local_orig', 'local_resp', 'history', 'proto']
	df2 = df.drop(columns=ip_cols, errors='ignore')
	df2 = pd.get_dummies(df2, columns=[c for c in categorical_cols if c in df2.columns], drop_first=True)
	# bỏ nhãn gốc
	if 'label' in df2.columns:
		df2 = df2.drop(columns=['label'])
	return df2


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
	for c in df.columns:
		if df[c].dtype == 'object':
			df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
	return df

# -----------------------------
# SAMPLING CHO RF/SVM
# -----------------------------

def stratified_sample_incremental(label_encoder: LabelEncoder) -> pd.DataFrame:
	"""
	Đọc CSV theo từng phần, cộng dồn và giữ tối đa SAMPLE_PER_CLASS cho mỗi lớp.
	Đảm bảo RAM ổn định với 6M dòng.
	"""
	log("Bắt đầu lấy mẫu phân tầng cho RF/SVM...")
	start = time.time()
	
	# Trước tiên, scan toàn bộ file để lấy tất cả các class unique
	log("Đang scan tất cả các class trong dữ liệu...")
	all_labels = set()
	for chunk in pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE, low_memory=False):
		if 'label' in chunk.columns:
			all_labels.update(chunk['label'].dropna().unique())
	
	# Fit label encoder với tất cả các class
	label_encoder.fit(list(all_labels))
	log(f"Tìm thấy {len(all_labels)} class: {list(all_labels)}")
	
	sample_df: pd.DataFrame = pd.DataFrame()
	for i, chunk in enumerate(pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE, low_memory=False)):
		log(f"- Đọc phần {i+1} (rows={len(chunk)})")
		# tiền xử lý nhẹ cho lấy mẫu
		df_pp = preprocess_raw_df(chunk, label_encoder, fit_encoder=False)
		df_pp = encode_features(df_pp)
		df_pp = ensure_numeric(df_pp)
		# cộng dồn
		sample_df = pd.concat([sample_df, df_pp], axis=0, ignore_index=True)
		# cắt gọn theo mỗi lớp
		if 'label_encoded' in sample_df.columns:
			sample_df = sample_df.groupby('label_encoded', group_keys=False).apply(
				lambda x: x.sample(n=min(len(x), SAMPLE_PER_CLASS), random_state=RANDOM_STATE)
			)
		log(f"  -> kích thước mẫu tạm thời: {sample_df.shape}")
		# giới hạn thêm để giữ RAM
		if len(sample_df) >= SAMPLE_PER_CLASS * 5:  # 5 lớp của IoT-23 subset
			break
	end = time.time()
	log(f"Hoàn tất lấy mẫu trong {end-start:.1f}s, kích thước mẫu cuối: {sample_df.shape}")
	return sample_df

# -----------------------------
# HUẤN LUYỆN RF/SVM TRÊN MẪU
# -----------------------------

def train_rf_svm_on_sample(sample_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
	log("Huấn luyện RandomForest & SVM trên mẫu phân tầng...")
	y = sample_df['label_encoded']
	X = sample_df.drop(columns=['label_encoded'])
	# Do get_dummies theo từng phần có thể làm lệch cột giữa chunks -> xuất hiện NaN
	# Điền 0 cho mọi giá trị thiếu trước khi xử lý tiếp
	X = X.fillna(0)
	# Impute trước khi chuẩn hóa để loại NaN
	numeric_cols = X.select_dtypes(include=[np.number]).columns
	imputer = SimpleImputer(strategy='constant', fill_value=0.0)
	X[numeric_cols] = imputer.fit_transform(X[numeric_cols])
	# chuẩn hóa
	scaler = StandardScaler()
	X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
	X[numeric_cols] = np.nan_to_num(X[numeric_cols], nan=0.0, posinf=0.0, neginf=0.0)
	# Loại bỏ các class có ít hơn 2 mẫu trước khi chia train/test
	class_counts = y.value_counts()
	valid_classes = class_counts[class_counts >= 2].index
	valid_mask = y.isin(valid_classes)
	X = X[valid_mask]
	y = y[valid_mask]
	
	log(f"Sau khi loại bỏ class có < 2 mẫu: {len(X)} mẫu, {len(valid_classes)} class")
	
	# chia dữ liệu
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
	)
	# lưu scaler và cột
	joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))
	with open(os.path.join(OUTPUT_DIR, 'feature_columns.json'), 'w', encoding='utf-8') as f:
		json.dump(list(X.columns), f, ensure_ascii=False, indent=2)
	# RF
	rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
	rf.fit(X_train, y_train)
	yp = rf.predict(X_test)
	rf_metrics = metrics_from_preds(y_test, yp, 'rf')
	joblib.dump(rf, os.path.join(OUTPUT_DIR, 'model_rf.pkl'))
	# SVM
	svm = SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
	svm.fit(X_train, y_train)
	yp2 = svm.predict(X_test)
	svm_metrics = metrics_from_preds(y_test, yp2, 'svm')
	joblib.dump(svm, os.path.join(OUTPUT_DIR, 'model_svm.pkl'))
	return {'rf': rf_metrics, 'svm': svm_metrics}

# -----------------------------
# INCREMENTAL LEARNING VỚI SGDClassifier
# -----------------------------

def run_incremental_learning(label_encoder: LabelEncoder) -> Dict[str, float]:
	log("Bắt đầu Incremental Learning (SGDClassifier) trên toàn bộ dữ liệu theo phần...")
	start_time = time.time()
	model = SGDClassifier(loss='log_loss', random_state=RANDOM_STATE)
	classes_seen = None
	# lấy một tập validation nhỏ từ mẫu đầu tiên để đánh giá tạm
	val_X, val_y = None, None
	# Định nghĩa cột features cố định từ scaler đã lưu
	with open(os.path.join(OUTPUT_DIR, 'feature_columns.json'), 'r', encoding='utf-8') as f:
		expected_columns = json.load(f)
	
	# Thu thập tất cả các class có thể có từ label_encoder
	all_classes = np.arange(len(label_encoder.classes_))
	
	for i, chunk in enumerate(pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE, low_memory=False)):
		log(f"IL - xử lý phần {i+1}")
		df_pp = preprocess_raw_df(chunk, label_encoder, fit_encoder=False)
		df_pp = encode_features(df_pp)
		df_pp = ensure_numeric(df_pp)
		y = df_pp['label_encoded']
		X = df_pp.drop(columns=['label_encoded'])
		X = X.fillna(0)
		
		# Đảm bảo X có cùng cột với expected_columns
		for col in expected_columns:
			if col not in X.columns:
				X[col] = 0
		X = X[expected_columns]  # Sắp xếp theo thứ tự cố định
		
		# Impute + chuẩn hóa theo từng phần
		num_cols = X.select_dtypes(include=[np.number]).columns
		imp = SimpleImputer(strategy='constant', fill_value=0.0)
		X[num_cols] = imp.fit_transform(X[num_cols])
		scaler = StandardScaler()
		X[num_cols] = scaler.fit_transform(X[num_cols])
		X[num_cols] = np.nan_to_num(X[num_cols], nan=0.0, posinf=0.0, neginf=0.0)
		
		# Sử dụng tất cả các class có thể có cho partial_fit
		if classes_seen is None:
			classes_seen = all_classes
			model.partial_fit(X, y, classes=classes_seen)
			# chuẩn bị validation
			val_X, _, val_y, _ = train_test_split(X, y, test_size=0.95, random_state=RANDOM_STATE, stratify=y)
		else:
			# Luôn truyền classes để đảm bảo consistency
			model.partial_fit(X, y, classes=classes_seen)
		# giải phóng
		del X, y, df_pp, chunk
		gc.collect()
	# đánh giá trên validation nhỏ
	metrics = {'accuracy': None, 'precision': None, 'recall': None, 'f1': None}
	if val_X is not None and val_y is not None:
		pred = model.predict(val_X)
		m = metrics_from_preds(val_y, pred, 'il', save_confusion=True)
		metrics = m
	# lưu model cuối
	joblib.dump(model, os.path.join(OUTPUT_DIR, 'model_il_sgd.pkl'))
	elapsed = time.time() - start_time
	log(f"Hoàn tất IL trong {elapsed/60:.1f} phút")
	return metrics

# -----------------------------
# TÍNH METRIC VÀ LƯU CONFUSION MATRIX
# -----------------------------

def metrics_from_preds(y_true, y_pred, name: str, save_confusion: bool = True) -> Dict[str, float]:
	acc = accuracy_score(y_true, y_pred)
	prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
	cm = confusion_matrix(y_true, y_pred)
	if save_confusion:
		pd.DataFrame(cm).to_csv(os.path.join(OUTPUT_DIR, f'confusion_{name}.csv'), index=False)
	# lưu chi tiết report
	rep = classification_report(y_true, y_pred, digits=4, zero_division=0)
	with open(os.path.join(OUTPUT_DIR, f'metrics_{name}.txt'), 'w', encoding='utf-8') as f:
		f.write(rep)
	return {'accuracy': float(acc), 'precision': float(prec), 'recall': float(rec), 'f1': float(f1)}

# -----------------------------
# MAIN
# -----------------------------

def main():
	open(LOG_PATH, 'w').close()  # reset log
	log("BẮT ĐẦU CHẠY PIPELINE")
	if not os.path.exists(CSV_PATH):
		log(f"KHÔNG TÌM THẤY FILE: {CSV_PATH}")
		return
	# label encoder chung
	le = LabelEncoder()
	# 1) Lấy mẫu phân tầng -> train RF & SVM
	sample_df = stratified_sample_incremental(le)
	rf_svm_metrics = train_rf_svm_on_sample(sample_df)
	# 2) Incremental Learning trên toàn bộ dữ liệu
	il_metrics = run_incremental_learning(le)
	# 3) Tổng hợp
	summary = {
		'random_forest': rf_svm_metrics.get('rf', {}),
		'svm': rf_svm_metrics.get('svm', {}),
		'incremental_sgd': il_metrics
	}
	with open(os.path.join(OUTPUT_DIR, 'metrics_summary.json'), 'w', encoding='utf-8') as f:
		json.dump(summary, f, ensure_ascii=False, indent=2)
	log("ĐÃ LƯU TẤT CẢ MÔ HÌNH VÀ BÁO CÁO TẠI THƯ MỤC 'outputs'")
	log("KẾT THÚC")

if __name__ == '__main__':
	main()
