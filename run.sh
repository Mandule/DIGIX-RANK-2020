python data/download.py
unzip train_dataset.zip
rm train_dataset.zip
rm train_dataset.csv
rm test_dataset_A.csv
rm test_dataset_B.csv

python src/preprocess.py
python src/lgb.py
python src/xgb.py
python src/lgbranker.py
python src/xgbranker.py
python src/stacking.py