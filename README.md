# project

Big Data Course Project

# spark

spark-submit --py-files jieba.zip feature.py   
spark-submit --py-files jieba.zip feature_original.py  
spark-submit models.py LR LR.txt train test   
spark-submit models.py SVM SVM.txt train test   
spark-submit models.py LR LR_normed.txt train_normed test_normed   
spark-submit models.py LR SVM_normed.txt train_normed test_normed  