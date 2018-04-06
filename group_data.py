
# import settings
import os
import random
import shutil

patient_names = []
training_names = []

for root, dirs, filenames in os.walk("/data/ak4966/MRI/data/new_data/T1"):
	for f in filenames:
		if "meta" in f:
			continue
		if "alldata" in f:
			continue
		if ".pkl" not in f:
			continue
		patient_name = f.split(".")[0]
		print(patient_name)
		if patient_name in patient_names:
			continue
		else:
			patient_names.append(patient_name)


random.shuffle(patient_names)


split = len(patient_names)*.67

for i in range(len(patient_names)):
	for root, dirs, filenames in os.walk("/data/ak4966/MRI/data/new_data/T1"):
		for f in filenames:
			if ".pkl" not in f:
				continue
			if patient_names[i] in f:
				if i<=split:
					training_names.append(patient_names[i])
				else:
					shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/T1_testing")

split_training = len(training_names)*.8

for i in range (len(training_names)):
	for root, dirs, filenames in os.walk("/data/ak4966/MRI/data/new_data/T1"):
		for f in filenames:
			if ".pkl" not in f:
				continue
			if training_names[i] in f:
				if i<=split_training:
					shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/T1_training/train")
				else:
					shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/T1_training/validation")
