
# import settings
import os
import random
import shutil


patient_names = []
training_names = []

##############T2##############

for root, dirs, filenames in os.walk("/data/ak4966/MRI/data/new_data/T2"):
	for f in filenames:
		if f in os.listdir("/data/ak4966/MRI/data/new_data/T1_training/train"):
			shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/T2_training/train")
		elif f in os.listdir("/data/ak4966/MRI/data/new_data/T1_training/validation"):
			shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/T2_training/validation")
		elif f in os.listdir("/data/ak4966/MRI/data/new_data/T1_testing"):
			shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/T2_testing")
		else:
			shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/to_be_organized")


for root, dirs, filenames in os.walk("/data/ak4966/MRI/data/new_data/to_be_organized"):
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
	for root, dirs, filenames in os.walk("/data/ak4966/MRI/data/new_data/to_be_organized"):
		for f in filenames:
			if ".pkl" not in f:
				continue
			if patient_names[i] in f:
				if i<=split:
					training_names.append(patient_names[i])
				else:
					shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/T2_testing")

split_training = len(training_names)*.8

for i in range (len(training_names)):
	for root, dirs, filenames in os.walk("/data/ak4966/MRI/data/new_data/to_be_organized"):
		for f in filenames:
			if ".pkl" not in f:
				continue
			if training_names[i] in f:
				if i<=split_training:
					shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/T2_training/train")
				else:
					shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/T2_training/validation")

shutil.rmtree('/data/ak4966/MRI/data/new_data/to_be_organized')
os.makedirs('/data/ak4966/MRI/data/new_data/to_be_organized')
patient_names = []
training_names = []

##############T2_FLAIR##############

for root, dirs, filenames in os.walk("/data/ak4966/MRI/data/new_data/T2_FLAIR"):
	for f in filenames:
		if f in os.listdir("/data/ak4966/MRI/data/new_data/T1_training/train"):
			shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/T2_FLAIR_training/train")
		elif f in os.listdir("/data/ak4966/MRI/data/new_data/T1_training/validation"):
			shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/T2_FLAIR_training/validation")
		elif f in os.listdir("/data/ak4966/MRI/data/new_data/T1_testing"):
			shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/T2_FLAIR_testing")
		else:
			shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/to_be_organized")


for root, dirs, filenames in os.walk("/data/ak4966/MRI/data/new_data/to_be_organized"):
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
	for root, dirs, filenames in os.walk("/data/ak4966/MRI/data/new_data/to_be_organized"):
		for f in filenames:
			if ".pkl" not in f:
				continue
			if patient_names[i] in f:
				if i<=split:
					training_names.append(patient_names[i])
				else:
					shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/T2_FLAIR_testing")

split_training = len(training_names)*.8

for i in range (len(training_names)):
	for root, dirs, filenames in os.walk("/data/ak4966/MRI/data/new_data/to_be_organized"):
		for f in filenames:
			if ".pkl" not in f:
				continue
			if training_names[i] in f:
				if i<=split_training:
					shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/T2_FLAIR_training/train")
				else:
					shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/T2_FLAIR_training/validation")

shutil.rmtree('/data/ak4966/MRI/data/new_data/to_be_organized')
os.makedirs('/data/ak4966/MRI/data/new_data/to_be_organized')

##############T1_FLAIR##############

for root, dirs, filenames in os.walk("/data/ak4966/MRI/data/new_data/T1_FLAIR"):
	for f in filenames:
		if f in os.listdir("/data/ak4966/MRI/data/new_data/T1_training/train"):
			shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/T1_FLAIR_training/train")
		elif f in os.listdir("/data/ak4966/MRI/data/new_data/T1_training/validation"):
			shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/T1_FLAIR_training/validation")
		elif f in os.listdir("/data/ak4966/MRI/data/new_data/T1_testing"):
			shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/T1_FLAIR_testing")
		else:
			shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/to_be_organized")


for root, dirs, filenames in os.walk("/data/ak4966/MRI/data/new_data/to_be_organized"):
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
	for root, dirs, filenames in os.walk("/data/ak4966/MRI/data/new_data/to_be_organized"):
		for f in filenames:
			if ".pkl" not in f:
				continue
			if patient_names[i] in f:
				if i<=split:
					training_names.append(patient_names[i])
				else:
					shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/T1_FLAIR_testing")

split_training = len(training_names)*.8

for i in range (len(training_names)):
	for root, dirs, filenames in os.walk("/data/ak4966/MRI/data/new_data/to_be_organized"):
		for f in filenames:
			if ".pkl" not in f:
				continue
			if training_names[i] in f:
				if i<=split_training:
					shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/T1_FLAIR_training/train")
				else:
					shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/T1_FLAIR_training/validation")

shutil.rmtree('/data/ak4966/MRI/data/new_data/to_be_organized')
os.makedirs('/data/ak4966/MRI/data/new_data/to_be_organized')


##############T1_GD##############

for root, dirs, filenames in os.walk("/data/ak4966/MRI/data/new_data/T1_GD"):
	for f in filenames:
		if f in os.listdir("/data/ak4966/MRI/data/new_data/T1_training/train"):
			shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/T1_GD_training/train")
		elif f in os.listdir("/data/ak4966/MRI/data/new_data/T1_training/validation"):
			shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/T1_GD_training/validation")
		elif f in os.listdir("/data/ak4966/MRI/data/new_data/T1_testing"):
			shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/T1_GD_testing")
		else:
			shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/to_be_organized")


for root, dirs, filenames in os.walk("/data/ak4966/MRI/data/new_data/to_be_organized"):
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
	for root, dirs, filenames in os.walk("/data/ak4966/MRI/data/new_data/to_be_organized"):
		for f in filenames:
			if ".pkl" not in f:
				continue
			if patient_names[i] in f:
				if i<=split:
					training_names.append(patient_names[i])
				else:
					shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/T1_GD_testing")

split_training = len(training_names)*.8

for i in range (len(training_names)):
	for root, dirs, filenames in os.walk("/data/ak4966/MRI/data/new_data/to_be_organized"):
		for f in filenames:
			if ".pkl" not in f:
				continue
			if training_names[i] in f:
				if i<=split_training:
					shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/T1_GD_training/train")
				else:
					shutil.copy2(os.path.join(root, f), "/data/ak4966/MRI/data/new_data/T1_GD_training/validation")

shutil.rmtree('/data/ak4966/MRI/data/new_data/to_be_organized')
os.makedirs('/data/ak4966/MRI/data/new_data/to_be_organized')


