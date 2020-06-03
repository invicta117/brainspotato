import math
import multiprocessing
import os
import shutil
import time
from random import sample

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom  # need gcdm
from scipy.ndimage import zoom
from tqdm import tqdm

from watershedSegmentation import get_pixels_hu, load_scan, seperate_skull

fileType = ".dcm"

MIN_BOUND = -100.0
MAX_BOUND = 1150.0
img_px_size = 50
#


def createTestFolder():
    testpath = str(os.getcwd() + "/TestDataFolder")
    try:
        os.mkdir(testpath)
    except OSError:
        print("This folder already exists!!")
    else:
        print("Created the Test Folder: " + testpath)

    return testpath


def populateTestFolder(directory, testpath, num):
    patientList = [patient for patient in os.listdir(
        directory) if os.path.isdir(directory + "/" + patient)]
    print(patientList)
    folderList = [directory + "/" + patient for patient in patientList]
    testpatients = sample(patientList, num)

    for patient in patientList:
        if patient in testpatients:
            moveTo = testpath + "/" + patient
            moveFrom = directory + "/" + patient
            shutil.move(moveFrom, moveTo)


def walkDirectory(directory):

    for file in os.listdir(directory):

        new_dir = os.path.join(directory, file)

        if os.path.isdir(new_dir):
            walkDirectory(new_dir)

        elif file.endswith(fileType):
            # print("Patient found at: " + directory + "!")
            patientFolder.append(directory)
            return

        else:
            print("No patient found in the directory: " + new_dir)
            return patientFolder

    return patientFolder

# create a dictionary of all the patient numbers and if they have mass effect, we will sum the mass effects
# values from the excel and use any values as > 0 as having a mass effect


def patientLabels(directory):
    column_list = ["R1:ICH", "R1:IPH"]
    df = pd.read_csv(directory + '/reads.csv', index_col=0)
    df = df[["R1:MassEffect", "R2:MassEffect", "R3:MassEffect"]]
    column_list = list(df)
    df["sum"] = df[column_list].sum(axis=1)
    name = df.index
    test = df["sum"]
    patient_dict = dict(zip(name, test))
    return patient_dict
# -----------------------------------------------------------


def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def processData(i):

    patientNum = paths[i].split("CQ500-CT-")[1].split("/")[0]
    patient = "CQ500-CT-" + patientNum
    label = patient_dict[str(patient)]
    # slices = [pydicom.read_file(paths[i] + "/" + s) for s in os.listdir(paths[i])]
    test_patient_scans = load_scan(paths[i])
    test_patient_images = get_pixels_hu(test_patient_scans)
    if len(test_patient_images) < 10:
        return
    slices = [seperate_skull(each_slice) for each_slice in test_patient_images]
    slices = [cv2.resize((each_slice), (img_px_size, img_px_size))
              for each_slice in slices]
    slices = [normalize(each_slice) for each_slice in slices]
    print("finsihing normlizing the data, you should see some patient and shape printing soon")
    final = []
    for each_slice in slices:
        if np.all(each_slice == each_slice[0]) | np.count_nonzero(each_slice) < (50 * 50 * 0.025):
            pass
        else:
            final.append(each_slice)

    if len(final) > 1:
        slices = np.stack(final, axis=0)
        slices = zoom(np.array(slices), (20 / len(slices), 1, 1))
        print(np.array(slices).shape)
        print(patient)
        # print(label)
        if label > 0:
            label = np.array([0, 1])
        else:
            label = np.array([1, 0])

        return slices, label
    else:
        print("sigh")
        return

    # return slices


if __name__ == "__main__":
    directory = input("Please enter the data directory: ")
    patient_dict = patientLabels(directory)
    print("Finished getting labels")

    print("Creating test patinet folder....")
    testfolder = createTestFolder()
    patientFolder = []
    paths = walkDirectory(testfolder)
    if len(paths) > 0:
        print("Data has already been split into test and train!!")
    else:
        numTest = input("Please enter the number of test patients: ")
        populateTestFolder(directory, testfolder, int(numTest))
        print("Completed creating and populating test data folder!!")

    print("Initializing walking test directory...")
    patientFolder = []
    paths = walkDirectory(testfolder)
    print("Completed finding all patient scans!!")
    print("Initializing processing training data...")
    start = time.time()
    pool = multiprocessing.Pool()
    L = pool.map(processData, range(len(paths)))
    pool.close()
    end = time.time()
    np.save("test_data.npy", L)
    print("Time elapsed: " + str(end - start))
    print("Finished processing data!!")
    print("Test data scans: " + str(len(L)))

    print("Initializing walking training directory...")
    patientFolder = []
    paths = walkDirectory(directory)
    print("Completed finding all patient scans!!")
    print("Initializing processing test data...")
    pool = multiprocessing.Pool()
    L = pool.map(processData, range(len(paths)))
    pool.close()
    np.save('training_data.npy', L)
    print("Finished processing training data!!")
    print("Training data scans: " + str(len(L)))
