import random

import cv2
import matplotlib.pyplot as plt
import numpy as np


def img_rotate(img, angle):
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows), borderValue=float((img).min()))
    return dst


def patient_rotate(patient):
    angle = random.choice([random.uniform(1, 20), random.uniform(-1, -20)])
    # print(angle)
    aslice = [img_rotate(slice, angle) for slice in patient[0]]
    aslice = np.stack(aslice, axis=0)
    # print(np.array(aslice).shape)
    return aslice, patient[1]


def augmentudlr(patient):
    aslice = [np.flip(slice).copy() for slice in patient[0]]
    aslice = np.stack(aslice, axis=0)
    # print(np.array(aslice).shape)
    return aslice, patient[1]


def augmentud(patient):
    aslice = [np.flipud(slice).copy() for slice in patient[0]]
    aslice = np.stack(aslice, axis=0)
    # print(np.array(aslice).shape)
    return aslice, patient[1]


def augmentlr(patient):
    aslice = [np.fliplr(slice).copy() for slice in patient[0]]
    aslice = np.stack(aslice, axis=0)
    # print(np.array(aslice).shape)
    # print(patient[1])
    return aslice, patient[1]


def augmentval(patient):
    #rint = list(range(1, 6)) + list(range(-5, 0))
    #rval = random.choice(rint)
    rval = random.gauss(0, 0.001)
    # print(rval)
    aslice = [rval + slice for slice in patient[0]]
    aslice = np.stack(aslice, axis=0)
    return aslice, patient[1]


def faugment(patient):
    patientlr = augmentlr(patient)
    #patientud = augmentud(patient)
    #patientudlr = augmentudlr(patient)
    return [patientlr]


def raugment(patient):
    return [patient_rotate(patient) for i in range(3)]


def valaugment(patient):
    return [augmentval(patient) for i in range(3)]


def questionHnadler(response):
    answer = str(response).strip().lower()
    if answer[:1] == "y":
        return True
    else:
        return False


def abstaract_augment(option, load_data, final_data):
    for i in range(len(load_data)):
        newAugdata = option(load_data[i])
        final_data = np.concatenate((final_data,  newAugdata), axis=0)
    return final_data


if __name__ == "__main__":
    print("Hello and welcome to the data augmentation Python program!!")
    training_dir = input("Please enter the training data file: ")
    load_data = np.load(training_dir, encoding='latin1')
    print(str(len(load_data)) + ": lenght of the dataset before the augmentation")

    final_data = load_data

    options = ["random rotation", "left-right flip", "value"]
    options_dict = {"random rotation": raugment,
                    "left-right flip": faugment, "value": valaugment}
    for i in range(len(options)):

        yn_answer = input("Would you like to include " +
                          options[i] + " augment? (y/n): ")

        if questionHnadler(yn_answer) == True:
            print("You have selected yes!")
            new_aug = options_dict.get(options[i])
            final_data = abstaract_augment(new_aug, load_data, final_data)
        else:
            pass

    print(str(len(final_data)) + ": lenght of the dataset after the augmentation")
    np.save("training_augmented_data.npy", final_data)
    print("Completed!!!")
