import os
import zipfile

fileType = ".zip"


def startUnzip(directory):

    print("Initializing unzipping of files for: " + directory)
    try:
        numFile = len(os.listdir(directory))
    except (FileNotFoundError, NotADirectoryError):
        print("No such directory as: " + str(directory) + "\nPlease try again!")
        quit()

    numFile = len(os.listdir(directory))
    for i, file in enumerate(os.listdir(directory)):
        if file.endswith(fileType):
            path = directory + "/" + file
            try:
                unzipfile = zipfile.ZipFile(path)
            except zipfile.BadZipFile:
                print(file + " Cannont be unzipped! Bad Zipfile deleting...")
            else:
                newPath = path[:-4]
                os.mkdir(newPath)
                unzipfile.extractall(newPath)
                unzipfile.close()
            finally:
                os.remove(path)
        print(f"Unzipping... : {i+1} / {numFile}")


if __name__ == "__main__":
    directory = input("Please enter the directory wish to unzip!: ")
    startUnzip(directory)
    print("Completed Unzipping!")
