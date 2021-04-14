import os


def change_name():
    src_dirs = list(filter(lambda x: x[0] != '.', os.listdir("./dataset_characters")))
    for folder in src_dirs:
        print("Renaming folder" + folder)
        counter = 1
        folder_path = os.path.join("./dataset_characters", folder)
        images = os.listdir(folder_path)
        for image in images:
            image_path = os.path.join(folder_path, image)
            file_name = folder + "_" + str(counter).zfill(3) + ".jpg"
            new_name = os.path.join(folder_path, file_name)
            os.rename(image_path, new_name)
            counter += 1


def change_name1():
    src_dirs = list(filter(lambda x: x[0] != '.', os.listdir("./images")))
    counter = 1
    for image in src_dirs:
        print(image)
        old_name = os.path.join("./images", image)
        new_name = "Plate_" + str(counter).zfill(4) + ".jpg"
        new_path = os.path.join("./images", new_name)
        os.rename(old_name, new_path)
        counter += 1


if __name__ == "__main__":
    change_name()
