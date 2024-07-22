import os

if __name__ == "__main__":
    def remove_non_image_file(my_data_dir):
        image_extension = ('.png', '.jpg', '.jpeg')
        folders = os.listdir(my_data_dir)
        for folder in folders:
            files = os.listdir(my_data_dir + '/' + folder)
            print(files)
            # print(files)
            i = []
            j = []
            for given_file in files:
                if not given_file.lower().endswith(image_extension):
                    file_location = my_data_dir + '/' + folder + '/' + given_file
                    print(file_location)
                    os.remove(file_location)  # remove non image file
                    i.append(1)
                else:
                    j.append(1)
                    pass
            print(f"Folder: {folder} - has image file", len(j))
            print(f"Folder: {folder} - has non-image file", len(i))


    remove_non_image_file(my_data_dir='inputs/malaria_dataset/cell_images')

else:
    print("This is not a library")