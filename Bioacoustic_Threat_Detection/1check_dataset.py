import os # we use os because our audio files is in folders in downloads.
dataset_path = r'C:\Users\POOJA\Downloads\FSM5' # path to show where the file is located
classes = os.listdir(dataset_path) # listdir - list everything inside this folder
print('Classes in the dataset are:' ,classes)

# Go through each class
for class_name in classes:
    class_path = os.path.join(dataset_path, class_name) # suppose a class_name = 'gunshot', then python combimes the dats location(C:\Users\POOJA\Downloads\FSM5) with gunshot(C:\Users\POOJA\Downloads\FSM5\Gunshot)
    
    # check whether it is actually a folder
    if os.path.isdir(class_path): # os.path.isdir - return files and folders and here it means, Only continue if this is a folder.
        # look inside the class folder
        files = os.listdir(class_path)
        print(class_name,':', len(files),'files') # len(files) - how many items are in the class(So we're finding the number of audio files in each class.)
