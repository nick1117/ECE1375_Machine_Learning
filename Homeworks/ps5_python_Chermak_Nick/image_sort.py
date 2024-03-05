import os as os
import shutil as shutil
import numpy as np

def image_sort():
    rand = np.random.permutation(10) + 1
    #print(rand)
    #train = rand[:8]
    #test = rand[8:]
    test_dir = "input\\test"
    train_dir = "input\\train"
    for p in [test_dir, train_dir]:
        if os.path.isdir(p):
            shutil.rmtree(p)
        os.mkdir(p)
    for i in range(1,41,1):
        subject = f"input\\all\\s{i}"
        for j in rand[:8]:
            src = os.path.join(subject,f"{j}.pgm")
            dest = os.path.join("input\\train",f"{i}_{j}.pgm")
            shutil.copy(src,dest)
        for j in rand[8:]:
            src = os.path.join(subject,f"{j}.pgm")
            dest = os.path.join("input\\test",f"{i}_{j}.pgm")
            shutil.copy(src,dest)

        # for j, index in enumerate(rand):
        #     if(j+1<9):
        #         shutil.copyfileobj(os.listdir(f"input\\all\\s{i}\\{rand[j]}.pgm"),f"\\input\\train\\{i}_{rand[j]}.pgm") #(origin, destination)
        #     else:
        #         shutil.copyfileobj(os.listdir(f"input\\all\\s{i}\\{rand[j]}.pgm"),f"\\input\\test\\{i}_{rand[j]}.pgm") #(origin, destination)
    
    #randomize 1-40
    #take first 8 numbers
    #input to training

# if __name__ == "__main__":
#     image_sort()
    
    #randomize numbers
    #open each s folder using os.listdir
    #take random s# and sort into input\\train or input\\test - "PersonID_image#"