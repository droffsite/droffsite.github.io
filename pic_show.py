import matplotlib.pyplot as plt

def pic_show_raw(img, title='Image'):
    pic_show(img, title, vmin=img.min(), vmax=img.max());
    
def pic_show(img, title='Image', vmin=0, vmax=1):
    plt.gray();
    plt.imshow(img, vmin=vmin, vmax=vmax);
    plt.xticks([], []);
    plt.yticks([], []);
    plt.title(title);
    
    return plt.gca();