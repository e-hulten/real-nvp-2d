import os
import imageio


def make_gif(distr_name, duration):
    png_dir = "results/"
    images = []
    sort = sorted(os.listdir(png_dir))
    for file_name in sort:
        if file_name.endswith(".png"):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))

    imageio.mimsave("gifs/" + distr_name + ".gif", images, duration=duration)
    print("Created gif!")

