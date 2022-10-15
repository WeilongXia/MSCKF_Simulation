from cmath import pi
import numpy as np
import random
from numpy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # draw 3D points

class gen_map():
    '''Generate landmarks randomly distributed on a circumscribing cylinder of radius 6m'''

    # if __name__ == "main":
    #     self.gen_points()
    #     self.draw_points()
        
    def __init__(self):

        # radius of the cylinder
        self.radius = 6.0

        # height of the cylinder
        self.height = 8.0

        # number of the generated 3D points
        self.npoints = 3000

        # preallocate for the generated map
        self.map = np.empty((self.npoints, 3), dtype=np.float64)

    def gen_points(self):

        for i in range(self.npoints):

            theta = random.random() * 2.0 * pi

            # calculate x and y
            x = self.radius * np.cos(theta)
            y = self.radius * np.sin(theta)

            # calculate z
            z = self.height * random.random() - 4.0

            # print([x,y,z])
            # print(self.map)

            self.map[i, :] = [x, y, z]

    def draw_points(self):

        X = self.map[:, 0]
        Y = self.map[:, 1]
        Z = self.map[:, 2]

        # draw points
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(X, Y, Z)

        # add axis
        ax.set_zlabel('Z', fontdict = {'size': 15, 'color': 'red'})
        ax.set_ylabel('Y', fontdict = {'size': 15, 'color': 'red'})
        ax.set_xlabel('X', fontdict = {'size': 15, 'color': 'red'})
        plt.show()

    
# test gen_map
Gen_Map = gen_map()
Gen_Map.gen_points()
Gen_Map.draw_points()




