import numpy as np
from scipy import spatial
from sklearn import neighbors
from data_handling import *

class Local_Outlier_Factor():
    '''
    Base class containing all the methods to extracts the outlier from
    the given dataset based on Local Outlier Factor Algorithm.
    '''
    ###################### CLASS ATTRIBUTES ########################
    #Attributes about all the points representation in space
    kdTree=None
    ball_tree=None
    #Attributes for the neighbourhood of each points
    k_distance=None
    k_nbr_dist=None
    k_nbr_indx=None
    #Local Recahbility density of points
    all_lrd=None


    ###################### MEMBER FUNCTIONS ########################
    #Initializer Function
    def __init__(self,k_value,points,labels):
        '''
        This function will initialize the attributes to be used for
        finding the outlier, like hyperparameters of the algorithm.
        '''
        #Setting the parameter governing the kth distance from a point
        self.k_value=k_value
        #The dataset point to perform the outlier detection
        self.points=points
        #Creating the kDTree from the points
        # print("Creating the KD Tree")
        # self.kdTree=spatial.cKDTree(points,
        #                                     leafsize=300000,
        #                                     compact_nodes=True,
        #                                     balanced_tree=True)
        #Creating the ball tree
        print("Creating the Ball tree")
        self.ball_tree=neighbors.BallTree(points,
                                            leaf_size=4)
        #Actual labels of the points to check the accuracy of method
        self.labels=labels

    #Function to find the kth distance
    def calculate_kth_distance(self,cache_path="cache/"):
        '''
        This function will find the kth nearest distance from the point
        by querying the
        '''
        print("\nFinding k-distance from each of the points")
        # distance,_=self.kdTree.query(self.points,k=[self.k_value+1])
        distance,_=self.ball_tree.query(self.points,
                                        k=self.k_value,
                                        return_distance=True,
                                        dualtree=True)
        print("shape of distance matrix:",distance.shape)
        # print(distance)
        self.k_distance=np.squeeze(distance[:,-1])

        #Saving the nearest neighbour distance in cache
        fname="{}_distance.npz".format(self.k_value)
        np.savez(cache_path+fname,k_distance=distance)

    #Function to get the k-neighbourhood
    def calculate_k_neighbourhood(self,cache_path="cache/"):
        '''
        This function will calculate the k-neighbourhood of the points
        according to their k-distance calculated.
        '''
        #Finding the k-neighbour of all the points
        print("\nFinding the k-neighborhood")
        indices,distance=self.ball_tree.query_radius(self.points,
                                                r=self.k_distance,
                                                return_distance=True)
        #Printing the result
        print("shape of neighbor: ",(distance).shape)
        # print("shape of neighbour"(indices).shape)
        #Assining the neighbors to the class attributes
        self.k_nbr_dist=distance
        self.k_nbr_indx=indices

        #Saving the neighbor information into cache
        fname="{}_neighbors.npz".format(self.k_value)
        np.savez(cache_path+fname,k_nbr_dist=distance,k_nbr_indx=indices)

    #Function to calculate the local rechability distance
    def calculate_local_rechability_density(self):
        '''
        Using the neighbors calculated we will now calculate the
        rechability density of the each of the points and classify
        them as outlier or inlier.
        '''
        #Initializing the prediciton array
        all_lrd=[]
        print("\nCalculating the Local-Rchability-Distance of points")
        #Now iterating ove all the points
        for pidx in range(self.points.shape[0]):
            #Calculating the number of neighbours for particular point
            nbr_indx=self.k_nbr_indx[pidx]
            num_nbr=nbr_indx.shape[0]-1

            #Adding up the distance of the rechability neighboures
            #Retreiving the neighbour and k distance from class
            nbr_dist=self.k_nbr_dist[pidx]
            k_dist=self.k_distance
            #Now adding rest of the rechablity distances
            reach_distance=np.maximum(nbr_dist,k_dist[nbr_indx])
            reach_distance=np.sum(reach_distance)-k_dist[pidx]

            #Now calculating the local_rechability density
            lrd=float(num_nbr)/reach_distance
            #Appending the lrd to all lrd array
            all_lrd.append(lrd)

        #Now assigning this lrd array to the class
        self.all_lrd=np.array(all_lrd)

    #Function to calculate the local outlier factor
    def get_the_outliers(self):
        '''
        This function will calculate the local outlier factor based
        on the local rechability density of all its neighbours
        with respect to its own local rechability density
        '''
        true_pos=0      #actuall pos and predicted pos
        false_neg=0     #actually pos but predicted neg
        false_pos=0     #actually neg but predicted pos
        true_neg=0      #actually neg and predicted neg

        #Iterating over all the points
        print("\nCalculating the Local-Outlier-Factor")
        for pidx in range(self.points.shape[0]):
            #Getting the actual labels
            actual_label=self.labels[pidx]

            #Calculating the lof of this point
            #Retreiving the lrd of itself
            self_lrd=self.all_lrd[pidx]
            #Retreiving the lrd of its neighbors
            nbr_indx=self.k_nbr_indx[pidx]
            if(nbr_indx.shape[0]<self.k_value-1):
                print(pidx,nbr_indx.shape)
            nbr_lrd=self.all_lrd[nbr_indx[nbr_indx!=pidx]]
            #Calculating the average neighbor lrd
            avg_nbr_lrd=np.mean(nbr_lrd)

            #Now calculating the lof
            lof=avg_nbr_lrd/self_lrd
            #Making the prediciton as outlier if lof>1
            prediction=int(lof>1)

            #Counting the confusion element 1: pos and 0:neg
            if(prediction==1 and actual_label==1):
                true_pos+=1
            elif(prediction==0 and actual_label==1):
                false_neg+=1
            elif(prediction==1 and actual_label==0):
                false_pos+=1
            elif(prediction==0 and actual_label==0):
                true_neg+=1

        #Printing the confusion matrix
        print("\nPrinting the confusion matrix:")
        print("\tPredict:1\tPredict:0")
        print("label:1\t{}\t{}".format(true_pos,false_neg))
        print("label:0\t{}\t{}\n".format(false_pos,true_neg))
        print("Precision: ",float(true_pos)/float(true_pos+false_pos))
        print("Recall: ",float(true_pos)/float(true_pos+false_neg))


if __name__=="__main__":
    #Getting the dataset into memory
    datapath="dataset/creditcard.csv"
    points,labels=get_points_and_labels(datapath)

    #Creating the LOF object
    my_lof=Local_Outlier_Factor(k_value=6,points=points,labels=labels)
    my_lof.calculate_kth_distance()
    my_lof.calculate_k_neighbourhood()
    my_lof.calculate_local_rechability_density()
    my_lof.get_the_outliers()
