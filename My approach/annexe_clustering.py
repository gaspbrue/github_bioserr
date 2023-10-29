import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pdb
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from matplotlib.colors import ListedColormap
from skimage.segmentation import slic
import skimage.segmentation as seg













def clustering_background(image, k=5):
    
    
    # on applique le k-Means clustering avec k=15 pour obtenir les nouveaux centroids de l'image sans background
    kmeans = KMeans(n_clusters=k, random_state=0)
    flattened_image = image.reshape((-1, image.shape[-1]))
    kmeans.fit(flattened_image)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    

    # on affiche l'image avec les pixels regroupés selon leurs clusters
    cluster_colors = ['#'+''.join([random.choice('0123456789ABCDEF') for i in range(6)]) for _ in range(k)]      
    clustered_image = labels.reshape(image.shape[:-1])
    plt.clf()
    plt.imshow(clustered_image, cmap=ListedColormap(cluster_colors))
    plt.title(f'Pixels Plantes regroupés par clusters ({k})')
    plt.legend()
    filename = f'Images_Clustering/Pixels_Plante_regroupés_par_clusters_{k}'
    plt.savefig(filename)

    plt.show()
    plt.clf()
    # On affiche les spectres des centroids
    for i in range(centroids.shape[0]): 
        plt.plot(centroids[i], label=f'Cluster {i + 1}', color=cluster_colors[i])
    plt.ylim(0, 1.2)
    plt.xlabel('Ordonnée (red = 138, nir = 200)')
    plt.ylabel('Valeurs')
    plt.title(f'Spectre des Centroids Plante, {k} clusters ')
    plt.legend()
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.8)  # Ajuster l'espace pour que la légende ne se chevauche pas avec l'image
    plt.grid(True)
    os.makedirs(f"Images_Clustering", exist_ok=True)
    filename = f'Images_Clustering/Spectres_Centroids_Plante_{k}'
    plt.savefig(filename)

    plt.show()
    
    return centroids, labels
   








def trier_clusters_crit_order_background(image, centroids, labels, k):
    
    # on applique la fonction Critorder pour avoir l'ordre souhaité des centroids
    critorder_values = []
    for centroid in centroids:
        mRENDVI_value = mRENDVI(centroid)
        PSRI_value = PSRI(centroid)
        critorder_value = Critorder(mRENDVI_value, PSRI_value)
        critorder_values.append(critorder_value)
        
    critorder_values_tries = np.sort(critorder_values)
    # on assigne les labels aux centroids en fonction de critorder_values
    indices_labels_tries = np.argsort(critorder_values) + 1     # on trie par ordre croissant    # on obtient les indices (entre 1 et k) par ordre croissant des labels selon crit_order
    
    pdb.set_trace()

    labels_tries = np.take(indices_labels_tries, labels)
    
    # Remise en forme du tableau 'labels_tries' pour qu'il corresponde à la forme de l'image
    image_regroupee_triee = labels_tries.reshape(image.shape[:-1])
    
    pdb.set_trace()

    
    # Affichage de l'image avec les pixels regroupés selon les clusters tries
    cluster_colors = ['#'+''.join([random.choice('0123456789ABCDEF') for i in range(6)]) for _ in range(k)]  
    clustered_image = labels.reshape(image.shape[:-1])
    plt.clf()
    plt.imshow(image_regroupee_triee, cmap=ListedColormap(cluster_colors))
    plt.title(f'Pixels Plantes regroupés par clusters tries selon crit_order ({k})')
    plt.legend()
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
    filename = f'Images_Clustering/Pixels_Plante_tries_par_Critorder_regroupés_par_clusters_{k}'
    plt.savefig(filename)
    """
    plt.draw()
    """
    plt.show()
    
    
        
    # on obtient les centroids tries selon crit_order
    centroids_tries = centroids[indices_labels_tries - 1]

    
    plt.clf()
    # On affiche les spectres des centroids tries
    for i in range(centroids_tries.shape[0]): 
        plt.plot(centroids_tries[i], label=f'Cluster {i + 1}', color=cluster_colors[i])
    plt.ylim(0, 1.2)
    plt.xlabel('Ordonnée (red = 138, nir = 200)')
    plt.ylabel('Valeurs')
    plt.title(f'Spectre des Centroids tries critorder, {k} clusters ')
    plt.legend()
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.8)  # Ajuster l'espace pour que la légende ne se chevauche pas avec l'image
    plt.grid(True)
    os.makedirs(f"Images_Clustering", exist_ok=True)
    filename = f'Images_Clustering/Spectres_Centroids_tries_critorder_Plante_{k}'
    plt.savefig(filename)
    """
    plt.draw()
    """
    plt.show()
    
    
    return labels_tries














def trier_clusters_ndvi_background(image, centroids, labels, k):
    
    # on applique la fonction ndvi pour avoir l'ordre souhaité des centroids
    ndvi_values = []
    for centroid in centroids:
        ndvi_value = ndvi(centroid)
        ndvi_values.append(ndvi_value)
       
    ndvi_values_tries = np.sort(ndvi_values)
    # on assigne les labels aux centroids en fonction de critorder_values
    indices_labels_tries = np.argsort(ndvi_values)     # on trie par ordre croissant    # on obtient les indices (entre 1 et 15) par ordre croissant des labels selon crit_order

    index_mapping = {value: index for index, value in enumerate(indices_labels_tries)}
    labels_tries = np.array([index_mapping[item] for item in labels])
 
    # Remise en forme du tableau 'labels_tries' pour qu'il corresponde à la forme de l'image
    image_regroupee_triee = labels_tries.reshape(image.shape[:-1])
    pdb.set_trace()
    
    # Affichage de l'image avec les pixels regroupés selon les clusters tries
    cluster_colors = ['#'+''.join([random.choice('0123456789ABCDEF') for i in range(6)]) for _ in range(k)]  
    clustered_image = labels.reshape(image.shape[:-1])
    plt.clf()
    plt.imshow(image_regroupee_triee, cmap=ListedColormap(cluster_colors))
    plt.title(f'Pixels Plantes regroupés par clusters tries selon ndvi ({k})')
    plt.legend()
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
    filename = f'Images_Clustering/Pixels_Plante_tries_par_ndvi_regroupés_par_clusters_{k}'
    plt.savefig(filename)
    """
    plt.draw()
    """
    plt.show()    
    
        
    # on obtient les centroids tries selon ndvi
    centroids_tries = centroids[indices_labels_tries - 1]

    plt.clf()
    # On affiche les spectres des centroids tries
    for i in range(centroids_tries.shape[0]): 
        plt.plot(centroids_tries[i], label=f'Cluster {i + 1}', color=cluster_colors[i])
    plt.ylim(0, 1.2)
    plt.xlabel('Ordonnée (red = 138, nir = 200)')
    plt.ylabel('Valeurs')
    plt.title(f'Spectre des Centroids tries selon ndvi, {k} clusters ')
    plt.legend()
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.8)  # Ajuster l'espace pour que la légende ne se chevauche pas avec l'image
    plt.grid(True)
    os.makedirs(f"Images_Clustering", exist_ok=True)
    filename = f'Images_Clustering/Spectres_Centroids_tries_ndvi_Plante_{k}'
    plt.savefig(filename)
    """
    plt.draw()
    """
    plt.show()
    
    
    return labels_tries

















def ndvi(spectrum):
    red_values = spectrum[138]
    nir_values = spectrum[200]
    ndvi_values = (nir_values - red_values) / (nir_values + red_values)
    return ndvi_values


def mRENDVI(spectrum):
    epsilon = 0.00000000000001
    r705 = spectrum[155]
    r750 = spectrum[177]  
    r445 = spectrum[31]
    mRENDVI = (r750 - r705) / (r750 + r705 - 2*r445 + epsilon)
    return mRENDVI

def PSRI(spectrum):
    r500 = spectrum[55]
    r680 = spectrum[143]  
    r750 = spectrum[178]
    PSRI = (r680 - r500) / r750
    return PSRI

def Critorder(mRENDVI_value, PSRI_value):
    Critorder = mRENDVI_value - PSRI_value
    return Critorder


