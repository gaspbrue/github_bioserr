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
from matplotlib.colors import ListedColormap, Normalize
from skimage.segmentation import slic
import skimage.segmentation as seg
import matplotlib.colors as mcolors 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches  
from skimage.color import rgb2lab
import cv2






def charger_imageHS(path_to_bil,path_to_hdr):
    data = np.fromfile(path_to_bil, dtype=np.float32)  # read binary data  # warning: dtype=np.float32 important ! (corresponds to datatype = 4 in the hdr file) !
    data = data.reshape((-1, 1))  # Remodeler les données en une colonne     
    # Load metadata from the .hdr file
    with open(path_to_hdr, 'r') as hdr_file:
        for line in hdr_file:
            if line.startswith('samples'):
                samples = int(line.split('=')[1])
            elif line.startswith('lines'):
                lines = int(line.split('=')[1])
            elif line.startswith('bands'):
                bands = int(line.split('=')[1])
                break
        data = data.reshape((lines, bands, samples))  # Reformat the data using bil interleave
        image = np.transpose(data, (0,2,1))  # Permute the axes to match the image's shape
        image = image[50:400, 190:620, :]
        image = np.nan_to_num(image)

    return image






def background_elimination(image, c, plante, cluster_colors_background_elimination, n_clusters=12, ndvi_threshold=0.5, nir_threshold=0.5):
    os.makedirs(f"Images/Plante_{plante}/Pixels_regroupes_par_clusters", exist_ok=True)
    os.makedirs(f"Images/Plante_{plante}/Spectres_Centroids", exist_ok=True)
    os.makedirs(f"Images/Plante_{plante}/Spectres_moyens", exist_ok=True)
        
    #K-Means clustering to separate plant and background pixels
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    flattened_image = image.reshape((-1, image.shape[-1]))
    kmeans.fit(flattened_image)

    # Get the cluster centroids and labels
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # We display the image with pixels grouped according to their clusters
    clustered_image = labels.reshape(image.shape[:-1])
    plt.clf()
    plt.imshow(clustered_image, cmap=ListedColormap(cluster_colors_background_elimination))
    plt.legend()
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.8)  # Adjust the spacing to prevent the caption from overlapping with the image
    plt.title(f'Pixels regroupés par clusters ({n_clusters}), Image {c}')
    filename = f'Images/Plante_{plante}/Pixels_regroupes_par_clusters/{n_clusters} clusters, Image {c}'
    plt.savefig(filename)
    plt.show()
    
    
    plt.clf()
    # We display the spectra of the centroids
    for i in range(centroids.shape[0]): 
        plt.plot(centroids[i], label=f'Cluster {i + 1}', color=cluster_colors_background_elimination[i])
    plt.ylim(0, 1.2)
    plt.xlabel('Bandes (red = 138, nir = 190)')
    plt.ylabel('Intensite')
    plt.title(f'Spectre des Centroids, {n_clusters} clusters, Image {c} ')
    plt.legend()
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.8)  # Ajuster l'espace pour que la légende ne se chevauche pas avec l'image
    plt.grid(True)
    filename = f'Images/Plante_{plante}/Spectres_Centroids/{n_clusters} clusters, Image {c}'
    plt.savefig(filename)
    """
    plt.draw()
    """
    plt.show()
    
    
    # Step 2: Select plant pixels based on NDVI and NIR thresholds
    red_values = centroids[:, 138]
    nir_values = centroids[:, 190]
    ndvi_values = (nir_values - red_values) / (nir_values + red_values)
    plant_indices = np.logical_and(ndvi_values > ndvi_threshold, nir_values > nir_threshold)

    # Create a copy of the original image to insert the plant pixels and fill the rest with zeros
    image_with_plant = np.zeros_like(image)
    selected_cluster_indices = np.where(plant_indices)[0] 
    labels = labels.reshape((image.shape[0], image.shape[1]))
    mask = np.isin(labels, selected_cluster_indices)
    plant_pixels = np.copy(image)
    plant_pixels[~mask] = 0
    
    
    """
    # Alternative for the background elimination : A pixel that has all background neighboring pixels or all -1 belongs to the background
    def is_background(pixel_row, pixel_col):
        neighborhood = mask[max(0, pixel_row - 1):pixel_row + 2, max(0, pixel_col - 1):pixel_col + 2]
        return np.all(~neighborhood)
    
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if mask[row, col] and is_background(row, col):
                plant_pixels[row, col] = 0
    """
    
    
    print(f'nombre de clusters correspondant à la plante: {selected_cluster_indices}')

    # Filter the non-zero pixels of the plant
    non_zero_pixels = plant_pixels[plant_pixels.sum(axis=2) != 0]

    
    # Display the average spectrum of all plant clusters
    spectre_moyen = np.mean(non_zero_pixels, axis=0)
    plt.figure(figsize=(8, 6))
    plt.plot(spectre_moyen, label='Spectre moyen de la plante')
    plt.xlabel('Longueur d\'onde')
    plt.ylabel('Intensité')
    plt.title('Spectre moyen de tous les clusters de plantes')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    filename = f'Images/Spectres_moyens/Spectres_moyen_image_{c}'
    plt.savefig(filename)
    plt.show()  
    

    return plant_pixels, mask
 




# Clutering on the flattened concatenated images
def clustering_flattened(concatenated_images, c, plante, cluster_colors_clustering, image_shape, n_clusters_clustering=5):
    os.makedirs(f'Images/Plante_{plante}/Pixels_plante_regroupes_par_cluster', exist_ok=True)
    os.makedirs(f'Images/Plante_{plante}/Spectres_Centroids_Plante', exist_ok=True)
    
    concatenated_images_reshaped = concatenated_images.reshape(350, -1, concatenated_images.shape[-1])  # I need to reshape in order to obtain the indices of my clustered pixels, which will then be used to reconstruct my individual images
    concatenated_images_reshaped_array = np.array(concatenated_images_reshaped)
    # Retrieve non-zero pixels and their indices
    non_zero_indices = np.transpose(np.where(concatenated_images_reshaped_array.sum(axis=2) != 0))
    non_zero_pixels = concatenated_images_reshaped_array[non_zero_indices[:, 0], non_zero_indices[:, 1]]
    
    # Apply K-Means to the non-zero pixels
    kmeans = KMeans(n_clusters = n_clusters_clustering)
    cluster_labels = kmeans.fit_predict(non_zero_pixels)
    centroids = kmeans.cluster_centers_

    """    
    flattened_image = image.reshape((-1, image.shape[-1]))
    kmeans.fit(flattened_image)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    """
    
    
    # Create an output image with the clustered pixels
    output_image = np.zeros_like(concatenated_images_reshaped_array[:,:,0])  # Une seule dimension pour les valeurs des pixels
    for index, label in zip(non_zero_indices, cluster_labels):    # j'associe la valeur des labels aux pixels correspondants sur mon image concatenee
        output_image[index[0], index[1]] = label + 1  # Ajout de +1 pour éviter les valeurs nulles
    labels = output_image.reshape((output_image.shape[0] * output_image.shape[1]))
    labeled_images_list = []
    for x in range(1, c+1):
        flattened_image = concatenated_images[(x-1)*image_shape[0]*image_shape[1] : x*image_shape[0]*image_shape[1]]
        image_labels = labels[(x-1)*image_shape[0]*image_shape[1] : x*image_shape[0]*image_shape[1]]
        labeled_images_list.append(image_labels)
        
    for i, labeled_image in enumerate(labeled_images_list): 
        unique_values = np.unique(labeled_image)
        print(f"Image {i+1}, Valeurs uniques : {unique_values}")
        
        plt.imshow(labeled_image.reshape(image_shape[:-1]).astype(np.uint8), cmap=ListedColormap(cluster_colors_clustering))
        plt.legend()
        filename = f'Images/Plante_{plante}/Pixels_plante_regroupes_par_cluster/Image {i+1}'
        plt.title(f'Pixels Plantes regroupés par clusters ({n_clusters_clustering}), Image {i+1}')
        plt.savefig(filename)
        plt.show()

    
    plt.clf()
    # Display the spectra of the centroids for ALL combined images (common clustering)
    for i in range(centroids.shape[0]): 
        plt.plot(centroids[i], label=f'Cluster {i + 1}', color=cluster_colors_clustering[i+1])            # i+1 because I skip the first color associated with the background.
    plt.ylim(0, 1.2)
    plt.xlabel('Bandes (red = 138, nir = 190)')
    plt.ylabel('INtensite')
    plt.title(f'Spectre des Centroids Plante, {n_clusters_clustering} clusters ')
    plt.legend()
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.8)  
    plt.grid(True)
    filename = f'Images/Plante_{plante}/Spectres_Centroids_Plante/Spectres_Centroids_all.png'
    plt.savefig(filename)
    plt.show()
    
    return centroids, labels, labeled_images_list





# If we want to sort the centroids according to critorder
def trier_clusters_crit_order(image, concatenated_images, c, plante, cluster_colors_clustering, centroids, labels, labeled_images_list, k):
    os.makedirs(f'Images/Plante_{plante}/Pixels_Plante_tries_par_crit_order_regroupes_par_clusters', exist_ok=True)
    os.makedirs(f'Images/Plante_{plante}/Spectres_Centroids_tries_crit_order_Plante', exist_ok=True)
    
    critorder_values = []
    for centroid in centroids:
        mRENDVI_value = mRENDVI(centroid)
        PSRI_value = PSRI(centroid)
        critorder_value = Critorder(mRENDVI_value, PSRI_value)
        critorder_values.append(critorder_value)
       
    critorder_values_tries = np.sort(critorder_values)
    indices_clusters_tries = np.argsort(critorder_values)      # Obtain indices (from 1 to 15) in ascending order of labels according to crit_order

    # Labeling centroids based on ascending crit_order
    labeled_images_list_sorted = []
    for x in range(c):    
        sorted_labels = np.zeros_like(labeled_images_list[0])
        image_labels = labeled_images_list[x]
        for new_label in indices_clusters_tries:
            sorted_labels[image_labels == new_label + 1] = indices_clusters_tries.tolist().index(new_label) + 1      # 'sorted_labels[image_labels == new_label]' selectionne les emplacements dans sorted_labels ou l'expression image_labels == new_label est True              # on fait "new_label + 1" car les valeurs de cluster_laels_sorted sont comprises entre 0 et 5 alors que celles de image_labels entre 1 et 6 si on ne prend pas en compte le background ! 
        # Reconstruct the labels into the shape of the original image
        image_labels_reshaped = sorted_labels.reshape(image.shape[:2])      
        labeled_images_list_sorted.append(image_labels_reshaped)
    
    
    # Create the continuous colorbar
    cluster_colors_clustering_without_white = cluster_colors_clustering[1:]
    custom_cmap = ListedColormap(cluster_colors_clustering_without_white) 
    norm = Normalize(vmin=0, vmax=len(centroids))
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])  # Vous pouvez passer un tableau de valeurs si nécessaire
    
    # Display the images of clustered pixels sorted according to crit_order
    for i, labeled_image in enumerate(labeled_images_list_sorted):
        plt.clf()
        plt.imshow(labeled_image.astype(np.uint8), cmap=ListedColormap(cluster_colors_clustering))
        plt.title(f'Pixels Plante tries par crit_order regroupes par clusters, Image {i+1}')
        plt.colorbar(sm, label='Cluster')
        filename = f"Images/Plante_{plante}/Pixels_Plante_tries_par_crit_order_regroupes_par_clusters/Image {i+1}.png"  # Nommage des fichiers avec un indice
        plt.savefig(filename)
        plt.show()

    centroids_tries = centroids[indices_clusters_tries]
    plt.clf()
    # Display the spectra of centroids sorted by crit_order from ALL combined images (common clustering)
    for i in range(centroids.shape[0]): 
        plt.plot(centroids_tries[i], color=cluster_colors_clustering[i+1])           
    plt.ylim(0, 1.2)
    plt.xlabel('Bandes (red = 138, nir = 190)')
    plt.ylabel('Intensite')
    plt.title(f'Spectre des Centroids Plante tries par crit_order, {k} clusters ')
    plt.colorbar(sm, label='Cluster')
    plt.legend()
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.8)  
    plt.grid(True)
    filename = f'Images/Plante_{plante}/Spectres_Centroids_tries_crit_order_Plante/Spectres_Centroids_tries_crit_order_all.png'
    plt.savefig(filename)
    plt.show()   

    return labeled_images_list_sorted



# Same that trier_clusters_crit_order but with NDVI
def trier_clusters_ndvi(image, concatenated_images, c, plante, cluster_colors_clustering, centroids, labels, labeled_images_list, k):
    os.makedirs(f'Images/Plante_{plante}/Pixels_Plante_tries_par_ndvi_regroupes_par_clusters', exist_ok=True)
    os.makedirs(f'Images/Plante_{plante}/Spectres_Centroids_tries_ndvi_Plante', exist_ok=True)
    
    # on applique la fonction ndvi pour avoir l'ordre souhaité des centroids
    ndvi_values = []
    for centroid in centroids:
        ndvi_value = ndvi(centroid)
        ndvi_values.append(ndvi_value)
       
    # on assigne les labels aux centroids en fonction de ndvi
    ndvi_values_tries = np.sort(ndvi_values)
    indices_clusters_tries = np.argsort(ndvi_values)      # on trie par ordre croissant de ndvi     # on obtient les indices (entre 1 et 15) par ordre croissant des labels selon crit_order

    # Étiquetage des centroïdes en fonction de ndvi croissant
    labeled_images_list_sorted = []
    # Affecter les étiquettes aux pixels dans chaque image
    for x in range(c):    
        # Réorganiser les étiquettes selon l'ordre croissant de ndvi des centroïdes
        sorted_labels = np.zeros_like(labeled_images_list[0])
        image_labels = labeled_images_list[x]
        for new_label in indices_clusters_tries:
            sorted_labels[image_labels == new_label + 1] = indices_clusters_tries.tolist().index(new_label) + 1      # 'sorted_labels[image_labels == new_label]' selectionne les emplacements dans sorted_labels ou l'expression image_labels == new_label est True              # on fait "new_label + 1" car les valeurs de cluster_laels_sorted sont comprises entre 0 et 5 alors que celles de image_labels entre 1 et 6 si on ne prend pas en compte le background ! 
        # Remettre les étiquettes dans la forme de l'image originale
        image_labels_reshaped = sorted_labels.reshape(image.shape[:2])      # comme j'ai laisse les 0 inchanges dans sorted_labels, on a seulement modif les pixels de la plante
        labeled_images_list_sorted.append(image_labels_reshaped)
    
    
    # Créer la colorbar continue
    cluster_colors_clustering_without_white = cluster_colors_clustering[1:]
    custom_cmap = ListedColormap(cluster_colors_clustering_without_white) 
    norm = Normalize(vmin=0, vmax=len(centroids))
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])  # Vous pouvez passer un tableau de valeurs si nécessaire
    
    
    # on affiche les images des pixels clusterises tries selon ndvi
    for i, labeled_image in enumerate(labeled_images_list_sorted):
        plt.clf()
        plt.imshow(labeled_image.astype(np.uint8), cmap=ListedColormap(cluster_colors_clustering))
        plt.title(f'Pixels Plante tries par ndvi regroupes par clusters, Image {i+1}')
        plt.colorbar(sm, label='Cluster')
        filename = f"Images/Plante_{plante}/Pixels_Plante_tries_par_ndvi_regroupes_par_clusters/Image {i+1}.png"  # Nommage des fichiers avec un indice
        plt.savefig(filename)
        plt.show()

    centroids_tries = centroids[indices_clusters_tries]
    plt.clf()
    # On affiche les spectres des centroids tries selon nir de TOUTES les images combinees (clustering commun)
    for i in range(centroids.shape[0]): 
        plt.plot(centroids_tries[i], color=cluster_colors_clustering[i+1])            # i+1 car je saute la première couleur associée au background
    plt.ylim(0, 1.2)
    plt.xlabel('Bandes (red = 138, nir = 190)')
    plt.ylabel('Intensite')
    plt.title(f'Spectre des Centroids Plante tries par ndvi, {k} clusters ')
    plt.colorbar(sm, label='Cluster')
    plt.legend()
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.8)  # Ajuster l'espace pour que la légende ne se chevauche pas avec l'image
    plt.grid(True)
    filename = f'Images/Plante_{plante}/Spectres_Centroids_tries_ndvi_Plante/Spectres_Centroids_tries_ndvi_all.png'
    plt.savefig(filename)
    plt.show()   
    
    return labeled_images_list_sorted





# Same that trier_clusters_crit_order but with the NIR band values
def trier_clusters_nir(image, concatenated_images, c, plante, cluster_colors_clustering, centroids, labels, labeled_images_list, k):
    os.makedirs(f'Images/Plante_{plante}/Pixels_Plante_tries_par_nir_regroupes_par_clusters', exist_ok=True)
    os.makedirs(f'Images/Plante_{plante}/Spectres_Centroids_tries_nir_Plante', exist_ok=True)
    os.makedirs(f'Images/Plante_{plante}/Histogrammes', exist_ok=True)

    # on applique la fonction ndvi pour avoir l'ordre souhaité des centroids
    nir_values = []
    for centroid in centroids:
        nir_value = nir(centroid)
        nir_values.append(nir_value)
       
    # on assigne les labels aux centroids en fonction de nir
    nir_values_tries = np.sort(nir_values)
    indices_clusters_tries = np.argsort(nir_values)      # on trie par ordre croissant de nir    # on obtient les indices (entre 1 et 15) par ordre croissant des labels selon crit_order

    # Étiquetage des centroïdes en fonction de ndvi croissant
    labeled_images_list_sorted = []
    # Affecter les étiquettes aux pixels dans chaque image
    for x in range(c):    
        # Réorganiser les étiquettes selon l'ordre croissant des centroïdes
        sorted_labels = np.zeros_like(labeled_images_list[0])
        image_labels = labeled_images_list[x]
        for new_label in indices_clusters_tries:
            sorted_labels[image_labels == new_label + 1] = indices_clusters_tries.tolist().index(new_label) + 1      # 'sorted_labels[image_labels == new_label]' selectionne les emplacements dans sorted_labels ou l'expression image_labels == new_label est True              # on fait "new_label + 1" car les valeurs de cluster_laels_sorted sont comprises entre 0 et 5 alors que celles de image_labels entre 1 et 6 si on ne prend pas en compte le background ! 
        # Remettre les étiquettes dans la forme de l'image originale
        image_labels_reshaped = sorted_labels.reshape(image.shape[:2])      # comme j'ai laisse les 0 inchanges dans sorted_labels, on a seulement modif les pixels de la plante
        labeled_images_list_sorted.append(image_labels_reshaped)
    
    
    # Créer la colorbar continue
    cluster_colors_clustering_without_white = cluster_colors_clustering[1:]
    custom_cmap = ListedColormap(cluster_colors_clustering_without_white) 
    norm = Normalize(vmin=0, vmax=len(centroids))
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])  # Vous pouvez passer un tableau de valeurs si nécessaire
    # Calculate the maximum cluster frequency across all images
    max_cluster_freq = 0
    for labeled_image in labeled_images_list_sorted:
        sorted_labels = labeled_image.reshape((-1, 1))
        non_zero_labels = sorted_labels[sorted_labels != 0]
        cluster_histogram = np.histogram(non_zero_labels, bins=np.arange(1, k + 2))[0]
        max_cluster_freq = max(max_cluster_freq, np.max(cluster_histogram))
    
        
    # Iterate through images and plot histograms
    for i, labeled_image in enumerate(labeled_images_list_sorted):
        plt.clf()
        plt.imshow(labeled_image.astype(np.uint8), cmap=ListedColormap(cluster_colors_clustering))
        plt.title(f'Pixels Plante triés par nir regroupés par clusters, Image {i+1}')
        plt.colorbar(sm, label='Cluster')
        filename = f"Images/Plante_{plante}/Pixels_Plante_tries_par_nir_regroupes_par_clusters/Image {i+1}.png"
        plt.savefig(filename)
        plt.show()
    
        sorted_labels = labeled_image.reshape((-1, 1))
        non_zero_labels = sorted_labels[sorted_labels != 0]
        cluster_histogram = np.histogram(non_zero_labels, bins=np.arange(1, k + 2))[0]
    
        plt.figure(figsize=(8, 6))  # Set the figure size for the histogram
        plt.bar(range(1, k + 1), cluster_histogram, color=cluster_colors_clustering[1:])
        plt.xlabel('Cluster')
        plt.ylabel('Number of Pixels')
        plt.title(f'Cluster Histogram ({k} Clusters), Image {i+1}')
        plt.xticks(range(1, k + 1))
        plt.ylim(0, max_cluster_freq)  # Set the y-axis limit to the calculated max cluster frequency
        filename = f"Images/Plante_{plante}/Histogrammes/Image {i+1}.png"
        plt.savefig(filename)
        plt.show()
    
        centroids_tries = centroids[indices_clusters_tries]
        
        plt.clf()
        # Boucle pour afficher les spectres triés avec les couleurs correspondantes
    for i in range(len(centroids_tries)):
        plt.plot(centroids_tries[i], color=cluster_colors_clustering[i + 1])
    plt.colorbar(sm, label='Cluster')
    plt.ylim(0, 1.2)
    plt.xlabel('Bandes (red = 138, nir = 190)')
    plt.ylabel('Intensité')
    plt.title(f'Spectre des Centroids Plante triés par nir, {k} clusters')
    plt.legend()
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.8)
    plt.grid(True)
    filename = f'Images/Plante_{plante}/Spectres_Centroids_tries_nir_Plante/Spectres_Centroids_tries_nir_all.png'
    plt.savefig(filename)
    plt.show()
                
    
    return labeled_images_list_sorted
    
    






# Same that trier_clusters_crit_order but with PSRI
def trier_clusters_psri(image, concatenated_images, c, plante, cluster_colors_clustering, centroids, labels, labeled_images_list, k):
    os.makedirs(f'Images/Plante_{plante}/Pixels_Plante_tries_par_psri_regroupes_par_clusters', exist_ok=True)
    os.makedirs(f'Images/Plante_{plante}/Spectres_Centroids_tries_psri_Plante', exist_ok=True)
    
    # on applique la fonction ndvi pour avoir l'ordre souhaité selon psri des centroids
    psri_values = []
    for centroid in centroids:
        psri_value = PSRI(centroid)
        psri_values.append(psri_value)
       
    # on assigne les labels aux centroids en fonction de psri
    psri_values_tries = np.sort(psri_values)
    indices_clusters_tries = np.argsort(psri_values)      # on trie par ordre croissant de psri    # on obtient les indices (entre 1 et 15) par ordre croissant des labels selon crit_order

    # Étiquetage des centroïdes en fonction de psri croissant
    labeled_images_list_sorted = []
    # Affecter les étiquettes aux pixels dans chaque image
    for x in range(c):    
        # Réorganiser les étiquettes selon l'ordre croissant de psri des centroïdes
        sorted_labels = np.zeros_like(labeled_images_list[0])
        image_labels = labeled_images_list[x]
        for new_label in indices_clusters_tries:
            sorted_labels[image_labels == new_label + 1] = indices_clusters_tries.tolist().index(new_label) + 1      # 'sorted_labels[image_labels == new_label]' selectionne les emplacements dans sorted_labels ou l'expression image_labels == new_label est True              # on fait "new_label + 1" car les valeurs de cluster_laels_sorted sont comprises entre 0 et 5 alors que celles de image_labels entre 1 et 6 si on ne prend pas en compte le background ! 
        # Remettre les étiquettes dans la forme de l'image originale
        image_labels_reshaped = sorted_labels.reshape(image.shape[:2])      # comme j'ai laisse les 0 inchanges dans sorted_labels, on a seulement modif les pixels de la plante
        labeled_images_list_sorted.append(image_labels_reshaped)
    
    # Créer la colorbar continue
    cluster_colors_clustering_without_white = cluster_colors_clustering[1:]
    custom_cmap = ListedColormap(cluster_colors_clustering_without_white) 
    norm = Normalize(vmin=0, vmax=len(centroids))
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])  # Vous pouvez passer un tableau de valeurs si nécessaire
    
    # on affiche les images des pixels clusterises tries selon psri
    for i, labeled_image in enumerate(labeled_images_list_sorted):
        plt.clf()
        plt.imshow(labeled_image.astype(np.uint8), cmap=ListedColormap(cluster_colors_clustering))
        plt.title(f'Pixels Plante tries par psri regroupes par clusters, Image {i+1}')
        plt.colorbar(sm, label='Cluster')
        filename = f"Images/Plante_{plante}/Pixels_Plante_tries_par_psri_regroupes_par_clusters/Image {i+1}.png"  # Nommage des fichiers avec un indice
        plt.savefig(filename)
        plt.show()

    centroids_tries = centroids[indices_clusters_tries]
    plt.clf()
    # On affiche les spectres des centroids tries selon psri de TOUTES les images combinees (clustering commun)
    for i in range(centroids.shape[0]): 
        plt.plot(centroids_tries[i], color=cluster_colors_clustering[i+1])            # i+1 car je saute la première couleur associée au background
    plt.ylim(0, 1.2)
    plt.xlabel('Bandes (red = 138, nir = 190)')
    plt.ylabel('Intensite')
    plt.title(f'Spectre des Centroids Plante tries par psri, {k} clusters ')
    plt.colorbar(sm, label='Cluster')
    plt.legend()
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.8)  # Ajuster l'espace pour que la légende ne se chevauche pas avec l'image
    plt.grid(True)
    filename = f'Images/Plante_{plante}/Spectres_Centroids_tries_psri_Plante/Spectres_Centroids_tries_psri_all.png'
    plt.savefig(filename)
    plt.show()   

    labeled_images_list_sorted




def trier_clusters_sumcentroid(image, concatenated_images, c, plante, cluster_colors_clustering, centroids, labels, labeled_images_list, k):
    os.makedirs(f'Images/Plante_{plante}/Pixels_Plante_tries_par_sumcentroid_regroupes_par_clusters', exist_ok=True)
    os.makedirs(f'Images/Plante_{plante}/Spectres_Centroids_tries_sumcentroid_Plante', exist_ok=True)
    
    # Étiquetage des centroïdes en fonction de l'ordre croissant
    indices_clusters_tries = np.argsort(centroids.sum(axis=1))
    labeled_images_list_sorted = []
    # Affecter les étiquettes aux pixels dans chaque image
    for x in range(c):    
        # Réorganiser les étiquettes selon l'ordre croissant des centroïdes
        sorted_labels = np.zeros_like(labeled_images_list[0])
        image_labels = labeled_images_list[x]
        for new_label in indices_clusters_tries:
            sorted_labels[image_labels == new_label + 1] = indices_clusters_tries.tolist().index(new_label) + 1      # 'sorted_labels[image_labels == new_label]' selectionne les emplacements dans sorted_labels ou l'expression image_labels == new_label est True              # on fait "new_label + 1" car les valeurs de cluster_laels_sorted sont comprises entre 0 et 5 alors que celles de image_labels entre 1 et 6 si on ne prend pas en compte le background ! 
        # Remettre les étiquettes dans la forme de l'image originale
        image_labels_reshaped = sorted_labels.reshape(image.shape[:2])      # comme j'ai laisse les 0 inchanges dans sorted_labels, on a seulement modif les pixels de la plante
        labeled_images_list_sorted.append(image_labels_reshaped)
    
    
    # Créer la colorbar continue
    cluster_colors_clustering_without_white = cluster_colors_clustering[1:]
    custom_cmap = ListedColormap(cluster_colors_clustering_without_white) 
    norm = Normalize(vmin=0, vmax=len(centroids))
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])  # Vous pouvez passer un tableau de valeurs si nécessaire
    
    # on affiche les images des pixels clusterises tries selon sumcentroid
    for i, labeled_image in enumerate(labeled_images_list_sorted):
        plt.clf()
        plt.imshow(labeled_image.astype(np.uint8), cmap=ListedColormap(cluster_colors_clustering))
        plt.title(f'Pixels Plante tries par sumcentroid regroupes par clusters, Image {i+1}')
        plt.colorbar(sm, label='Cluster')
        filename = f"Images/Plante_{plante}/Pixels_Plante_tries_par_sumcentroid_regroupes_par_clusters/Image {i+1}.png"  # Nommage des fichiers avec un indice
        plt.savefig(filename)
        plt.show()

    centroids_tries = centroids[indices_clusters_tries]

    plt.clf()
    # On affiche les spectres des centroids de TOUTES les images combinees (clustering commun)
    for i in range(centroids.shape[0]): 
        plt.plot(centroids_tries[i], color=cluster_colors_clustering[i+1])            # i+1 car je saute la première couleur associée au background
    plt.ylim(0, 1.2)
    plt.xlabel('Bandes (red = 138, nir = 190)')
    plt.ylabel('Intensite')
    plt.title(f'Spectre des Centroids Plante tries par sumcentroid, {k} clusters ')
    plt.colorbar(sm, label='Cluster')
    plt.legend()
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.8)  # Ajuster l'espace pour que la légende ne se chevauche pas avec l'image
    plt.grid(True)
    filename = f'Images/Plante_{plante}/Spectres_Centroids_tries_sumcentroid_Plante/Spectres_Centroids_tries_sumcntroid_all.png'
    plt.savefig(filename)
    plt.show()        
        
    labeled_images_list_sorted
        
        
        
        
        


def nir(spectrum):
    nir_values = spectrum[190]
    return nir_values

def ndvi(spectrum):
    red_values = spectrum[138]
    nir_values = spectrum[190]
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

def similarite_spectrale(spectre1, spectre2):
    # Calculer la distance euclidienne entre deux réflectances spectrales
    return np.linalg.norm(spectre1 - spectre2)










def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def generate_color_gradient(n):
    color1 = "#D4CC47"
    color2 = "#7C4D8B"
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(color1))/255
    c2_rgb = np.array(hex_to_RGB(color2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]













def smooth_image(image):
        image[image > 1] = 1  # Remplacer les valeurs > 1 par 1
        image[image < 0] = 0  # Remplacer les valeurs négatives par 0
        return image






def data_augmentation(images):
    augmented_images = []
    for image in images:
        augmented_images.append(image)
        """
        augmented_images.append(random_noise(image))
        """
        augmented_images.append(random_cut(image))
        augmented_images.append(rotate_image(image, 90))
        augmented_images.append(rotate_image(image, 180))
        augmented_images.append(rotate_image(image, 270))  
        augmented_images.append(flip_image(image))  # Flip horizontally
        augmented_images.append(mirror_image(image))  # Mirror vertically
    return augmented_images










def calculate_vegetation_indices(image, IV, c):
    liste_IV = []
    
    for x in IV:
        
        if x == 'NDVI': 
            epsilon = 0.0000000001
            red_band = image[:, :, 138] 
            nir_band = image[:, :, 200]
            ndvi = (nir_band - red_band) / (nir_band + red_band + epsilon)
            liste_IV.append(ndvi)
                
        if x == 'SR4':
            epsilon = 0.000000000000000001
            r700 = image[:, :, 153]
            r670 = image[:, :, 138]  
            sr4 = r700 / (r670 + epsilon)

            liste_IV.append(sr4)
    
        if x == 'mRENDVI':      
            epsilon = 0.00000000001
            r705 = image[:, :, 155]   
            r750 = image[:, :, 177]  
            r445 = image[:,:, 31]
            mRENDVI = (r750 - r705) / (r750 + r705 - 2*r445 + epsilon)
            liste_IV.append(mRENDVI)
    
        if x == 'PSRI':
            epsilon = 0.00000000001
            r500 = image[:, :, 55]
            r680 = image[:, :, 143]  
            r750 = image[:,:, 178]
            PSRI = (r750 - r500) / (r750 + epsilon)
            liste_IV.append(PSRI)
        
        if x == 'Crit_order':   
            epsilon = 0.0000000001
            r500 = image[:, :, 55]
            r680 = image[:, :, 143]  
            r750 = image[:,:, 178]
            PSRI = (r680 - r500) / (r750 + epsilon)
            r705 = image[:, :, 155]
            r445 = image[:,:, 31]
            mRENDVI = (r750 - r705) / (r750 + r705 - 2*r445 + epsilon)
            Crit_order = mRENDVI - PSRI
            liste_IV.append(Crit_order)
            
        if x == 'CAR1':   
            epsilon = 0.0000000001
            r510 = image[:, :, 60]
            r550 = image[:, :, 80]  
            liste_IV.append(1/(r510+epsilon) - 1/(r550+epsilon))
            
        if x == 'CAR2':   
            epsilon = 0.0000000001
            r510 = image[:, :, 60]
            r700 = image[:, :, 153]  
            liste_IV.append(1/(r510+epsilon) - 1/(r700+epsilon))
            
        if x == 'SR':   
            epsilon = 0.0000000001
            r800 = image[:, :, 200]   #NIR
            r670 = image[:, :, 138]   #Red
            liste_IV.append(r800/(r670+epsilon))
            
        if x == 'RGRI':   
            epsilon = 0.0000000001
            r500 = image[:, :, 55]
            S500599 = 0
            S600699 = 0
            for i in range(10):
                S500599 += image[:, :, 55+5*i]
                S600699 += image[:, :, 105+5*i]
            liste_IV.append(S600699/(S500599+epsilon))
            
        if x == 'RENDVI':   
            epsilon = 0.0000000001
            r750 = image[:, :, 178]
            r705 = image[:, :, 155]   
            liste_IV.append((r750-r705)/(r750+r705+epsilon))
            
        if x == 'SG':   
            epsilon = 0.0000000001
            S500599 = 0
            for i in range(10):
                S500599 += image[:, :, 55+5*i]
            liste_IV.append(S500599/10)
        

    concatenated_IV = np.stack(liste_IV, axis=2)

    return concatenated_IV






def rotate_image(image, angle):
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))
    return rotated_image

def flip_image(image):
    return cv2.flip(image, 1)

def mirror_image(image):
    return cv2.flip(image, 0)
    




def random_cut(image):
    height, width, channels = image.shape
    cut_width = np.random.randint(50, width)
    cut_height = np.random.randint(50, height)
    x = np.random.randint(0, width - cut_width)
    y = np.random.randint(0, height - cut_height)
    cropped_image = image[y:y+cut_height, x:x+cut_width, :]
    return cropped_image








