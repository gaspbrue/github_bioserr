import numpy as np
from utils_clustering_serietemporelle import background_elimination
from utils_clustering_serietemporelle import charger_imageHS
import pdb
import os
from utils_clustering_serietemporelle import clustering_flattened
from utils_clustering_serietemporelle import trier_clusters_crit_order
from utils_clustering_serietemporelle import trier_clusters_ndvi
from utils_clustering_serietemporelle import trier_clusters_nir
from utils_clustering_serietemporelle import trier_clusters_psri
from utils_clustering_serietemporelle import trier_clusters_sumcentroid
from utils_clustering_serietemporelle import generate_color_gradient
from utils_clustering_serietemporelle import smooth_image
from utils_clustering_serietemporelle import data_augmentation
from utils_clustering_serietemporelle import calculate_vegetation_indices
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import random
from collections import Counter
import pandas as pd
from sklearn.utils import resample




plante = 5 # choose the plant for which you want to work with
os.makedirs("Images", exist_ok=True)
os.makedirs(f"Images/Plante_{plante}", exist_ok=True)
folder_path = f'Data/Plante {plante}'
file_list = os.listdir(folder_path)
file_list.sort() 
path_to_hdr = {}  

#Hyperparameters
c=0
n_clusters_background = 12   # nb of clusters for the elimination of the background
n_clusters_clustering = 12    # nb of classe for the clustering of the plant
cluster_colors_background_elimination = ['#'+''.join([random.choice('0123456789ABCDEF') for i in range(6)]) for _ in range(n_clusters_background)]    # je definis les couleurs avant la boucle pour que ce soient les memes pour toutes les images
cluster_colors_clustering_random = ['#'+''.join([random.choice('0123456789ABCDEF') for i in range(6)]) for _ in range(n_clusters_clustering+1)]     # couleurs aleatoires
cluster_colors_clustering = generate_color_gradient(n_clusters_clustering + 1)
cluster_colors_clustering[0] = '#FFFFFF'  # Color code for white


masks = []
images = []
# Browse through each file in the list to load the hyperspectral images
for file_name in file_list:
    if file_name.endswith('.hdr'):
        c+=1
        path_to_hdr[f'path_to_hdr_{c}'] = os.path.join(folder_path, file_name)
        file_name_bil = file_name.replace('.bil.hdr', '')
        file_name_bil += '.bil'
        path_to_bil = os.path.join(folder_path, file_name_bil)
        image = charger_imageHS(path_to_bil, path_to_hdr[f'path_to_hdr_{c}'])
        image = smooth_image(image)
        images.append(image)
  
        
"""
# Data augmentation
images = data_augmentation(images)  
"""



# Elimination of the background
c=0   
new_images = []         
for image in images:
    c+=1
    image, mask = background_elimination(image, c, plante, cluster_colors_background_elimination, n_clusters_background)     # image = image without background, non_zero_pixels_reshaped = on garde que les pixels de l'objet
    masks.append(mask)
    image = image[:, :, :210]
    new_images.append(image)
    image_shape = image.shape
    flattened_image = image.reshape((-1, image.shape[-1]))     
    if c == 1:
        concatenated_images = flattened_image
    else:
        concatenated_images = np.concatenate((concatenated_images, flattened_image), axis=0)
  
# We then concatenate ALL the images of the plants (with background) to process the clustering
centroids, labels, labeled_images_list = clustering_flattened(concatenated_images, c, plante, cluster_colors_clustering, image_shape, n_clusters_clustering)    # j'utilise clustering_flattened car l'image n'est pas flattened et donc je rjoute une etape dans la fonction pour la flatten          # labels = classe associée à chq pixel (pour tous les pixels de l'image), en flattened 
labels = labels.astype(np.int32)






# Sort the clusters to make sure the centroids are in the good order of senescence
unsupervised_labels_nir = trier_clusters_nir(image, concatenated_images, c, plante, cluster_colors_clustering, centroids, labels, labeled_images_list, n_clusters_clustering)
"""
unsupervised_labels_ndvi = trier_clusters_ndvi(image, concatenated_images, c, plante, cluster_colors_clustering, centroids, labels, labeled_images_list, n_clusters_clustering)
unsupervised_labels_sumcentroid = trier_clusters_sumcentroid(image, concatenated_images, c, plante, cluster_colors_clustering, centroids, labels, labeled_images_list, n_clusters_clustering)
unsupervised_labels_psri = trier_clusters_psri(image, concatenated_images, c, plante, cluster_colors_clustering, centroids, labels, labeled_images_list, n_clusters_clustering)
unsupervised_labels_crit_order = trier_clusters_crit_order(image, concatenated_images, c, plante, cluster_colors_clustering, centroids, labels, labeled_images_list, n_clusters_clustering)
"""






# Feature extraction : vegetation indices
IV = ['NDVI', 'mRENDVI', 'SR4','PSRI','Crit_order', 'SG', 'RENDVI', 'RGRI', 'SR', 'CAR2', 'CAR1']
images_IV = []
c=0
for image in new_images:
    c += 1
    vegetation_image = calculate_vegetation_indices(image, IV, c)
    images_IV.append(vegetation_image)
images_IV = np.array(images_IV)



# Sample training/test data
images_IV = np.array(images_IV).reshape(-1, 11)
unsupervised_labels_nir = np.array(unsupervised_labels_nir).reshape(-1)
new_mask = unsupervised_labels_nir != 0
X_filtered = images_IV[new_mask]
y_filtered = unsupervised_labels_nir[new_mask]
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)


"""
# Undersampling
classes, counts = np.unique(y_train, return_counts=True)
counter = Counter(y_train)
minority_class_count = min(counter.values())
resampled_data = []
"""


X_train_flattened = X_train.reshape((X_train.shape[0], -1))
X_test_flattened = X_test.reshape((X_test.shape[0], -1))
X_train_flattened = X_train_flattened.astype(float)



# Random forest clasifier
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_classifier.fit(X_train_flattened, y_train)
y_pred = random_forest_classifier.predict(X_test_flattened)



# Performances
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Accuracy, modele RF : {accuracy}")
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(report)

# Accuracy with 1 class error tolerance
y_true = y_test
tolerance = 1
correct_predictions = 0
total_samples = len(y_true)
for true_label, predicted_label in zip(y_true, y_pred):
    if abs(true_label - predicted_label) <= tolerance:
        correct_predictions += 1
accuracy_with_tolerance = correct_predictions / total_samples
print(f"Accuracy with 1-class error tolerance: {accuracy_with_tolerance:.2f}")





"""
y1 = random_forest_classifier.predict(X_test_flattened)
ylast = random_forest_classifier.predict(X_test_flattened)
"""




















"""
# Echantillonnage des donnees d'entrainement, on selectionne les pixels les plus homogenes pour gagner en rapidite dans le training
ntrain = 20*n_clusters_clustering
n_superpixels = 20*ntrain    # nb de superpixels pour l'ensemble d'entraînement
# on calcule les spectre moyens pour les superpixels selectionnes (où la variance est min), régions spatiales présentant des caractéristiques spectrales homogènes
region_size = (30,30) 
ntest = 30
training_datas, training_labels = [],[]
test_datas, test_labels = [],[]
for x in range(c):
    unsupervised_label_nir = unsupervised_labels_nir[x]
    tableau_labels_nir = unsupervised_label_nir.reshape(image.shape[0], image.shape[1])
    mask = masks[x]
    training_data, training_label = sample_training_data(image, tableau_labels_nir, mask, n_superpixels, ntrain, plante, x+1)
    test_data, test_label = sample_test_data(image, mask, tableau_labels_nir, ntest, region_size, plante, x+1)   # spectres moyens de régions sélectionnées au hasard pour le test
    
    training_datas.append(training_data)
    training_labels.append(training_label)
    test_datas.append(test_data)
    test_labels.append(test_label)
    
    # Concaténation des données d'entraînement et des étiquettes d'entraînement
    if x == 0:
        concatenated_training_data = training_data
        concatenated_training_label = training_label
        concatenated_test_data = test_data
        concatenated_test_label = test_label
    else:
        concatenated_training_data = np.concatenate((concatenated_training_data, training_data), axis=0)
        concatenated_training_label = np.concatenate((concatenated_training_label, training_label), axis=0)
        concatenated_test_data = np.concatenate((concatenated_test_data, test_data), axis=0)
        concatenated_test_label = np.concatenate((concatenated_test_label, test_label), axis=0)
"""



pdb.set_trace()




