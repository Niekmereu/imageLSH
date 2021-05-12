from PIL import Image
import numpy as np
import imagehash
from typing import Dict, List, Optional, Tuple
import sys
from os import listdir
from os.path import join
import itertools
import streamlit as st
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)

def hamming_similarity(x1: List, x2: List, hash_size: int) -> float:
    hamming_distance = np.sum(x1 != x2)
    similarity = (hash_size**2 - hamming_distance) * 1/hash_size**2
    return similarity

def load_image_original(file: str):
        return Image.open(file)

class ImageFinder:
    def __init__(self, hash_size: int, n_bands: int):
        self.hash_size = hash_size
        self.n_bands = n_bands
        self.rows = int(pow(hash_size, 2)/n_bands)
        self.signatures = {}
        self._hashed_buckets = []
        
    def _load_image(self, file: str):
        return Image.open(file).convert("L").resize(
                            (self.hash_size, self.hash_size), 
                            Image.ANTIALIAS)

        
    def _create_signature(self, file: str) -> List:
        PIL_img = self._load_image(file)
        hsh = imagehash.dhash(PIL_img, self.hash_size)
        signature = hsh.hash.flatten()

        PIL_img.close()
        return signature
    
    def _band_mapper(self, sgn: List, bnd: int) -> List:
        
        # create a signature for each band
        return sgn[bnd*self.rows:(bnd+1)*self.rows].tobytes()

    def _lsh(self, files: List):

        for file in files:
            
            # create a signature for each file
            # store for later use
            sgn = self._create_signature(file)
            self.signatures[file] = sgn
            for bnd in range(self.n_bands):
                
                # init a dict for each band
                # map each band to a bucket
                # use band signature as index
                self._hashed_buckets.append(dict())
                sgn_bnd = self._band_mapper(sgn, bnd)
                if sgn_bnd not in self._hashed_buckets[bnd]:
                    self._hashed_buckets[bnd][sgn_bnd] = list()
                self._hashed_buckets[bnd][sgn_bnd].append(file)
                
     
    
    def process_input_dir(self, directory: str):
        files = [join(directory, file) for file in listdir(directory)]
        self._lsh(files)

        
    def find_similar_img(self, file: str, threshold: float) -> List:
        # returns list of tuples, file followed by similarity.
        
        self.image_signature = self._create_signature(file)
        
        # first lsh to find candidate pairs (hashed to same bucket)
        # its a candidate if hashed atleast once to the same bucket
        candidates = list()
        for bnd in range(self.n_bands):
            sgn_bnd = self._band_mapper(self.image_signature, bnd)
            if len(self._hashed_buckets[bnd][sgn_bnd]) > 0:
                candidates.append(self._hashed_buckets[bnd][sgn_bnd])
        candidates = set(itertools.chain.from_iterable(candidates))
        
        # check if candidates are truly pairs
        similar_imgs = list()
        for candidate in candidates:
            
            similarity = hamming_similarity(self.image_signature, self.signatures[candidate], self.hash_size)
            
            if similarity > threshold:
                similar_imgs.append((candidate, round(similarity, 3)))
        
        return sorted(similar_imgs, key=lambda tup: tup[1], reverse=True)



def get_accuracy_and_plot(hash_size, image_folder, image_file, max_bands, threshold):
    accuracy_list = []
    for n in range(1, max_bands):
        model = ImageFinder(hash_size=hash_size, n_bands=n)
        model.process_input_dir(image_folder)
        similar_lsh = model.find_similar_img(image_file, threshold=threshold)
        similar_true = []
        for sgn in model.signatures:
            similarity = hamming_similarity(model.image_signature, model.signatures[sgn], hash_size)
            if similarity > threshold:
                similar_true.append((sgn, similarity))
        accuracy = len(similar_lsh)/len(similar_true)
        accuracy_list.append(accuracy)
    plt.plot(range(1, max_bands), accuracy_list)
    plt.xlabel("Number of Bands")
    plt.ylabel("Accuracy")
    st.pyplot()

def main():
    hash_size = 250
    st.title("Do you want to find similar pokemon?")
    st.subheader("This is an imagefinder using an LSH algorithm. Use a higher number of bands to increase the probability of finding similar images.")
    st.subheader("Selected Folder:")
    image_folder = st.text_input("Image folder", "images/")
    st.subheader("Selected image:")
    image_file = st.text_input("Image file", "images/castform.png")
    st.image(load_image_original(image_file))


    n_bands = st.slider("Number of bands", 3, 10)
    threshold = st.slider("Similarity threshold", 0.5, 1.0)
    st.subheader("For performance assessment it is possible to calculate the accuracy for a specific number of bands. This calculation takes some time.")
    max_bands = st.slider("Maximum number of bands for performance assessment", 5, 10)
    if st.button("calculate accuracy"):
        get_accuracy_and_plot(hash_size, image_folder, image_file, max_bands, threshold)


    model = ImageFinder(hash_size=hash_size, n_bands=n_bands)
    model.process_input_dir(image_folder)
    results = model.find_similar_img(file=image_file, threshold=threshold)
    # model.show_images(results)
    for image in results:
                st.image(load_image_original(image[0]))

if __name__ == "__main__":
    main()