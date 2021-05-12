python version 3.6

LSH implementation to find similar images.

to run:
 run ImageFinderLSH.py

1. Parameters
 n_bands: number of bands
 hash_size: nr of bytes (booleans) for the representation of one image.
 threshold: what should the similarity be before we classify an image as similar?  

2. Similar images with LSH.
Hash each item to a (binary) signature. Compare signatures using a similarity metric (Hamming) to find similar images. 
Takes a long time when the number of items is large.

This is solved with LSH. Cut each signature into r bands. Now hash each band to a bucket. When images are hashed to the same bucket for atleast one band, they are potential
duplicates.

Finally, only calculate the similarity on the whole signature for the almost-pairs found with LSH.

3. Integration in production
Signatures should be stored as binary. If we pick a signature size of 250, we can store 10^7 signatures with less than 2.5GB. These signatures could even be kept in working memory without issues.

If the number of images grows even larger, signatures could be kept out of working memory. The only thing strictly required in memory is the hash table.

Model could also be easily implemented in spark using map-reduce.

4. Further details
Let's discuss during interview.
