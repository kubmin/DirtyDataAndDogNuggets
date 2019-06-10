#!/bin/bash

echo "Which dataset do you want to download?"
select yn in "Caltech101" "Caltech256"; do
    case $yn in
        Caltech101 ) echo "[INFO] Downloading Caltech101 dataset"; wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz; echo "[INFO] Unzipping ..."; tar -xzf 101_ObjectCategories.tar.gz; rm 101_ObjectCategories.tar.gz; mv 101_ObjectCategories Caltech101; break;;
        Caltech256 ) echo "[INFO] Downloading Caltech256 dataset"; wget http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar; echo "[INFO] Unzipping ..."; tar -xf 256_ObjectCategories.tar; rm 256_ObjectCategories.tar; mv 256_ObjectCategories Caltech256; break;;
    esac
done
echo "Bye!"
