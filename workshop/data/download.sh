#!/bin/bash

echo "Which features do you want to download?"
select yn in "Caltech101" "Caltech256"; do
    case $yn in
        Caltech101 ) echo "[INFO] Downloading extracted Caltech101 features"; wget http://92.111.10.191/features/callecht101/features_caltech256.p; echo "[INFO] Unzipping ..."; break;;
        Caltech256 ) echo "[INFO] Downloading extracted Caltech256 features"; wget http://92.111.10.191/features/callecht256/features_caltech256.p; echo "[INFO] Unzipping ..."; break;;
    esac
done
echo "Bye!"
