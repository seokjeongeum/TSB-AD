git submodule update --init --recursive
mkdir -p Datasets
wget https://www.thedatum.org/datasets/TSB-AD-U.zip
unzip -o TSB-AD-U.zip -d Datasets
rm TSB-AD-U.zip 
wget https://www.thedatum.org/datasets/TSB-AD-M.zip
unzip -o TSB-AD-M.zip -d Datasets
rm TSB-AD-M.zip 
conda env create -f TSPulse2/environment.yml
