
mkdir -p data
cd data

echo 'Downloading Penn Treebank dataset'

mkdir -p ptb
cd ptb
wget --quiet --continue https://github.com/wojzaremba/lstm/blob/master/data/ptb.train.txt
wget --quiet --continue https://github.com/wojzaremba/lstm/blob/master/data/ptb.valid.txt
wget --quiet --continue https://github.com/wojzaremba/lstm/blob/master/data/ptb.test.txt

echo 'Making .npz file and vocabulary for Penn Treebank dataset'

cd ../..
python ../utils.py -i ptb -o ptb -p ptb > ptb/ptb.train.vocab


