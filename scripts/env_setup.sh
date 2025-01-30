#!/bin/bash

pip install -r requirements.txt

# download officail_eval
git clone https://github.com/hotpotqa/hotpot evaluation/official_evaluation/hotpotqa
cd evaluation/official_evaluation/hotpotqa ; git checkout 3635853403a8735609ee997664e1528f4480762a
rm -rf .git
cd ../../..

git clone https://github.com/Alab-NII/2wikimultihop evaluation/official_evaluation/2wikimultihopqa
cd evaluation/official_evaluation/2wikimultihopqa ; git checkout 6bdd033bd51aae2d36ba939688c651b5c54ec28a
rm -rf .git
cd ../../..

git clone https://github.com/stonybrooknlp/musique evaluation/official_evaluation/musique
cd evaluation/official_evaluation/musique ; git checkout 24cc5b297acc2abfc5fb3d0becb6ef7b73d03717
rm -rf .git
cd ../../..