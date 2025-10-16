#num terminal do seu pc:

ssh -L 8888:localhost:8888 joao.poca@login-server.lps.ufrj.br

#depois que alocar a caloba dentro do login-server:

ssh -L 8888:localhost:8888 caloba91

#dentro do singularity, na pasta que voce quer abrir:

singularity exec --nv --bind /mnt/shared/storage02/projects/sonar_data/data/iara:/home/iara --bind /mnt/cern_data/pedro.lisboa/sonar_data:/home/data -H $PWD:/home/code $HOME/hydra.sif bash

cd /hom/code/examples/
python3 -m notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root