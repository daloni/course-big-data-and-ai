docker image pull mmcauliffe/montreal-forced-aligner:latest
docker run -it -v ./mfa:/data mmcauliffe/montreal-forced-aligner:latest

mfa model download dictionary spanish_spain_mfa
mfa model download acoustic spanish_mfa

mfa align /data spanish_spain_mfa spanish_mfa /data/output -j 4