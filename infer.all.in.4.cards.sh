export CUDA_VISIBLE_DEVICES=0,1,2,3
hs=$(hostname)
echo $hs
nohup bash scripts/DMS.5fold.inference.sh $1 $2 P15056.itan.split.large.window,P07949.itan.split.large.window 0 > $hs.log.0.txt &
nohup bash scripts/DMS.5fold.inference.sh $1 $2 P04637.itan.split.large.window,Q09428.itan.split.large.window 1 > $hs.log.1.txt &
nohup bash scripts/DMS.5fold.inference.sh $1 $2 O00555.itan.split.large.window,Q14524.clean.itan.split.large.window 2 > $hs.log.2.txt &
nohup bash scripts/DMS.5fold.inference.sh $1 $2 Q99250.itan.split.large.window,Q14654.itan.split.large.window,P21802.itan.split.large.window 3 > $hs.log.3.txt &
