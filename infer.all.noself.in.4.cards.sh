export CUDA_VISIBLE_DEVICES=0,1,2,3
hs=$(hostname)
echo $hs
nohup bash scripts/DMS.5fold.inference.sh scripts/PreMode/ analysis/PreMode/ P21802.IPR016248.IPR016248.noself.large.window,P07949.IPR020635.IPR020635.noself.large.window 0 > $hs.log.0.txt &
nohup bash scripts/DMS.5fold.inference.sh scripts/PreMode/ analysis/PreMode/ O00555.IPR005821.IPR005821.noself.large.window 1 > $hs.log.1.txt &
nohup bash scripts/DMS.5fold.inference.sh scripts/PreMode/ analysis/PreMode/ Q99250.IPR005821.IPR005821.noself.large.window 2 > $hs.log.2.txt &
nohup bash scripts/DMS.5fold.inference.sh scripts/PreMode/ analysis/PreMode/ Q14524.IPR005821.IPR005821.noself.large.window,Q14654.IPR013518.IPR013518.noself.large.window 3 > $hs.log.3.txt &
