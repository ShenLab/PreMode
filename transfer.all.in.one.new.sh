export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nohup bash scripts/DMS.5fold.run.sh $1 IonChannel.chps.even.uniprotID,P15056,P21802 0 > euler.log.0.txt &
nohup bash scripts/DMS.5fold.run.sh $1 IPR001806.even.uniprotID,Q09428,O00555 1 > euler.log.1.txt &
nohup bash scripts/DMS.5fold.run.sh $1 IPR000719.even.uniprotID,P07949,P04637 2 > euler.log.2.txt &
nohup bash scripts/DMS.5fold.run.sh $1 Q14654,Q99250,Q14524,P15056.PF00130,P15056.PF07714 3 > euler.log.3.txt &
nohup bash scripts/DMS.5fold.run.sh $1 P15056.IPR001245.self,P15056.IPR001245.IPR001245,Q99250.IPR005821.IPR005821 4 > euler.log.4.txt &
nohup bash scripts/DMS.5fold.run.sh $1 P21802.IPR016248.self,P21802.IPR016248.IPR016248,Q14524.IPR005821.IPR005821 5 > euler.log.5.txt &
nohup bash scripts/DMS.5fold.run.sh $1 Q99250.IPR005821.self,Q14524.IPR005821.self,Q14524.IPR027359.IPR027359 6 > euler.log.6.txt &
nohup bash scripts/DMS.5fold.run.sh $1 Q14524.IPR027359.self,Q99250.heyne.IPR005821.self,Q99250.heyne.IPR005821.IPR005821 7 > euler.log.7.txt &
