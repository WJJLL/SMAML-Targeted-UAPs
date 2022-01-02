targetcls='24 99 245 344 471 555 661 701 802 919'
#SMAML LOSS: CE ; Proxy DATA: MS-COCO
for target in $targetcls; do
   CUDA_VISIBLE_DEVICES=1,0 python3 trainSMAML.py --save_dir 'noise_model' --lr 0.005 --iterations 4000 --confidence 10 --batch_size 32 --src coco --match_target $target --eps 10   --loss_function ce --model_type M3CE
done


TARGET_NETS="googlenet vgg16 vgg19_bn resnet50 densenet121 resnet152 wide_resnet50_2"
Main_types='ensSMaml'
for main_type in $Main_types; do
	  for target_net in $TARGET_NETS; do
		      CUDA_VISIBLE_DEVICES=0,1 python eval_10T.py  --source_model M3CE --source_domain coco  --target_model $target_net --main_type $main_type  --batch_size 20 --eps 10 --iterations 4000
    done
done

#####################################################################################################################

#MAML LOSS: CE ; Proxy DATA: MS-COCO
for target in $targetcls; do
   CUDA_VISIBLE_DEVICES=1,0 python3 trainMaml.py --save_dir 'noise_model' --lr 0.005 --iterations 4000 --confidence 10 --batch_size 32 --src coco --match_target $target --eps 10   --loss_function ce --model_type resnet50
done

#MIM LOSS: CE ; Proxy DATA: MS-COCO
for target in $targetcls; do
   CUDA_VISIBLE_DEVICES=1,0 python3 trainMIM.py --save_dir 'noise_model' --lr 0.005 --iterations 4000 --confidence 10 --batch_size 32 --src coco --match_target $target --eps 10   --loss_function ce --model_type resnet50
done

#GHOST LOSS: CE ; Proxy DATA: MS-COCO
for target in $targetcls; do
   CUDA_VISIBLE_DEVICES=1,0 python3 trainGhost.py --save_dir 'ghost' --lr 0.005 --iterations 4000 --confidence 10 --batch_size 32 --src coco --match_target $target --eps 10   --loss_function ce --model_type resnet50
done

#GHOST-SMAML LOSS: CE ; Proxy DATA: MS-COCO
for target in $targetcls; do
   CUDA_VISIBLE_DEVICES=1,0 python3 trainGhostSmaml.py --save_dir 'ghost' --lr 0.005 --iterations 4000 --confidence 10 --batch_size 32 --src coco --match_target $target --eps 10   --loss_function ce --model_type resnet50
done


