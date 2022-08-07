targetcls='24 99 245 344 471 555 661 701 802 919'
#SMAML LOSS: CE ; Proxy DATA: MS-COCO
for target in $targetcls; do
   CUDA_VISIBLE_DEVICES=1,0 python3 trainSMAML.py --save_dir 'noise_model' --lr 0.002 --iterations 4000 --confidence 10 --batch_size 32 --src coco --match_target $target --eps 10 --di --ti --loss_function abslogit
done


TARGET_NETS="googlenet vgg16 vgg19_bn resnet50 densenet121 resnet152 wide_resnet50_2"
for target_net in $TARGET_NETS; do
  CUDA_VISIBLE_DEVICES=0,1 python eval_10T.py  --target_model $target_net  --batch_size 30 --eps 16  --noise_path './noise_model/netG_meta_coco_'
done
