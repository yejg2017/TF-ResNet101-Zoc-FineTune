python finetune_v2.py \
    --learning_rate "0.0001" \
    --num_classes "5" \
    --resnet_depth "101" \
    --batch_size "20" \
    --num_epochs "200" \
    --train_layers "fc,scale5" 
    #--training_file "/home/ye/user/yejg/LEARN/DL_MODEL_LEARN/Keras/Diabetic-Retinopathy-with-CNN/data/seg_train.txt" \
    #--val_file "/home/ye/user/yejg/LEARN/DL_MODEL_LEARN/Keras/Diabetic-Retinopathy-with-CNN/data/seg_test.txt"


