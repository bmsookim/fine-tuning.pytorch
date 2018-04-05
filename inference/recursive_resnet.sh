for ((i=7;i<=9;i++)); do
    python inference.py \
        --net_type resnet \
        --depth 152 \
        --path /home/bumsoo/Data/test/inbreast_patches_test_1_$i
done
