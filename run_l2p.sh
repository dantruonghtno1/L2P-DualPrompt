echo '-------------------------------------------buffer size 5000 ---------------------------------------'
python3 -m main --info INFO --seed 100 \
            --model l2p_vit --area CV --dataset seq-cifar100 \
            --lr 0.005  --ptm vit --pw 1 --freeze_clf 0 --init_type unif \
            --eval_freq 1 --prob_l -1  --batch_size 16 --n_epochs 5 --buffer_size 5000 --minibatch_size 16 --alpha 0.5 --beta 1.0 --cuda 0



echo '-------------------------------------------buffer size 1000 ---------------------------------------'
python3 -m main --info INFO --seed 100 \
            --model l2p_vit --area CV --dataset seq-cifar100 \
            --lr 0.005  --ptm vit --pw 1 --freeze_clf 0 --init_type unif \
            --eval_freq 1 --prob_l -1  --batch_size 16 --n_epochs 5 --buffer_size 1000 --minibatch_size 16 --alpha 0.5 --beta 1.0 --cuda 0


echo '-------------------------------------------buffer	size 0 ---------------------------------------'
python3 -m main --info INFO --seed 100 \
            --model l2p_vit --area CV --dataset seq-cifar100 \
            --lr 0.005  --ptm vit --pw 1 --freeze_clf 0 --init_type unif \
            --eval_freq 1 --prob_l -1  --batch_size 16 --n_epochs 5 --buffer_size 0 --minibatch_size 16 --alpha 0.5 --beta 1.0 --cuda 0




