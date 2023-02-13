python3 -m main --info INFO --seed 100 \
            --model l2p_bert --area NLP --dataset seq-fewrel80 \
            --lr 0.005  --ptm vit --pw 1 --freeze_clf 0 --init_type unif \
            --eval_freq 1 --prob_l -1  --batch_size 4 --n_epochs 5 --alpha 0.5 --beta 1.0 --cuda 0 --buffer_size 0 --minibatch_size 0

