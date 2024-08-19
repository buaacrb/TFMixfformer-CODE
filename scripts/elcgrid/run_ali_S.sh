export CUDA_VISIBLE_DEVICES=0


for model in  TFMixformer
do


for preLen in 96 192 336 640 
do

 # elcgrid
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/elcgrid/ \
 --data_path HttpRt.csv \
 --task_id HttpRt \
 --model $model \
 --data custom \
 --features S \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 5 \
 --enc_in 8 \
 --dec_in 8 \
 --c_out 8 \
 --des 'Exp' \
 --itr 2

  # elcgrid
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/elcgrid/ \
 --data_path SvcReq.csv \
 --task_id SvcReq \
 --model $model \
 --data custom \
 --features S \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 5 \
 --enc_in 8 \
 --dec_in 8 \
 --c_out 8 \
 --des 'Exp' \
 --itr 2
 

done
done

