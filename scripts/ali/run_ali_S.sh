export CUDA_VISIBLE_DEVICES=0


for model in  TFMixformer
do


for preLen in 96 192 336 640 
do


# alibaba
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/alibaba/ \
 --data_path RpcCons.csv \
 --task_id RpcCons \
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

 # alibaba
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/alibaba/ \
 --data_path HttpReq.csv \
 --task_id HttpReq \
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

# alibaba
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/alibaba/ \
 --data_path RpcProv.csv \
 --task_id RpcProv \
 --model $model \
 --data custom \
 --features S \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 5 \
 --enc_in 7 \
 --dec_in 7 \
 --c_out 7 \
 --des 'Exp' \
 --itr 2


done
done

