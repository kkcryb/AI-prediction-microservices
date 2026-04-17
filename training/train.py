import os
from tqdm import tqdm
import torch
import numpy as np
import utils
import pandas as pd
import time

def training(args, net, optim, loss_func, train_loader, valid_loader, fold):
        # ================= [新增: 打印模型参数量] =================
        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"\n{'=' * 20} Model Performance {'=' * 20}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"{'=' * 60}\n")
        # =========================================================
        valid_loss = 1000
        net.train()
        for _ in tqdm(range(args.epoch), desc='Training'):
            for j, data in enumerate(train_loader):
                '''
                occupancy = (batch, seq, node)
                add_tensor = (batch, seq, node)
                label = (batch, node)
                '''
                net.train()
                #extra_feat = 'None'
                extra_feat = None
                if args.add_feat != 'None':
                    occupancy, label, extra_feat = data
                else:
                    occupancy, label = data

                optim.zero_grad()
                predict = net(occupancy, extra_feat)
                if predict.shape != label.shape:
                    loss = loss_func(predict.unsqueeze(-1), label)
                else:
                    loss = loss_func(predict, label)
                loss.backward()
                optim.step()

            # validation
            net.eval()
            for j, data in enumerate(valid_loader):
                '''
                occupancy = (batch, seq, node)
             = (batch, seq, node)
                label = (batch, node)
                '''
                net.train()
                #extra_feat = 'None'
                extra_feat = None
                if args.add_feat != 'None':
                    occupancy, label, extra_feat = data
                else:
                    occupancy, label = data

                predict = net(occupancy,extra_feat)
                if predict.shape != label.shape:
                    loss = loss_func(predict.unsqueeze(-1), label)
                else:
                    loss = loss_func(predict, label)
                if loss.item() < valid_loss:
                    valid_loss = loss.item()
                    output_dir = '../checkpoints/'
                    os.makedirs(output_dir, exist_ok=True)
                    path = (output_dir + args.model + '_' +
                            'feat-' + args.feat + '_' +
                            'pred_len-' + str(args.pred_len) + '_' +
                            'fold-' + str(args.fold) + '_' +
                            'node-' + str(args.pred_type) + '_' +
                            'add_feat-' + str(args.add_feat) + '_' +
                            'epoch-' + str(args.epoch) + '.pth')
                    torch.save(net.state_dict(), path)
        # ================= [新增: 最简单的一行显存记录] =================
        if torch.cuda.is_available():
            print(f"\nPeak GPU Memory (Training): {torch.cuda.max_memory_allocated() / 1024**2:.0f} MiB\n")
        # ===============================================================

def test(args, test_loader, occ,net,scaler='None'):
    # ----init---
    result_list = []
    predict_list = np.zeros([1, occ.shape[1]])
    label_list = np.zeros([1, occ.shape[1]])
    if args.pred_type != 'region':
        predict_list = np.zeros([1,1])
        label_list = np.zeros([1,1])
    # ----init---

    total_inference_time = 0

    if not args.stat_model:
        output_dir = '../checkpoints/'
        os.makedirs(output_dir,exist_ok=True)
        path = (output_dir + args.model + '_' +
                'feat-' + args.feat + '_' +
                'pred_len-' + str(args.pred_len) + '_' +
                'fold-' + str(args.fold) + '_' +
                'node-' + str(args.pred_type) + '_' +
                'add_feat-' + str(args.add_feat) + '_' +
                'epoch-' + str(args.epoch) + '.pth')
        state_dict = torch.load(path,weights_only=True)
        net.load_state_dict(state_dict)
        net.eval()
        for j, data in enumerate(test_loader):
            #extra_feat = 'None'
            extra_feat = None
            if args.add_feat != 'None':
                occupancy, label, extra_feat = data
            else:
                occupancy, label = data
            with torch.no_grad():
                # [新增] 推理计时开始
                if torch.cuda.is_available(): torch.cuda.synchronize()
                start_time = time.time()

                predict = net(occupancy, extra_feat)

                # [新增] 推理计时结束并累加
                if torch.cuda.is_available(): torch.cuda.synchronize()
                end_time = time.time()
                total_inference_time += (end_time - start_time)


                if predict.shape != label.shape:
                    predict = predict.unsqueeze(-1)
                predict = predict.cpu().detach().numpy()
            label = label.cpu().detach().numpy()

            # ================= [核心修复: 在 for 循环内部拼接数据] =================
            # 这样就能把测试集里每一个 Batch 的数据都收集起来，不再只有最后一个 Batch
            predict_list = np.concatenate((predict_list, predict), axis=0)
            label_list = np.concatenate((label_list, label), axis=0)
            # =====================================================================

        # [新增] 打印整个测试集的总推理时间
        print(f"\n{'=' * 20} Inference Performance {'=' * 20}")
        # 这里乘以1000将秒转换为毫秒，单位改为 ms
        print(f"Total Inference Time (Entire Test Set): {total_inference_time * 1000:.4f} ms")
        # 如果 Table 6 依然需要填入秒数，可以保留原本的秒数提示，或者同时打印两者
        print(f"  (Use {total_inference_time:.4f} for 'Inference (s)' in Table 6)")
        print(f"{'=' * 60}\n")
    else:
        train_valid_occ,test_occ = test_loader
        predict = net.predict(train_valid_occ,test_occ)
        label = test_occ

    #predict_list = np.concatenate((predict_list, predict), axis=0)
    #label_list = np.concatenate((label_list, label), axis=0)
    if scaler != 'None':
        predict_list = scaler.inverse_transform(predict_list)
        label_list = scaler.inverse_transform(label_list)


    output_no_noise = utils.metrics(test_pre=predict_list[1:], test_real=label_list[1:], args=args)
    result_list.append(output_no_noise)

    # Adding model name, pre_l and metrics and so on to DataFrame
    result_df = pd.DataFrame(result_list, columns=['MSE', 'RMSE', 'MAPE', 'RAE', 'MAE'])
    result_df['model_name'] = args.model
    result_df['pred_len'] = args.pred_len
    result_df['fold'] = args.fold 

    # Save the results in a CSV file
    output_dir = '../result' + '/' + 'main_exp' + '/' + 'region'
    os.makedirs(output_dir, exist_ok=True)
    csv_file = output_dir + '/' + f'results.csv'

    # Append the result if the file exists, otherwise create a new file
    if os.path.exists(csv_file):
        result_df.to_csv(csv_file, mode='a', header=False, index=False, encoding='gbk')
    else:
        result_df.to_csv(csv_file, index=False, encoding='gbk')


