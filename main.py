import os.path
import time
import argparse
import numpy as np
import torch

from utils import *
from model import Model
from structure_refinement import StructureRefiner

def main(args):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 1. Load and preprocess data
    print(f"Loading dataset {args.dataset}...")
    raw_features, features, adj1, adj2, ano_label, raw_adj1, raw_adj2, config = load_dataset(args)

    # ============================================================
    # S²E-CARE STEP 0: SIMILARITY-AWARE STRUCTURE REFINEMENT
    # ============================================================
    if args.enable_refine:
        print(f"\n[S²E-CARE] Executing Step 0: Structure Refinement (Threshold={args.refine_threshold})...")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # use_text_model=False vì dataset Amazon là số
        refiner = StructureRefiner(threshold=args.refine_threshold, device=device, use_text_model=False)
        
        # --- Refine View 1 (Topology) ---
        print(" -> Refining View 1 (raw_adj1)...")
        raw_adj1 = refiner.refine(raw_adj1, raw_features)
        
        # --- Refine View 2 (Auxiliary) ---
        if raw_adj2 is not None:
            print(" -> Refining View 2 (raw_adj2)...")
            raw_adj2 = refiner.refine(raw_adj2, raw_features)
        
        print("[S²E-CARE] Structure Refinement Completed.\n")
    # ============================================================

    # 2. Initialize model and optimizer
    # Lưu ý: Cần thêm tham số beta và norm vào config hoặc args nếu model yêu cầu
    # Ở đây ta giả định config đã đủ hoặc gán mặc định
    if 'beta' not in config: config['beta'] = 0.1 
    if 'norm' not in config: config['norm'] = True
    
    optimiser_list = []
    model_list = []
    
    # Khởi tạo mô hình cho từng lần cắt (cutting)
    for i in range(args.cutting):
        model = Model(config['ft_size'], args.embedding_dim, 'prelu', args.readout, config)
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        if torch.cuda.is_available():
            model = model.cuda()
            
        optimiser_list.append(optimiser)
        model_list.append(model)

    if torch.cuda.is_available():
        print('Using CUDA for training')
        features = features.cuda()
        raw_features = raw_features.cuda()
        raw_adj1 = raw_adj1.cuda()
        if raw_adj2 is not None:
            raw_adj2 = raw_adj2.cuda()

    start = time.time()
    
    # 3. Prepare Distance Matrices (NSGT preparation)
    total_epoch = args.num_epoch * config['cutting']
    new_adj_list1 = []
    new_adj_list2 = []
    
    new_adj_list1.append(raw_adj1)
    all_cut_adj1 = torch.cat(new_adj_list1)

    # [Logic Cache]
    # Thêm suffix '_refined' nếu dùng refine để không lẫn file cache
    suffix = "_refined" if args.enable_refine else ""
    dist_file_path1 = f'./data/{args.dataset}_distance1{suffix}.npy'
    
    use_cache = os.path.exists(dist_file_path1)

    if use_cache:
        print(f"Loading cached distance matrix 1 from {dist_file_path1}...")
        dis_array1 = torch.FloatTensor(np.load(dist_file_path1))
    else:
        print("Calculating distance matrix 1 (Fresh)...")
        dis_array1 = calc_distance(raw_adj1[0, :, :], raw_features[0, :, :])
        np.save(dist_file_path1, dis_array1.cpu().numpy())

    if raw_adj2 is not None:
        new_adj_list2.append(raw_adj2)
        all_cut_adj2 = torch.cat(new_adj_list2)
        
        dist_file_path2 = f'./data/{args.dataset}_distance2{suffix}.npy'
        if os.path.exists(dist_file_path2):
             print(f"Loading cached distance matrix 2 from {dist_file_path2}...")
             dis_array2 = torch.FloatTensor(np.load(dist_file_path2))
        else:
            print("Calculating distance matrix 2 (Fresh)...")
            dis_array2 = calc_distance(raw_adj2[0, :, :], raw_features[0, :, :])
            np.save(dist_file_path2, dis_array2.cpu().numpy())
    
    # 4. Training Loop
    index = 0
    message_mean_list = []
    
    print(f"\nStarting Training Loop ({config['cutting']} cuts x {args.num_epoch} epochs)...")
    
    for n_cut in range(config['cutting']):
        message_list = []
        optimiser_list[index].zero_grad()
        model_list[index].train()
        
        if torch.cuda.is_available():
            dis_array1 = dis_array1.cuda()
            
        # Graph Cutting / NSGT View 1
        cut_adj1 = graph_nsgt(dis_array1, all_cut_adj1[0, :, :])
        dis_array1 = dis_array1.cpu()
        cut_adj1 = cut_adj1.unsqueeze(0)
        adj_norm1 = normalize_adj_tensor(cut_adj1, args.dataset)
        
        # Graph Cutting / NSGT View 2
        if raw_adj2 is not None:
            if torch.cuda.is_available():
                dis_array2 = dis_array2.cuda()
            cut_adj2 = graph_nsgt(dis_array2, all_cut_adj2[0, :, :])
            dis_array2 = dis_array2.cpu()
            cut_adj2 = cut_adj2.unsqueeze(0)
            adj_norm2 = normalize_adj_tensor(cut_adj2)
        else:
            adj_norm2 = None
            
        for epoch in range(args.num_epoch):
            # [S²E-CARE UPDATE] Forward trả về 4 giá trị: feat, cluster_sim, loss, alpha
            node_emb, cluster_sim, loss, alpha = model_list[index].forward(features[0], adj_norm1, raw_adj1, adj_norm2, raw_adj2)
            
            loss.backward()
            optimiser_list[index].step()
            loss = loss.detach().cpu().numpy()
            
            if (epoch + 1) % 100 == 0:
                print('Cut [{}/{}] | Epoch [{}/{}] | Loss: {:.4f}'.format(
                    n_cut + 1, config['cutting'], 
                    epoch + 1, args.num_epoch, 
                    loss))

        # Inference
        # [S²E-CARE UPDATE] Inference bây giờ dùng alpha để tính uncertainty
        # message_sum ở đây đại diện cho Anomaly Score thô
        # Ta dùng hàm inference mới trong model.py
        
        # Lấy kết quả cuối cùng của cut này
        with torch.no_grad():
            node_emb, cluster_sim, _, alpha = model_list[index].forward(features[0], adj_norm1, raw_adj1, adj_norm2, raw_adj2)
            
            # Tính score dựa trên S²E logic (Norm * Uncertainty)
            # Hàm view_consistency trả về tensor cùng device
            consistency_score = model_list[index].view_consistency(features[0], adj_norm1, adj_norm2)
            base_score = model_list[index].inference(node_emb, alpha)
            
            message_sum = base_score + consistency_score

        message_list.append(torch.unsqueeze(message_sum.cpu().detach(), 0))
        
        all_cut_adj1[0, :, :] = torch.squeeze(cut_adj1)
        if raw_adj2 is not None:
            all_cut_adj2[0, :, :] = torch.squeeze(cut_adj2)
            
        index += 1
        message_list = torch.mean(torch.cat(message_list), 0)
        message_mean_list.append(torch.unsqueeze(message_list, 0))
        
        # Evaluation Logic
        # Chỉ đánh giá ở cuối mỗi cut để xem tiến độ
        if n_cut % 5 == 0 or n_cut == config['cutting'] - 1:
            message_mean_cut = torch.mean(torch.cat(message_mean_list), 0)
            message_mean = np.array(message_mean_cut.cpu().detach())
            score = normalize_score(message_mean) # Score càng cao -> Càng bất thường
            
            print(f"--- Evaluation at Cut {n_cut+1} ---")
            model_list[index-1].evaluation(score, ano_label)
            
    end = time.time()
    print(f"\nTotal Training Time: {end - start:.2f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='S²E-CARE: Scalable Spectral-Evidential Graph Anomaly Detection')
    parser.add_argument('--dataset', type=str, default='Amazon', help='Amazon | BlogCatalog | imdb | dblp')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--num_epoch', type=int, default=500)
    parser.add_argument('--readout', type=str, default='avg') 
    parser.add_argument('--cutting', type=int, default=25) 
    parser.add_argument('--lamb', type=float, default=0.1) 
    parser.add_argument('--alpha', type=float, default=0.8) 
    parser.add_argument('--clusters', type=int, default=10) 
    
    # [NEW ARGUMENTS] Tên biến đã được chuẩn hóa
    parser.add_argument('--enable_refine', type=bool, default=True, help='Bật/Tắt Step 0: Structure Refinement')
    parser.add_argument('--refine_threshold', type=float, default=0.1, help='Ngưỡng lọc cạnh (0.1 cho features số)')
    
    args = parser.parse_args()
    print('Dataset: ', args.dataset)
    print(f'Structure Refinement: {args.enable_refine}')
    main(args)