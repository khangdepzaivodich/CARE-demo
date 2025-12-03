import torch
import numpy as np
from tqdm import tqdm

# Import có điều kiện: Nếu không có thư viện này vẫn chạy được với dữ liệu số
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class StructureRefiner:
    def __init__(self, threshold=0.6, device='cuda', batch_size=2048, use_text_model=False):
        """
        StructureRefiner: Module lọc nhiễu cấu trúc đồ thị (S²E-CARE Step 0).
        Hoạt động dựa trên độ tương đồng đặc trưng (Feature Similarity).
        
        Args:
            threshold (float): Ngưỡng tương đồng (0.0 -> 1.0). Cạnh < threshold sẽ bị cắt.
            device (str): 'cuda' hoặc 'cpu'.
            batch_size (int): Kích thước lô xử lý để tránh OOM trên GPU.
            use_text_model (bool): True nếu input là Text (cần SentenceBERT). False nếu input là Số.
        """
        self.device = device
        self.threshold = threshold
        self.batch_size = batch_size
        self.use_text_model = use_text_model
        self.text_model = None

    def _prepare_features(self, features):
        """
        Chuẩn hóa đầu vào:
        - Nếu là Tensor số -> Đẩy lên GPU.
        - Nếu là Text -> Gọi Model để Encode thành Vector.
        """
        # Case 1: Dữ liệu số (Tensor) - Dành cho Amazon Dataset
        if torch.is_tensor(features):
            return features.to(self.device)
            
        # Case 2: Dữ liệu Text (List[str]) - Dành cho YelpChi (nếu có raw text)
        elif isinstance(features, list) and isinstance(features[0], str):
            # Tự động bật chế độ text nếu phát hiện input là string
            if not self.use_text_model:
                self.use_text_model = True
                
            if self.text_model is None:
                if SentenceTransformer is None:
                    raise ImportError("Lỗi: Bạn cần cài đặt thư viện để xử lý text: pip install sentence-transformers")
                
                print("[Refiner] Loading Text Embedding Model (all-MiniLM-L6-v2)...")
                self.text_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
                
            print("[Refiner] Encoding text attributes...")
            return self.text_model.encode(features, batch_size=64, show_progress_bar=True, convert_to_tensor=True)
            
        else:
            raise ValueError(f"Unsupported feature format: {type(features)}")

    def refine(self, adj_tensor, features):
        """
        Hàm chính thực hiện lọc cấu trúc.
        """
        print(f"\n[S²E-CARE Step 0] Running Structure Refinement (Threshold={self.threshold})...")
        
        # 1. Xử lý Shape (Chuyển 3D [1, N, N] thành 2D [N, N])
        if adj_tensor.dim() == 3:
            adj_matrix = adj_tensor.squeeze(0)
        else:
            adj_matrix = adj_tensor

        if torch.is_tensor(features) and features.dim() == 3:
            node_feats = features.squeeze(0)
        else:
            node_feats = features

        # 2. Chuẩn bị Features & Normalize
        # Lấy features (số hoặc text emb)
        embeddings = self._prepare_features(node_feats)
        
        # Normalize (L2 Norm) để Dot Product trở thành Cosine Similarity
        # Đây là bước quan trọng để so sánh công bằng
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        num_nodes = adj_matrix.shape[0]

        # 3. Lấy danh sách cạnh hiện có
        # Sử dụng .nonzero() để lấy tọa độ các cạnh (Sparse format)
        rows, cols = adj_matrix.nonzero(as_tuple=True)
        rows = rows.to(self.device)
        cols = cols.to(self.device)
        total_edges = len(rows)
        
        print(f" -> Input Graph: {num_nodes} nodes, {total_edges} edges.")
        print(f" -> Computing Similarity & Filtering (Batch size: {self.batch_size})...")

        # 4. Tính toán và Lọc theo Batch (Batch Processing)
        keep_mask = []
        with torch.no_grad(): # Tắt Gradient để tiết kiệm VRAM
            for i in tqdm(range(0, total_edges, self.batch_size), desc="   Filtering"):
                end = min(i + self.batch_size, total_edges)
                
                # Lấy batch cạnh
                batch_rows = rows[i:end]
                batch_cols = cols[i:end]
                
                # Lấy embedding của node nguồn và đích
                emb_src = embeddings[batch_rows]
                emb_dst = embeddings[batch_cols]
                
                # Tính Similarity (Dot Product)
                # Vì đã normalize nên đây chính là Cosine Similarity (-1.0 đến 1.0)
                sim_scores = torch.sum(emb_src * emb_dst, dim=1)
                
                # Quyết định: Giữ lại nếu Sim > Threshold
                batch_mask = sim_scores > self.threshold
                keep_mask.append(batch_mask)

        # Gộp kết quả các batch
        final_mask = torch.cat(keep_mask)
        
        # 5. Tái tạo ma trận kề mới
        refined_rows = rows[final_mask]
        refined_cols = cols[final_mask]
        
        # Tạo ma trận Dense [N, N] toàn số 0
        new_adj = torch.zeros((num_nodes, num_nodes), device=self.device)
        # Điền số 1 vào các cạnh được giữ lại
        new_adj[refined_rows, refined_cols] = 1.0
        
        removed = total_edges - len(refined_rows)
        if total_edges > 0:
            percent = (removed / total_edges) * 100
        else:
            percent = 0
            
        print(f" -> Completed: Removed {removed} edges ({percent:.2f}%).")

        # Trả về CPU tensor shape [1, N, N] để khớp với input ban đầu
        return new_adj.unsqueeze(0).cpu()