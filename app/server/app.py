# Author: Darius H. 2023/10/15
# 
import cv2
import os
import torch
import core.checkpoint as checkpoint

from model.CVNet_Rerank_model import CVNet_Rerank
from modules.reranking.MDescAug import MDescAug
from modules.reranking.RerankwMDA import RerankwMDA
from app.server.img_db import ImageDatabase

# default hyperparameters
RESNET_DEPTH = 50
REDUCTION_DIM = 2048
relup = True
CHECKPOINT_FILE = ''
DATAROOT = 'resorts/forbidden_city'


def init_model():
    model = CVNet_Rerank(RESNET_DEPTH, REDUCTION_DIM, relup)
    model = model.cuda(device=torch.cuda.current_device())
    checkpoint.load_checkpoint(CHECKPOINT_FILE, model)
    return model

@torch.no_grad()
def rank(query_img):
    model = init_model()
    db = ImageDatabase(model, 'resorts/forbidden_city')
    MDescAug_obj = MDescAug()
    RerankwMDA_obj = RerankwMDA()
    
    # Query and key image feature extraction
    Q = db.extract_img_feat(query_img)
    X = db.extract_all_img_feats()
    
    Q = torch.tensor(Q).cuda()
    X = torch.tensor(X).cuda()
    
    # Global retrieval 
    sim = torch.matmul(X, Q.T) # 6322 70
    ranks = torch.argsort(-sim, axis=0) # 6322 70
    
    # Rerank
    rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba = MDescAug_obj(X, Q, ranks)
    ranks = RerankwMDA_obj(ranks, rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba)
    ranks = ranks.data.cpu().numpy()
    
    return ranks

def server(query_img, geo_location):
    """
    Toy serving logics.
    """
    if geo_location == 'forbidden_city':
        pass
    elif geo_location == 'changcheng':
        pass
    else:
        pass
    
    ranks = rank(query_img)
    ans = ranks[-1]
    return ans

def test():
    img_path = os.path.join(os.path.dirname(__file__), 'resorts/forbidden_city/query.jpeg')
    query_img = cv2.imread(img_path)
    ranks = rank(query_img)
    print(ranks.shape)
    
test()